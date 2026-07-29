//! Reduce-Only Logic and Position Netting Validators
//! 
//! Implements reduce-only order validation and position netting to prevent
//! accidental position flipping. Uses lock-free atomic operations and
//! fixed-point arithmetic throughout.
//! 
//! All state is cache-line aligned and pre-allocated.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, Ordering};

/// Fixed-point quantity (scaled by 1e8)
pub type FixedQty = i64;
/// Fixed-point price (scaled by 1e9)
pub type FixedPrice = i64;

/// Position side
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Flat = 0,
    Long = 1,
    Short = 2,
}

/// Cache-line aligned position state (64 bytes)
#[repr(C, align(64))]
pub struct PositionState {
    pub symbol_id: u32,
    pub side: AtomicU8,
    pub quantity: AtomicI64, // Positive = long, negative = short
    pub entry_price: AtomicI64,
    pub unrealized_pnl: AtomicI64,
    pub realized_pnl: AtomicI64,
    pub max_position: AtomicI64,
    pub reduce_only_flag: AtomicU8,
    _padding: [u8; 16],
}

impl PositionState {
    pub const fn new(symbol_id: u32, max_pos: FixedQty) -> Self {
        Self {
            symbol_id,
            side: AtomicU8::new(PositionSide::Flat as u8),
            quantity: AtomicI64::new(0),
            entry_price: AtomicI64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            realized_pnl: AtomicI64::new(0),
            max_position: AtomicI64::new(max_pos),
            reduce_only_flag: AtomicU8::new(0),
            _padding: [0u8; 16],
        }
    }

    #[inline]
    pub fn get_side(&self) -> PositionSide {
        let qty = self.quantity.load(Ordering::Acquire);
        if qty > 0 {
            PositionSide::Long
        } else if qty < 0 {
            PositionSide::Short
        } else {
            PositionSide::Flat
        }
    }

    #[inline]
    pub fn get_quantity(&self) -> FixedQty {
        self.quantity.load(Ordering::Acquire)
    }

    /// Validate reduce-only order: can only decrease position size
    #[inline]
    pub fn validate_reduce_only(&self, order_side: u8, order_qty: FixedQty) -> bool {
        let current_side = self.get_side();
        let current_qty = self.get_quantity();
        
        match current_side {
            PositionSide::Flat => false, // Cannot open with reduce-only
            PositionSide::Long => {
                // Must be sell order, and qty <= current long position
                order_side == 1 && order_qty <= current_qty
            }
            PositionSide::Short => {
                // Must be buy order, and qty <= abs(current short position)
                order_side == 0 && order_qty <= -current_qty
            }
        }
    }

    /// Calculate max reduce-only quantity for given side
    #[inline]
    pub fn max_reduce_qty(&self, order_side: u8) -> FixedQty {
        let current_qty = self.quantity.load(Ordering::Acquire);
        if order_side == 0 {
            // Buy: can only reduce short position
            if current_qty < 0 { -current_qty } else { 0 }
        } else {
            // Sell: can only reduce long position
            if current_qty > 0 { current_qty } else { 0 }
        }
    }

    /// Update position atomically after fill
    #[inline]
    pub fn update_position(&self, side: u8, qty: FixedQty, price: FixedPrice) -> FixedQty {
        let fill_signed = if side == 0 { qty } else { -qty };
        let old_qty = self.quantity.fetch_add(fill_signed, Ordering::AcqRel);
        let new_qty = old_qty + fill_signed;
        
        // Update side flag
        let new_side = if new_qty > 0 {
            PositionSide::Long as u8
        } else if new_qty < 0 {
            PositionSide::Short as u8
        } else {
            PositionSide::Flat as u8
        };
        self.side.store(new_side, Ordering::Release);
        
        // Calculate PnL if reducing
        let mut pnl: FixedQty = 0;
        if (old_qty > 0 && fill_signed < 0) || (old_qty < 0 && fill_signed > 0) {
            // Reducing position - calculate realized PnL
            let reduce_qty = if old_qty > 0 {
                core::cmp::min(old_qty, -fill_signed)
            } else {
                core::cmp::min(-old_qty, fill_signed)
            };
            let entry = self.entry_price.load(Ordering::Acquire);
            pnl = if side == 1 {
                // Sell: (price - entry) * qty
                ((price - entry) * reduce_qty) / 1_000_000_000
            } else {
                // Buy: (entry - price) * qty
                ((entry - price) * reduce_qty) / 1_000_000_000
            };
            self.realized_pnl.fetch_add(pnl, Ordering::AcqRel);
        }
        
        // Update average entry if increasing position
        if (old_qty >= 0 && fill_signed > 0) || (old_qty <= 0 && fill_signed < 0) {
            let total_qty = old_qty.abs() + qty;
            let old_entry = self.entry_price.load(Ordering::Acquire);
            let new_entry = if total_qty > 0 {
                ((old_entry * old_qty.abs()) + (price * qty)) / total_qty
            } else {
                price
            };
            self.entry_price.store(new_entry, Ordering::Release);
        }
        
        pnl
    }

    /// Check if order would flip position (not allowed in reduce-only mode)
    #[inline]
    pub fn would_flip_position(&self, order_side: u8, order_qty: FixedQty) -> bool {
        let current_qty = self.quantity.load(Ordering::Acquire);
        
        if order_side == 0 {
            // Buy order
            current_qty < 0 && order_qty > -current_qty
        } else {
            // Sell order
            current_qty > 0 && order_qty > current_qty
        }
    }
}

/// Position Netting Engine - handles multi-symbol portfolio netting
#[repr(C, align(64))]
pub struct PositionNettingEngine {
    pub positions_ptr: *mut PositionState,
    pub num_positions: usize,
    pub netting_mode: AtomicU8, // 0 = no netting, 1 = per-symbol, 2 = portfolio
    pub total_exposure: AtomicI64,
    pub max_portfolio_exposure: AtomicI64,
    _padding: [u8; 32],
}

impl PositionNettingEngine {
    pub const fn new(ptr: *mut PositionState, num: usize, max_exposure: FixedQty) -> Self {
        Self {
            positions_ptr: ptr,
            num_positions: num,
            netting_mode: AtomicU8::new(0),
            total_exposure: AtomicI64::new(0),
            max_portfolio_exposure: AtomicI64::new(max_exposure),
            _padding: [0u8; 32],
        }
    }

    /// Calculate total portfolio exposure (sum of absolute positions)
    #[inline]
    pub fn calculate_total_exposure(&self) -> FixedQty {
        let mut total: FixedQty = 0;
        unsafe {
            for i in 0..self.num_positions {
                let pos = &*self.positions_ptr.add(i);
                total += pos.quantity.load(Ordering::Acquire).abs();
            }
        }
        self.total_exposure.store(total, Ordering::Release);
        total
    }

    /// Validate order against portfolio limits
    #[inline]
    pub fn validate_portfolio_limit(&self, additional_exposure: FixedQty) -> bool {
        let current = self.total_exposure.load(Ordering::Acquire);
        let max = self.max_portfolio_exposure.load(Ordering::Acquire);
        current + additional_exposure <= max
    }

    /// Set netting mode atomically
    #[inline]
    pub fn set_netting_mode(&self, mode: u8) {
        self.netting_mode.store(mode, Ordering::Release);
    }
}

/// Reduce-Only Order Validator with circuit breaker
#[repr(C, align(64))]
pub struct ReduceOnlyValidator {
    pub enabled: AtomicU8,
    pub violation_count: AtomicU64,
    pub max_violations: AtomicU64,
    pub circuit_open: AtomicU8,
    _padding: [u8; 40],
}

impl ReduceOnlyValidator {
    pub const fn new(max_violations: u64) -> Self {
        Self {
            enabled: AtomicU8::new(1),
            violation_count: AtomicU64::new(0),
            max_violations: AtomicU64::new(max_violations),
            circuit_open: AtomicU8::new(0),
            _padding: [0u8; 40],
        }
    }

    /// Validate reduce-only order with circuit breaker
    #[inline]
    pub fn validate(&self, position: &PositionState, order_side: u8, order_qty: FixedQty) -> bool {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return true; // Disabled
        }
        
        if self.circuit_open.load(Ordering::Acquire) != 0 {
            return false; // Circuit open
        }
        
        if !position.validate_reduce_only(order_side, order_qty) {
            let count = self.violation_count.fetch_add(1, Ordering::AcqRel) + 1;
            let max = self.max_violations.load(Ordering::Acquire);
            
            if count >= max {
                self.circuit_open.store(1, Ordering::Release);
            }
            return false;
        }
        
        true
    }

    /// Reset circuit breaker
    #[inline]
    pub fn reset_circuit(&self) {
        self.violation_count.store(0, Ordering::Release);
        self.circuit_open.store(0, Ordering::Release);
    }

    /// Toggle enable/disable
    #[inline]
    pub fn toggle(&self) {
        let current = self.enabled.load(Ordering::Acquire);
        self.enabled.store(1 - current, Ordering::Release);
    }
}

/// Partial Fill Handler for reduce-only orders
#[repr(C, align(64))]
pub struct PartialFillHandler {
    pub original_qty: AtomicI64,
    pub filled_qty: AtomicI64,
    pub remaining_allowed: AtomicI64,
    pub position_at_start: AtomicI64,
    _padding: [u8; 32],
}

impl PartialFillHandler {
    pub const fn new() -> Self {
        Self {
            original_qty: AtomicI64::new(0),
            filled_qty: AtomicI64::new(0),
            remaining_allowed: AtomicI64::new(0),
            position_at_start: AtomicI64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Initialize for new reduce-only order
    #[inline]
    pub fn init(&self, order_qty: FixedQty, position_qty: FixedQty) {
        self.original_qty.store(order_qty, Ordering::Release);
        self.filled_qty.store(0, Ordering::Release);
        self.remaining_allowed.store(position_qty.abs(), Ordering::Release);
        self.position_at_start.store(position_qty, Ordering::Release);
    }

    /// Update after partial fill, recalculate remaining allowed
    #[inline]
    pub fn update_partial(&self, fill_qty: FixedQty) -> FixedQty {
        let filled = self.filled_qty.fetch_add(fill_qty, Ordering::AcqRel) + fill_qty;
        let start_pos = self.position_at_start.load(Ordering::Acquire);
        
        // Recalculate based on current position (which may have changed)
        let current_remaining = start_pos.abs() - filled;
        let allowed = core::cmp::max(0, current_remaining);
        self.remaining_allowed.store(allowed, Ordering::Release);
        
        allowed
    }

    /// Check if more fills are allowed
    #[inline]
    pub fn can_fill_more(&self) -> bool {
        self.remaining_allowed.load(Ordering::Acquire) > 0
    }

    /// Get max additional fill quantity
    #[inline]
    pub fn max_additional_fill(&self) -> FixedQty {
        self.remaining_allowed.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_state_creation() {
        let pos = PositionState::new(1, 10_000_000_000i64);
        assert_eq!(pos.get_side(), PositionSide::Flat);
        assert_eq!(pos.get_quantity(), 0);
    }

    #[test]
    fn test_position_updates() {
        let pos = PositionState::new(1, 10_000_000_000i64);
        
        // Open long position
        let pnl = pos.update_position(0, 1_000_000_000i64, 50_000_000_000i64);
        assert_eq!(pos.get_side(), PositionSide::Long);
        assert_eq!(pos.get_quantity(), 1_000_000_000i64);
        assert_eq!(pnl, 0); // No PnL on opening
        
        // Partially reduce
        let pnl = pos.update_position(1, 500_000_000i64, 51_000_000_000i64);
        assert_eq!(pos.get_quantity(), 500_000_000i64);
        assert!(pnl > 0); // Realized profit
    }

    #[test]
    fn test_reduce_only_validation() {
        let pos = PositionState::new(1, 10_000_000_000i64);
        pos.update_position(0, 1_000_000_000i64, 50_000_000_000i64);
        
        // Valid reduce-only sell
        assert!(pos.validate_reduce_only(1, 500_000_000i64));
        
        // Invalid: too large
        assert!(!pos.validate_reduce_only(1, 1_500_000_000i64));
        
        // Invalid: wrong side (buy would increase)
        assert!(!pos.validate_reduce_only(0, 500_000_000i64));
    }

    #[test]
    fn test_position_flip_detection() {
        let pos = PositionState::new(1, 10_000_000_000i64);
        pos.update_position(0, 1_000_000_000i64, 50_000_000_000i64);
        
        // This would flip to short
        assert!(pos.would_flip_position(1, 1_500_000_000i64));
        
        // This would not flip
        assert!(!pos.would_flip_position(1, 500_000_000i64));
    }

    #[test]
    fn test_reduce_only_validator_circuit() {
        let validator = ReduceOnlyValidator::new(3);
        let pos = PositionState::new(1, 10_000_000_000i64);
        
        // First violations
        assert!(!validator.validate(&pos, 0, 1_000_000_000i64)); // Can't reduce flat
        assert!(!validator.validate(&pos, 0, 1_000_000_000i64));
        assert!(!validator.validate(&pos, 0, 1_000_000_000i64));
        
        // Circuit should be open now
        assert_eq!(validator.circuit_open.load(Ordering::Acquire), 1);
        assert!(!validator.validate(&pos, 1, 1_000_000_000i64)); // Even valid ones rejected
    }

    #[test]
    fn test_partial_fill_handler() {
        let handler = PartialFillHandler::new();
        handler.init(2_000_000_000i64, 1_500_000_000i64);
        
        assert!(handler.can_fill_more());
        assert_eq!(handler.max_additional_fill(), 1_500_000_000i64);
        
        let remaining = handler.update_partial(1_000_000_000i64);
        assert_eq!(remaining, 500_000_000i64);
        
        assert!(handler.can_fill_more());
        
        handler.update_partial(500_000_000i64);
        assert!(!handler.can_fill_more());
    }

    #[test]
    fn test_max_reduce_qty() {
        let pos = PositionState::new(1, 10_000_000_000i64);
        
        // Flat position
        assert_eq!(pos.max_reduce_qty(0), 0);
        assert_eq!(pos.max_reduce_qty(1), 0);
        
        // Long position
        pos.update_position(0, 1_000_000_000i64, 50_000_000_000i64);
        assert_eq!(pos.max_reduce_qty(0), 0); // Can't reduce long with buy
        assert_eq!(pos.max_reduce_qty(1), 1_000_000_000i64);
        
        // Short position
        pos.update_position(1, 2_000_000_000i64, 49_000_000_000i64);
        assert_eq!(pos.max_reduce_qty(0), 1_000_000_000i64);
        assert_eq!(pos.max_reduce_qty(1), 0);
    }
}
