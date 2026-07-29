//! Advanced Order Types & Execution Nuances
//! 
//! Implements Post-only, IOC, FOK, GTT, and hidden order state machines
//! with strict validation using fixed-point arithmetic and zero-copy semantics.
//! 
//! All state machines are lock-free and use atomic flags for instant transitions.
//! Memory is pre-allocated; no heap usage in hot paths.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use core::ptr;

/// Fixed-point price type (scaled by 1e9)
pub type FixedPrice = i64;
/// Fixed-point quantity type (scaled by 1e8)
pub type FixedQty = i64;
/// Timestamp in nanoseconds
pub type TimestampNs = u64;

/// Order type flags - bitfield for compact storage
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderTypeFlag {
    Limit = 0b0000_0001,
    Market = 0b0000_0010,
    PostOnly = 0b0000_0100,
    IOC = 0b0000_1000,
    FOK = 0b0001_0000,
    GTT = 0b0010_0000,
    Hidden = 0b0100_0000,
    ReduceOnly = 0b1000_0000,
}

/// Order state machine states
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    New = 0,
    PendingNew = 1,
    PartiallyFilled = 2,
    Filled = 3,
    Cancelled = 4,
    Rejected = 5,
    Expired = 6,
}

/// Cache-line aligned order header (64 bytes)
#[repr(C, align(64))]
pub struct OrderHeader {
    pub order_id: AtomicU64,
    pub symbol_id: u32,
    pub side: u8, // 0 = buy, 1 = sell
    pub order_type_flags: u8,
    pub state: AtomicU8,
    pub time_in_force: u8,
    pub padding0: u8,
    pub creation_ts: AtomicU64,
    pub expiry_ts: AtomicU64,
    pub price: FixedPrice,
    pub original_qty: FixedQty,
    pub remaining_qty: AtomicU64, // stored as fixed-point
    pub filled_qty: AtomicU64,
    pub hidden_qty: FixedQty,
    pub visible_qty: FixedQty,
    pub client_order_id: u64,
    pub sequence_num: u64,
    _padding: [u8; 8], // Ensure 64-byte alignment
}

impl OrderHeader {
    /// Create a new order header with pre-validated parameters
    #[inline]
    pub const fn new(
        order_id: u64,
        symbol_id: u32,
        side: u8,
        order_type_flags: u8,
        price: FixedPrice,
        qty: FixedQty,
        expiry_ts: u64,
        client_order_id: u64,
    ) -> Self {
        Self {
            order_id: AtomicU64::new(order_id),
            symbol_id,
            side,
            order_type_flags,
            state: AtomicU8::new(OrderState::New as u8),
            time_in_force: 0,
            padding0: 0,
            creation_ts: AtomicU64::new(unsafe { core::arch::x86_64::_rdtsc() } as u64),
            expiry_ts: AtomicU64::new(expiry_ts),
            price,
            original_qty: qty,
            remaining_qty: AtomicU64::new(qty as u64),
            filled_qty: AtomicU64::new(0),
            hidden_qty: if order_type_flags & OrderTypeFlag::Hidden as u8 != 0 { qty } else { 0 },
            visible_qty: if order_type_flags & OrderTypeFlag::Hidden as u8 != 0 { 0 } else { qty },
            client_order_id,
            sequence_num: 0,
            _padding: [0u8; 8],
        }
    }

    /// Validate order constraints without allocation
    #[inline]
    pub fn validate(&self) -> bool {
        let flags = self.order_type_flags;
        
        // Post-only cannot be FOK or IOC
        if flags & OrderTypeFlag::PostOnly as u8 != 0 {
            if flags & (OrderTypeFlag::FOK as u8 | OrderTypeFlag::IOC as u8) != 0 {
                return false;
            }
        }
        
        // FOK must have full fill or reject
        if flags & OrderTypeFlag::FOK as u8 != 0 {
            if flags & OrderTypeFlag::Limit as u8 == 0 {
                return false;
            }
        }
        
        // Hidden orders must have valid hidden qty
        if flags & OrderTypeFlag::Hidden as u8 != 0 {
            if self.hidden_qty <= 0 || self.visible_qty < 0 {
                return false;
            }
        }
        
        true
    }

    /// Transition state atomically
    #[inline]
    pub fn transition_state(&self, new_state: OrderState) -> bool {
        let current = self.state.load(Ordering::Acquire);
        let valid_transition = match (current, new_state) {
            (0, 1) | (0, 4) | (0, 5) => true, // New -> PendingNew/Cancelled/Rejected
            (1, 2) | (1, 3) | (1, 4) | (1, 5) => true, // PendingNew -> Partial/Filled/Cancel/Reject
            (2, 3) | (2, 4) => true, // PartiallyFilled -> Filled/Cancelled
            _ => false,
        };
        
        if !valid_transition {
            return false;
        }
        
        self.state.store(new_state as u8, Ordering::Release);
        true
    }

    /// Update remaining quantity atomically (lock-free)
    #[inline]
    pub fn update_remaining(&self, fill_qty: FixedQty) -> FixedQty {
        let mut current = self.remaining_qty.load(Ordering::Acquire) as FixedQty;
        loop {
            let new_remaining = current - fill_qty;
            if new_remaining < 0 {
                break current; // Should not happen with proper validation
            }
            match self.remaining_qty.compare_exchange_weak(
                current as u64,
                new_remaining as u64,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let filled = self.filled_qty.load(Ordering::Acquire) as FixedQty + fill_qty;
                    self.filled_qty.store(filled as u64, Ordering::Release);
                    return new_remaining;
                }
                Err(curr) => current = curr as FixedQty,
            }
        }
    }
}

/// Post-Only Order Validator
pub struct PostOnlyValidator {
    pub best_bid: AtomicU64,
    pub best_ask: AtomicU64,
    pub last_update_ts: AtomicU64,
}

impl PostOnlyValidator {
    pub const fn new() -> Self {
        Self {
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(i64::MAX as u64),
            last_update_ts: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn validate_post_only(&self, price: FixedPrice, side: u8) -> bool {
        if side == 0 {
            // Buy: must be below best bid
            price < self.best_bid.load(Ordering::Acquire) as FixedPrice
        } else {
            // Sell: must be above best ask
            price > self.best_ask.load(Ordering::Acquire) as FixedPrice
        }
    }

    #[inline]
    pub fn update_book(&self, bid: FixedPrice, ask: FixedPrice) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.best_bid.store(bid as u64, Ordering::Release);
        self.best_ask.store(ask as u64, Ordering::Release);
        self.last_update_ts.store(ts, Ordering::Release);
    }
}

/// IOC/FOK Execution Engine
pub struct ImmediateExecutionEngine {
    pub available_liquidity: AtomicU64,
    pub execution_mask: AtomicU8,
    _cache_pad: [u8; 55],
}

impl ImmediateExecutionEngine {
    pub const fn new() -> Self {
        Self {
            available_liquidity: AtomicU64::new(0),
            execution_mask: AtomicU8::new(0),
            _cache_pad: [0u8; 55],
        }
    }

    /// Check if IOC order can be partially filled
    #[inline]
    pub fn check_ioc(&self, required_qty: FixedQty) -> FixedQty {
        let avail = self.available_liquidity.load(Ordering::Acquire) as FixedQty;
        if avail < required_qty {
            avail // Return what's available (partial)
        } else {
            required_qty
        }
    }

    /// Check if FOK order can be fully filled
    #[inline]
    pub fn check_fok(&self, required_qty: FixedQty) -> bool {
        let avail = self.available_liquidity.load(Ordering::Acquire) as FixedQty;
        avail >= required_qty
    }

    #[inline]
    pub fn update_liquidity(&self, qty: FixedQty) {
        self.available_liquidity.store(qty as u64, Ordering::Release);
    }
}

/// GTT (Good-Til-Time) Expiration Checker
pub struct GTTChecker {
    pub current_ts: AtomicU64,
    pub expiration_threshold: AtomicU64,
    _pad: [u8; 48],
}

impl GTTChecker {
    pub const fn new() -> Self {
        Self {
            current_ts: AtomicU64::new(0),
            expiration_threshold: AtomicU64::new(0),
            _pad: [0u8; 48],
        }
    }

    #[inline]
    pub fn is_expired(&self, expiry_ts: u64) -> bool {
        let now = self.current_ts.load(Ordering::Acquire);
        now >= expiry_ts
    }

    #[inline]
    pub fn update_time(&self, ts: u64) {
        self.current_ts.store(ts, Ordering::Release);
    }
}

/// Hidden Order Tracker - manages visible vs hidden quantities
pub struct HiddenOrderTracker {
    pub total_qty: AtomicU64,
    pub displayed_qty: AtomicU64,
    pub hidden_qty: AtomicU64,
    pub refresh_threshold: AtomicU64,
    pub fills_since_refresh: AtomicU64,
    _pad: [u8; 24],
}

impl HiddenOrderTracker {
    pub const fn new(total: FixedQty, display: FixedQty) -> Self {
        Self {
            total_qty: AtomicU64::new(total as u64),
            displayed_qty: AtomicU64::new(display as u64),
            hidden_qty: AtomicU64::new((total - display) as u64),
            refresh_threshold: AtomicU64::new(display as u64),
            fills_since_refresh: AtomicU64::new(0),
            _pad: [0u8; 24],
        }
    }

    /// Refresh displayed quantity when depleted
    #[inline]
    pub fn refresh_display(&self) -> FixedQty {
        let filled = self.fills_since_refresh.load(Ordering::Acquire);
        let threshold = self.refresh_threshold.load(Ordering::Acquire);
        
        if filled >= threshold {
            self.fills_since_refresh.store(0, Ordering::Release);
            let hidden = self.hidden_qty.load(Ordering::Acquire);
            let display = self.displayed_qty.load(Ordering::Acquire);
            
            if hidden > 0 {
                let refresh_amt = core::cmp::min(hidden, display);
                self.hidden_qty.fetch_sub(refresh_amt, Ordering::AcqRel);
                self.displayed_qty.fetch_add(refresh_amt, Ordering::AcqRel);
                refresh_amt as FixedQty
            } else {
                0
            }
        } else {
            0
        }
    }

    #[inline]
    pub fn record_fill(&self, qty: FixedQty) {
        self.fills_since_refresh.fetch_add(qty as u64, Ordering::AcqRel);
        self.total_qty.fetch_sub(qty as u64, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_header_creation() {
        let order = OrderHeader::new(
            12345,
            1,
            0,
            OrderTypeFlag::Limit as u8 | OrderTypeFlag::PostOnly as u8,
            50_000_000_000i64,
            1_000_000_000i64,
            0,
            99999,
        );
        
        assert!(order.validate());
        assert_eq!(order.state.load(Ordering::Acquire), OrderState::New as u8);
    }

    #[test]
    fn test_invalid_order_combinations() {
        let order = OrderHeader::new(
            12346,
            1,
            0,
            OrderTypeFlag::PostOnly as u8 | OrderTypeFlag::FOK as u8,
            50_000_000_000i64,
            1_000_000_000i64,
            0,
            99998,
        );
        
        assert!(!order.validate());
    }

    #[test]
    fn test_state_transitions() {
        let order = OrderHeader::new(
            12347,
            1,
            0,
            OrderTypeFlag::Limit as u8,
            50_000_000_000i64,
            1_000_000_000i64,
            0,
            99997,
        );
        
        assert!(order.transition_state(OrderState::PendingNew));
        assert!(order.transition_state(OrderState::PartiallyFilled));
        assert!(order.transition_state(OrderState::Filled));
        
        // Invalid transition
        assert!(!order.transition_state(OrderState::PendingNew));
    }

    #[test]
    fn test_quantity_updates() {
        let order = OrderHeader::new(
            12348,
            1,
            0,
            OrderTypeFlag::Limit as u8,
            50_000_000_000i64,
            1_000_000_000i64,
            0,
            99996,
        );
        
        let remaining = order.update_remaining(250_000_000i64);
        assert_eq!(remaining, 750_000_000i64);
        assert_eq!(order.filled_qty.load(Ordering::Acquire) as FixedQty, 250_000_000i64);
    }

    #[test]
    fn test_post_only_validator() {
        let validator = PostOnlyValidator::new();
        validator.update_book(49_900_000_000i64, 50_100_000_000i64);
        
        // Buy below best bid - valid
        assert!(validator.validate_post_only(49_800_000_000i64, 0));
        // Buy at or above best bid - invalid
        assert!(!validator.validate_post_only(49_900_000_000i64, 0));
        
        // Sell above best ask - valid
        assert!(validator.validate_post_only(50_200_000_000i64, 1));
        // Sell at or below best ask - invalid
        assert!(!validator.validate_post_only(50_100_000_000i64, 1));
    }

    #[test]
    fn test_ioc_fok_execution() {
        let engine = ImmediateExecutionEngine::new();
        engine.update_liquidity(500_000_000i64);
        
        // IOC partial fill
        assert_eq!(engine.check_ioc(750_000_000i64), 500_000_000i64);
        assert_eq!(engine.check_ioc(250_000_000i64), 250_000_000i64);
        
        // FOK full fill required
        assert!(!engine.check_fok(750_000_000i64));
        assert!(engine.check_fok(500_000_000i64));
    }

    #[test]
    fn test_hidden_order_tracker() {
        let tracker = HiddenOrderTracker::new(1_000_000_000i64, 100_000_000i64);
        
        tracker.record_fill(50_000_000i64);
        assert_eq!(tracker.refresh_display(), 0); // Not yet at threshold
        
        tracker.record_fill(50_000_000i64);
        let refreshed = tracker.refresh_display();
        assert_eq!(refreshed, 100_000_000i64);
    }
}
