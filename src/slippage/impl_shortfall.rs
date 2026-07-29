//! Implementation Shortfall Algorithm - minimizing impact vs timing risk.
//! 
//! Implementation Shortfall (IS) measures the difference between:
//! - The decision price (when we decided to trade)
//! - The actual execution price (including all costs)
//! 
//! This module implements an IS minimization algorithm that balances:
//! - Market impact (trading too fast moves the market)
//! - Timing risk (trading too slow exposes us to price movements)
//! 
//! Uses fixed-point arithmetic and lock-free state tracking.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Maximum number of slices for order execution
const MAX_SLICES: usize = 64;

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Order slice for IS execution
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OrderSlice {
    pub slice_id: u32,
    pub target_size: i64,          // Target size for this slice
    pub executed_size: i64,        // Actually executed size
    pub target_price: i64,         // Target execution price
    pub avg_exec_price: i64,       // Average execution price achieved
    pub shortfall_bps: i64,        // Shortfall in basis points
    pub timestamp_cycles: u64,     // When this slice was executed
    _padding: [u8; 32],            // Pad to 64 bytes
}

impl Default for OrderSlice {
    fn default() -> Self {
        Self {
            slice_id: 0,
            target_size: 0,
            executed_size: 0,
            target_price: 0,
            avg_exec_price: 0,
            shortfall_bps: 0,
            timestamp_cycles: 0,
            _padding: [0; 32],
        }
    }
}

/// IS execution state - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ISState {
    pub order_id: u64,
    pub side: i8,                  // +1 for buy, -1 for sell
    pub total_size: i64,
    pub decision_price: i64,       // Price when decision was made
    pub total_executed: i64,
    pub total_cost: i64,           // Total cost in fixed-point
    pub is_complete: bool,
    _padding: [u8; 35],            // Pad to 64 bytes
}

impl Default for ISState {
    fn default() -> Self {
        Self {
            order_id: 0,
            side: 0,
            total_size: 0,
            decision_price: 0,
            total_executed: 0,
            total_cost: 0,
            is_complete: false,
            _padding: [0; 35],
        }
    }
}

/// Lock-free Implementation Shortfall tracker
#[repr(C)]
pub struct ImplementationShortfall {
    /// Current order state
    state: ISState,
    
    /// Order slices
    slices: [OrderSlice; MAX_SLICES],
    slice_count: AtomicU64,
    current_slice: AtomicU64,
    
    /// Statistics
    total_shortfall_bps: AtomicI64,
    orders_completed: AtomicU64,
    max_shortfall_bps: AtomicI64,
    
    /// Risk parameters
    urgency_factor: AtomicI64,     // Higher = trade faster (0 to FIXED_SCALE)
    risk_aversion: AtomicI64,      // Higher = more conservative
    
    /// Kill switches
    trading_enabled: AtomicBool,
    shortfall_limit_bps: AtomicI64, // Max acceptable shortfall
    
    _padding: [u8; 24],            // Pad to cache line
}

impl ImplementationShortfall {
    /// Create new IS tracker
    pub const fn new() -> Self {
        Self {
            state: ISState::default(),
            slices: [OrderSlice::default(); MAX_SLICES],
            slice_count: AtomicU64::new(0),
            current_slice: AtomicU64::new(0),
            total_shortfall_bps: AtomicI64::new(0),
            orders_completed: AtomicU64::new(0),
            max_shortfall_bps: AtomicI64::new(0),
            urgency_factor: AtomicI64::new(FIXED_SCALE / 2), // Medium urgency
            risk_aversion: AtomicI64::new(FIXED_SCALE / 2),
            trading_enabled: AtomicBool::new(true),
            shortfall_limit_bps: AtomicI64::new(1000), // 10 bps limit
            _padding: [0; 24],
        }
    }
    
    /// Initialize a new IS order
    #[inline(always)]
    pub fn initialize_order(&self, order_id: u64, side: i8, size: i64, decision_price: i64) {
        if !self.trading_enabled.load(Ordering::Acquire) {
            return;
        }
        
        self.state.order_id = order_id;
        self.state.side = side;
        self.state.total_size = size;
        self.state.decision_price = decision_price;
        self.state.total_executed = 0;
        self.state.total_cost = 0;
        self.state.is_complete = false;
        
        // Calculate optimal slicing based on urgency and risk aversion
        let num_slices = self.calculate_optimal_slices(size, decision_price);
        self.slice_count.store(num_slices, Ordering::Release);
        self.current_slice.store(0, Ordering::Release);
        
        // Initialize slices
        let slice_size = size / num_slices as i64;
        for i in 0..num_slices as usize {
            unsafe {
                let slice = self.slices.get_unchecked_mut(i);
                slice.slice_id = i as u32;
                slice.target_size = slice_size;
                slice.executed_size = 0;
                slice.target_price = decision_price;
                slice.avg_exec_price = 0;
                slice.shortfall_bps = 0;
            }
        }
    }
    
    /// Calculate optimal number of slices
    #[inline(always)]
    fn calculate_optimal_slices(&self, size: i64, price: i64) -> u64 {
        let urgency = self.urgency_factor.load(Ordering::Acquire);
        let risk = self.risk_aversion.load(Ordering::Acquire);
        
        // Base slices from size (larger orders need more slices)
        let base_slices = ((size.abs() as f64 / 100_000.0).ceil() as u64).min(MAX_SLICES as u64).max(1);
        
        // Adjust for urgency (higher urgency = fewer slices, trade faster)
        let urgency_adjustment = if urgency > FIXED_SCALE / 2 {
            (FIXED_SCALE * 2 / urgency.max(1)) as u64
        } else {
            (urgency * 2 / FIXED_SCALE.max(1)) as u64
        };
        
        // Adjust for risk aversion (higher risk aversion = more slices)
        let risk_adjustment = (risk * 2 / FIXED_SCALE.max(1)).max(1) as u64;
        
        (base_slices * urgency_adjustment * risk_adjustment).min(MAX_SLICES as u64).max(1)
    }
    
    /// Record an execution for a slice
    #[inline(always)]
    pub fn record_execution(&self, slice_idx: usize, exec_size: i64, exec_price: i64) {
        if slice_idx >= MAX_SLICES {
            return;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        let slice = unsafe { self.slices.get_unchecked_mut(slice_idx) };
        
        // Update slice execution
        let old_size = slice.executed_size;
        let old_cost = slice.avg_exec_price * old_size;
        
        slice.executed_size += exec_size;
        let new_cost = old_cost + (exec_price * exec_size);
        slice.avg_exec_price = new_cost / slice.executed_size.max(1);
        slice.timestamp_cycles = cycles;
        
        // Calculate shortfall for this slice
        // For buys: shortfall = exec_price - decision_price (positive = bad)
        // For sells: shortfall = decision_price - exec_price (positive = bad)
        let shortfall = if self.state.side > 0 {
            exec_price - self.state.decision_price
        } else {
            self.state.decision_price - exec_price
        };
        
        slice.shortfall_bps = (shortfall * 10_000) / self.state.decision_price.abs().max(1);
        
        // Update total state
        self.state.total_executed += exec_size;
        self.state.total_cost += exec_price * exec_size;
        
        // Check if order is complete
        if self.state.total_executed >= self.state.total_size {
            self.state.is_complete = true;
            self.orders_completed.fetch_add(1, Ordering::Relaxed);
            
            // Calculate total shortfall
            let total_shortfall = if self.state.side > 0 {
                self.state.total_cost - (self.state.decision_price * self.state.total_size)
            } else {
                (self.state.decision_price * self.state.total_size) - self.state.total_cost
            };
            
            let total_shortfall_bps = (total_shortfall * 10_000) / 
                (self.state.decision_price.abs() * self.state.total_size).max(1);
            
            self.total_shortfall_bps.store(total_shortfall_bps, Ordering::Release);
            
            // Track maximum
            let max = self.max_shortfall_bps.load(Ordering::Relaxed);
            if total_shortfall_bps > max {
                self.max_shortfall_bps.store(total_shortfall_bps, Ordering::Relaxed);
            }
            
            // Check if shortfall exceeded limit
            if total_shortfall_bps > self.shortfall_limit_bps.load(Ordering::Acquire) {
                self.trading_enabled.store(false, Ordering::Release);
            }
        }
    }
    
    /// Get current slice to execute
    #[inline(always)]
    pub fn get_current_slice(&self) -> Option<(u32, i64)> {
        if self.state.is_complete || !self.trading_enabled.load(Ordering::Acquire) {
            return None;
        }
        
        let current = self.current_slice.load(Ordering::Acquire);
        let count = self.slice_count.load(Ordering::Acquire);
        
        if current >= count {
            return None;
        }
        
        unsafe {
            let slice = self.slices.get_unchecked(current as usize);
            Some((slice.slice_id, slice.target_size - slice.executed_size))
        }
    }
    
    /// Advance to next slice
    #[inline(always)]
    pub fn advance_slice(&self) {
        let current = self.current_slice.load(Ordering::Relaxed);
        let count = self.slice_count.load(Ordering::Relaxed);
        
        if current < count - 1 {
            self.current_slice.store(current + 1, Ordering::Release);
        }
    }
    
    /// Get current shortfall in bps
    #[inline(always)]
    pub fn current_shortfall_bps(&self) -> i64 {
        if self.state.total_executed == 0 {
            return 0;
        }
        
        let avg_price = self.state.total_cost / self.state.total_executed;
        let shortfall = if self.state.side > 0 {
            avg_price - self.state.decision_price
        } else {
            self.state.decision_price - avg_price
        };
        
        (shortfall * 10_000) / self.state.decision_price.abs().max(1)
    }
    
    /// Check if trading should halt due to shortfall
    #[inline(always)]
    pub fn should_halt(&self) -> bool {
        !self.trading_enabled.load(Ordering::Acquire) ||
        self.current_shortfall_bps() > self.shortfall_limit_bps.load(Ordering::Acquire)
    }
    
    /// Reset trading enabled flag
    #[inline(always)]
    pub fn reset_trading(&self) {
        self.trading_enabled.store(true, Ordering::Release);
    }
    
    /// Set urgency factor
    #[inline(always)]
    pub fn set_urgency(&self, urgency: i64) {
        self.urgency_factor.store(urgency.clamp(0, FIXED_SCALE), Ordering::Release);
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<OrderSlice>() == 64);
        assert!(core::mem::size_of::<ISState>() == 64);
        assert!(core::mem::size_of::<ImplementationShortfall>() % 64 == 0);
    }
    
    #[test]
    fn test_is_initialization() {
        let is_tracker = ImplementationShortfall::new();
        
        is_tracker.initialize_order(12345, 1, 100_000, 100_000_000);
        
        assert_eq!(is_tracker.state.order_id, 12345);
        assert_eq!(is_tracker.state.total_size, 100_000);
        assert!(!is_tracker.state.is_complete);
    }
}
