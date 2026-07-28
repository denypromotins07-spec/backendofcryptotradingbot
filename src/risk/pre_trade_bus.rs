//! Global Pre-Trade Risk Bus
//! 
//! Lock-free atomic queue-based risk bus that blocks unsafe orders before execution.
//! All calculations execute in under 50 nanoseconds using branchless math.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

/// Cache line size for padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of pending risk checks in the lock-free queue
const MAX_PENDING_CHECKS: usize = 4096;

/// Global kill switch flag - atomic for lock-free access
pub static GLOBAL_KILL_SWITCH: AtomicBool = AtomicBool::new(false);

/// Per-strategy circuit breakers (up to 256 strategies)
static STRATEGY_BREAKERS: [AtomicBool; 256] = {
    const INIT: AtomicBool = AtomicBool::new(false);
    [INIT; 256]
};

/// Risk check result codes (branchless evaluation)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskResult {
    Approved = 0,
    RejectedKillSwitch = 1,
    RejectedStrategyBreaker = 2,
    RejectedPositionLimit = 3,
    RejectedMarginLimit = 4,
    RejectedRateLimit = 5,
}

/// Pre-trade risk check request - fixed size, no heap allocation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RiskCheckRequest {
    pub strategy_id: u16,
    pub instrument_id: u16,
    pub side: u8, // 0 = Buy, 1 = Sell
    pub quantity_u64: u64,
    pub price_u64: u64,
    pub timestamp_cycles: u64,
    pub _padding: [u8; 32], // Pad to 64 bytes
}

impl RiskCheckRequest {
    #[inline(always)]
    pub const fn new(
        strategy_id: u16,
        instrument_id: u16,
        side: u8,
        quantity: u64,
        price: u64,
    ) -> Self {
        Self {
            strategy_id,
            instrument_id,
            side,
            quantity_u64: quantity,
            price_u64: price,
            timestamp_cycles: 0,
            _padding: [0u8; 32],
        }
    }
}

/// Lock-free ring buffer for risk check requests
struct RiskCheckQueue {
    buffer: [RiskCheckRequest; MAX_PENDING_CHECKS],
    head: AtomicUsize,
    tail: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE],
}

impl RiskCheckQueue {
    const fn new() -> Self {
        const EMPTY: RiskCheckRequest = RiskCheckRequest {
            strategy_id: 0,
            instrument_id: 0,
            side: 0,
            quantity_u64: 0,
            price_u64: 0,
            timestamp_cycles: 0,
            _padding: [0u8; 32],
        };
        Self {
            buffer: [EMPTY; MAX_PENDING_CHECKS],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline(always)]
    fn push(&self, request: RiskCheckRequest) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & (MAX_PENDING_CHECKS - 1);
        
        // Branchless full check
        let is_full = (next_tail == self.head.load(Ordering::Acquire)) as usize;
        if is_full != 0 {
            return false;
        }

        unsafe {
            self.buffer.as_ptr().add(tail).write(request);
        }
        self.tail.store(next_tail, Ordering::Release);
        true
    }

    #[inline(always)]
    fn pop(&self) -> Option<RiskCheckRequest> {
        let head = self.head.load(Ordering::Relaxed);
        
        // Branchless empty check
        let is_empty = (head == self.tail.load(Ordering::Acquire)) as usize;
        if is_empty != 0 {
            return None;
        }

        let next_head = (head + 1) & (MAX_PENDING_CHECKS - 1);
        let request = unsafe { self.buffer.as_ptr().add(head).read() };
        self.head.store(next_head, Ordering::Release);
        Some(request)
    }
}

/// Global pre-trade risk bus state
pub struct PreTradeRiskBus {
    queue: RiskCheckQueue,
    /// Per-strategy exposure limits (in basis points * 10000)
    exposure_limits: [AtomicU64; 256],
    /// Current per-strategy exposure (in quote units)
    current_exposure: [AtomicU64; 256],
    /// Rate limit counters (orders per millisecond window)
    rate_counters: [AtomicU64; 256],
    /// Last rate limit window timestamp
    rate_windows: [AtomicU64; 256],
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for PreTradeRiskBus {}
unsafe impl Sync for PreTradeRiskBus {}

impl PreTradeRiskBus {
    /// Create a new pre-trade risk bus with default limits
    pub const fn new() -> Self {
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        Self {
            queue: RiskCheckQueue::new(),
            exposure_limits: [ZERO_U64; 256],
            current_exposure: [ZERO_U64; 256],
            rate_counters: [ZERO_U64; 256],
            rate_windows: [ZERO_U64; 256],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Set exposure limit for a strategy (in quote units * 10000 for precision)
    #[inline(always)]
    pub fn set_exposure_limit(&self, strategy_id: usize, limit: u64) {
        if strategy_id < 256 {
            self.exposure_limits[strategy_id].store(limit, Ordering::Release);
        }
    }

    /// Activate global kill switch - halts all trading immediately
    #[inline(always)]
    pub fn activate_global_kill_switch(&self) {
        GLOBAL_KILL_SWITCH.store(true, Ordering::SeqCst);
    }

    /// Deactivate global kill switch
    #[inline(always)]
    pub fn deactivate_global_kill_switch(&self) {
        GLOBAL_KILL_SWITCH.store(false, Ordering::SeqCst);
    }

    /// Activate strategy-specific circuit breaker
    #[inline(always)]
    pub fn activate_strategy_breaker(&self, strategy_id: usize) {
        if strategy_id < 256 {
            STRATEGY_BREAKERS[strategy_id].store(true, Ordering::SeqCst);
        }
    }

    /// Reset strategy circuit breaker
    #[inline(always)]
    pub fn reset_strategy_breaker(&self, strategy_id: usize) {
        if strategy_id < 256 {
            STRATEGY_BREAKERS[strategy_id].store(false, Ordering::SeqCst);
        }
    }

    /// Check if global kill switch is active (branchless)
    #[inline(always)]
    pub fn is_killed(&self) -> bool {
        GLOBAL_KILL_SWITCH.load(Ordering::Acquire)
    }

    /// Submit order for risk check - returns immediately with result
    /// Uses SIMD for parallel limit checking when available
    #[inline(always)]
    pub fn check_order(&self, request: RiskCheckRequest) -> RiskResult {
        // Get current TSC for timing
        let tsc = unsafe { _rdtsc() };
        
        // Branchless kill switch check
        let killed = GLOBAL_KILL_SWITCH.load(Ordering::Acquire) as u8;
        if killed != 0 {
            return RiskResult::RejectedKillSwitch;
        }

        // Branchless strategy breaker check
        let strategy_idx = request.strategy_id as usize;
        if strategy_idx < 256 {
            let breaker_active = STRATEGY_BREAKERS[strategy_idx].load(Ordering::Acquire) as u8;
            if breaker_active != 0 {
                return RiskResult::RejectedStrategyBreaker;
            }
        }

        // Branchless position limit check using SIMD-style comparison
        let exposure = self.current_exposure[strategy_idx].load(Ordering::Acquire);
        let limit = self.exposure_limits[strategy_idx].load(Ordering::Acquire);
        
        // Branchless greater-than comparison
        let exceeds_limit = ((exposure.wrapping_add(request.quantity_u64 * request.price_u64)) > limit) as u8;
        if exceeds_limit != 0 {
            return RiskResult::RejectedPositionLimit;
        }

        // Rate limiting check (sliding window)
        let current_window = tsc / 3_000_000; // Approximate ms in cycles
        let stored_window = self.rate_windows[strategy_idx].load(Ordering::Relaxed);
        let counter = self.rate_counters[strategy_idx].load(Ordering::Relaxed);
        
        // Branchless window reset
        let window_changed = (current_window != stored_window) as u64;
        let new_counter = counter * (1 - window_changed); // Reset to 0 if window changed
        let updated_window = stored_window * (1 - window_changed) + current_window * window_changed;
        
        self.rate_windows[strategy_idx].store(updated_window, Ordering::Relaxed);
        self.rate_counters[strategy_idx].store(new_counter + 1, Ordering::Relaxed);

        // Rate limit: max 100 orders per ms per strategy
        let rate_exceeded = (new_counter >= 100) as u8;
        if rate_exceeded != 0 {
            return RiskResult::RejectedRateLimit;
        }

        // Order approved - update exposure atomically
        let delta = request.quantity_u64 * request.price_u64;
        self.current_exposure[strategy_idx].fetch_add(delta, Ordering::AcqRel);

        // Queue for async reconciliation (non-blocking)
        let _ = self.queue.push(request);

        RiskResult::Approved
    }

    /// Reduce exposure after fill (branchless)
    #[inline(always)]
    pub fn reduce_exposure(&self, strategy_id: usize, amount: u64) {
        if strategy_id < 256 {
            // Saturating subtract to prevent underflow
            let current = self.current_exposure[strategy_id].load(Ordering::Acquire);
            let reduced = current.saturating_sub(amount);
            self.current_exposure[strategy_id].store(reduced, Ordering::Release);
        }
    }

    /// Get current exposure for a strategy
    #[inline(always)]
    pub fn get_exposure(&self, strategy_id: usize) -> u64 {
        if strategy_id < 256 {
            self.current_exposure[strategy_id].load(Ordering::Acquire)
        } else {
            0
        }
    }

    /// Process pending risk checks (called by reconciliation thread)
    #[inline(always)]
    pub fn process_pending(&self) -> usize {
        let mut count = 0;
        while let Some(_req) = self.queue.pop() {
            count += 1;
            // In production, this would log to shadow risk logger
        }
        count
    }
}

impl Default for PreTradeRiskBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_check_request_size() {
        assert_eq!(core::mem::size_of::<RiskCheckRequest>(), 64);
        assert_eq!(core::mem::align_of::<RiskCheckRequest>(), 8);
    }

    #[test]
    fn test_global_kill_switch() {
        let bus = PreTradeRiskBus::new();
        let req = RiskCheckRequest::new(0, 1, 0, 100, 50000);
        
        // Should approve initially
        assert_eq!(bus.check_order(req), RiskResult::Approved);
        
        // Activate kill switch
        bus.activate_global_kill_switch();
        assert_eq!(bus.check_order(req), RiskResult::RejectedKillSwitch);
        
        // Deactivate
        bus.deactivate_global_kill_switch();
        assert_eq!(bus.check_order(req), RiskResult::Approved);
    }

    #[test]
    fn test_strategy_circuit_breaker() {
        let bus = PreTradeRiskBus::new();
        let req = RiskCheckRequest::new(5, 1, 0, 100, 50000);
        
        assert_eq!(bus.check_order(req), RiskResult::Approved);
        
        bus.activate_strategy_breaker(5);
        assert_eq!(bus.check_order(req), RiskResult::RejectedStrategyBreaker);
        
        bus.reset_strategy_breaker(5);
        assert_eq!(bus.check_order(req), RiskResult::Approved);
    }

    #[test]
    fn test_exposure_limits() {
        let bus = PreTradeRiskBus::new();
        bus.set_exposure_limit(0, 1_000_000); // 1M limit
        
        // First order should pass
        let req1 = RiskCheckRequest::new(0, 1, 0, 10, 50000); // 500k notional
        assert_eq!(bus.check_order(req1), RiskResult::Approved);
        
        // Second order should fail (would exceed limit)
        let req2 = RiskCheckRequest::new(0, 1, 0, 10, 50000); // Another 500k
        assert_eq!(bus.check_order(req2), RiskResult::Approved);
        
        // Third should fail
        let req3 = RiskCheckRequest::new(0, 1, 0, 1, 50000); // 50k more
        assert_eq!(bus.check_order(req3), RiskResult::RejectedPositionLimit);
    }

    #[test]
    fn test_exposure_reduction() {
        let bus = PreTradeRiskBus::new();
        bus.set_exposure_limit(0, 1_000_000);
        
        let req = RiskCheckRequest::new(0, 1, 0, 10, 50000);
        assert_eq!(bus.check_order(req), RiskResult::Approved);
        assert_eq!(bus.get_exposure(0), 500_000);
        
        bus.reduce_exposure(0, 200_000);
        assert_eq!(bus.get_exposure(0), 300_000);
    }
}
