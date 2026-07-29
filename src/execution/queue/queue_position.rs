//! Lock-free queue position estimator using FIFO logic.
//! 
//! Estimates our position in the order book queue based on:
//! - Initial queue size when we placed the order
//! - Trade flow ahead of us (depletion)
//! - New orders joining behind us (rare in FIFO, but possible)
//! 
//! Uses fixed-point arithmetic and circular buffers for O(1) updates.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64;

/// Cache line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Maximum queue depth we track
const MAX_QUEUE_DEPTH: usize = 1024;

/// Circular buffer for trade flow tracking
#[repr(C)]
struct TradeFlowBuffer {
    data: [i64; MAX_QUEUE_DEPTH],
    head: AtomicU64,
    tail: AtomicU64,
    sum: AtomicU64, // Running sum for O(1) window calculations
    _padding: [u8; 32], // Pad to cache line
}

impl TradeFlowBuffer {
    const fn new() -> Self {
        Self {
            data: [0; MAX_QUEUE_DEPTH],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            _padding: [0; 32],
        }
    }
    
    #[inline(always)]
    fn push(&self, volume: i64) {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        // Calculate next position
        let next_head = (head + 1) % MAX_QUEUE_DEPTH as u64;
        
        // If buffer is full, remove oldest element from sum
        if next_head == tail {
            let old_val = unsafe { *self.data.get_unchecked(tail as usize) };
            let current_sum = self.sum.load(Ordering::Relaxed);
            self.sum.store(current_sum.saturating_sub(old_val as u64), Ordering::Relaxed);
            self.tail.store((tail + 1) % MAX_QUEUE_DEPTH as u64, Ordering::Relaxed);
        }
        
        // Add new element
        unsafe {
            *self.data.get_unchecked_mut(head as usize) = volume;
        }
        
        // Update running sum
        let current_sum = self.sum.load(Ordering::Relaxed);
        self.sum.store(current_sum.saturating_add(volume as u64), Ordering::Relaxed);
        
        // Update head
        self.head.store(next_head, Ordering::Release);
    }
    
    #[inline(always)]
    fn total_flow(&self) -> i64 {
        self.sum.load(Ordering::Acquire) as i64
    }
    
    #[inline(always)]
    fn clear(&self) {
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
    }
}

/// Queue position state - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QueuePositionState {
    pub initial_position: i64,      // Our initial position in queue (shares)
    pub current_position: i64,      // Estimated current position
    pub queue_size_ahead: i64,      // Total size ahead of us
    pub order_id_hash: u64,         // Hash of our order ID for tracking
    pub timestamp_cycles: u64,      // rdtsc timestamp
    pub is_active: bool,            // Is this order still active
    pub was_filled: bool,           // Was the order filled
    _padding: [u8; 35],             // Pad to 64 bytes
}

impl Default for QueuePositionState {
    fn default() -> Self {
        Self {
            initial_position: 0,
            current_position: 0,
            queue_size_ahead: 0,
            order_id_hash: 0,
            timestamp_cycles: 0,
            is_active: false,
            was_filled: false,
            _padding: [0; 35],
        }
    }
}

/// Lock-free queue position estimator
#[repr(C)]
pub struct QueuePositionEstimator {
    state: QueuePositionState,
    trade_flow: TradeFlowBuffer,
    initial_queue_size: AtomicU64,
    depletion_rate: AtomicU64,      // Fixed-point: shares per microsecond
    estimated_fill_time: AtomicU64, // In cycles
    toxicity_flag: AtomicBool,      // Set if adverse selection detected
    _padding: [u8; 44],             // Pad to cache line
}

impl QueuePositionEstimator {
    /// Create a new queue position estimator
    pub const fn new() -> Self {
        Self {
            state: QueuePositionState {
                initial_position: 0,
                current_position: 0,
                queue_size_ahead: 0,
                order_id_hash: 0,
                timestamp_cycles: 0,
                is_active: false,
                was_filled: false,
                _padding: [0; 35],
            },
            trade_flow: TradeFlowBuffer::new(),
            initial_queue_size: AtomicU64::new(0),
            depletion_rate: AtomicU64::new(0),
            estimated_fill_time: AtomicU64::new(0),
            toxicity_flag: AtomicBool::new(false),
            _padding: [0; 44],
        }
    }
    
    /// Initialize queue position when order is placed
    #[inline(always)]
    pub fn initialize(&self, order_id: u64, initial_pos: i64, queue_size: i64) {
        let cycles = unsafe { x86_64::_rdtsc() };
        
        self.initial_queue_size.store(queue_size as u64, Ordering::Release);
        self.state.order_id_hash = order_id.wrapping_mul(0x517cc1b727220a95);
        self.state.initial_position = initial_pos;
        self.state.current_position = initial_pos;
        self.state.queue_size_ahead = queue_size;
        self.state.timestamp_cycles = cycles;
        self.state.is_active = true;
        self.state.was_filled = false;
        
        self.trade_flow.clear();
        self.toxicity_flag.store(false, Ordering::Release);
    }
    
    /// Update queue position based on trade flow ahead
    #[inline(always)]
    pub fn update_trade_flow(&self, trade_volume: i64, is_ahead: bool) {
        if !self.state.is_active {
            return;
        }
        
        // Branchless: only count trades ahead of us
        let mask = -(is_ahead as i64);
        let effective_volume = trade_volume & mask;
        
        self.trade_flow.push(effective_volume);
        
        // Update position
        let flow_total = self.trade_flow.total_flow();
        let new_position = self.state.initial_position.saturating_sub(flow_total);
        
        // Branchless comparison and update
        let is_filled = (new_position <= 0) as i64;
        self.state.current_position = new_position.max(0);
        self.state.was_filled |= is_filled != 0;
        self.state.is_active &= is_filled == 0;
        
        // Update depletion rate
        if flow_total > 0 {
            let elapsed = unsafe { x86_64::_rdtsc().wrapping_sub(self.state.timestamp_cycles) };
            let rate = ((flow_total as u128 * 1_000_000) / elapsed.max(1) as u128) as u64;
            self.depletion_rate.store(rate, Ordering::Relaxed);
            
            // Estimate fill time
            if self.state.current_position > 0 && rate > 0 {
                let remaining = self.state.current_position as u128;
                let est_time = (remaining * 1_000_000) / rate as u128;
                self.estimated_fill_time.store(est_time as u64, Ordering::Relaxed);
            }
        }
    }
    
    /// Get estimated probability of fill within N microseconds
    #[inline(always)]
    pub fn fill_probability(&self, time_horizon_us: u64) -> i64 {
        if !self.state.is_active || self.state.was_filled {
            return if self.state.was_filled { FIXED_SCALE } else { 0 };
        }
        
        let rate = self.depletion_rate.load(Ordering::Acquire);
        if rate == 0 {
            return 0;
        }
        
        // Expected depletion in time horizon
        let expected_depletion = ((rate as u128 * time_horizon_us as u128) / 1_000_000) as i64;
        let remaining = self.state.current_position;
        
        // Probability = min(1.0, expected/remaining) in fixed point
        if expected_depletion >= remaining {
            FIXED_SCALE
        } else {
            ((expected_depletion as i128 * FIXED_SCALE as i128) / remaining.max(1) as i128) as i64
        }
    }
    
    /// Check if we should cancel due to toxicity (adverse selection)
    #[inline(always)]
    pub fn should_cancel_toxic(&self, threshold: i64) -> bool {
        self.toxicity_flag.load(Ordering::Acquire) || 
        self.fill_probability(1000) < threshold // Low fill prob in 1ms
    }
    
    /// Mark order as toxic (adverse selection detected)
    #[inline(always)]
    pub fn mark_toxic(&self) {
        self.toxicity_flag.store(true, Ordering::Release);
    }
    
    /// Get current position
    #[inline(always)]
    pub fn current_position(&self) -> i64 {
        self.state.current_position
    }
    
    /// Get estimated fill time in microseconds
    #[inline(always)]
    pub fn estimated_fill_time_us(&self) -> u64 {
        self.estimated_fill_time.load(Ordering::Acquire)
    }
    
    /// Cancel the order
    #[inline(always)]
    pub fn cancel(&self) {
        self.state.is_active = false;
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<QueuePositionState>() == 64);
        assert!(core::mem::size_of::<TradeFlowBuffer>() % 64 == 0);
        assert!(core::mem::size_of::<QueuePositionEstimator>() % 64 == 0);
    }
    
    #[test]
    fn test_queue_position_basic() {
        let estimator = QueuePositionEstimator::new();
        estimator.initialize(12345, 1000, 5000);
        
        assert_eq!(estimator.current_position(), 1000);
        
        // Simulate 500 shares traded ahead
        estimator.update_trade_flow(500, true);
        assert_eq!(estimator.current_position(), 500);
        
        // Simulate another 600 shares traded ahead (should fill)
        estimator.update_trade_flow(600, true);
        assert!(estimator.current_position() == 0);
    }
    
    #[test]
    fn test_fill_probability() {
        let estimator = QueuePositionEstimator::new();
        estimator.initialize(12345, 1000, 5000);
        
        // No flow yet, probability should be 0
        assert_eq!(estimator.fill_probability(1000), 0);
        
        // Add some flow
        for _ in 0..10 {
            estimator.update_trade_flow(100, true);
        }
        
        // Should have some probability now
        let prob = estimator.fill_probability(1000);
        assert!(prob > 0);
    }
}
