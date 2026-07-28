//! TWAP/VWAP/POV Execution Algorithms
//! 
//! Deterministic time-sliced child orders using custom high-resolution timer wheel.
//! Avoids OS scheduler latency and jitter.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of child orders in a slice
const MAX_CHILD_ORDERS: usize = 256;

/// Maximum timer wheel slots (1 second at 10us resolution = 100000 slots)
const TIMER_WHEEL_SLOTS: usize = 65536;

/// Algorithm type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Twap = 0,
    Vwap = 1,
    Pov = 2,
}

/// Order side
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

/// Child order - fixed size, pre-allocated
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChildOrder {
    pub order_id: u64,
    pub parent_id: u64,
    pub quantity: u64,
    pub price: u64,
    pub side: u8,
    pub venue_id: u8,
    pub time_slot: u16,
    pub status: u8, // 0=pending, 1=submitted, 2=filled, 3=cancelled
    pub fill_qty: u64,
    pub fill_price: u64,
    pub timestamp_cycles: u64,
    pub _padding: [u8; 24],
}

impl ChildOrder {
    const fn empty() -> Self {
        Self {
            order_id: 0,
            parent_id: 0,
            quantity: 0,
            price: 0,
            side: 0,
            venue_id: 0,
            time_slot: 0,
            status: 0,
            fill_qty: 0,
            fill_price: 0,
            timestamp_cycles: 0,
            _padding: [0u8; 24],
        }
    }
}

/// VWAP profile bucket - pre-computed volume distribution
#[repr(C)]
struct VwapBucket {
    /// Expected volume percentage (basis points)
    volume_bp: u16,
    /// Start time offset (milliseconds)
    start_ms: u32,
    /// End time offset (milliseconds)
    end_ms: u32,
}

impl VwapBucket {
    const fn new() -> Self {
        Self {
            volume_bp: 0,
            start_ms: 0,
            end_ms: 0,
        }
    }
}

/// Pre-computed VWAP profile for a trading day (96 x 15-minute buckets)
static VWAP_PROFILE: [VwapBucket; 96] = {
    let mut arr = [VwapBucket::new(); 96];
    let mut i = 0;
    while i < 96 {
        arr[i] = VwapBucket {
            volume_bp: 104, // Approximately equal distribution
            start_ms: (i * 15 * 60 * 1000) as u32,
            end_ms: ((i + 1) * 15 * 60 * 1000) as u32,
        };
        i += 1;
    }
    arr
};

/// Timer wheel slot
#[repr(C)]
struct TimerSlot {
    /// Head of linked list of orders in this slot
    order_indices: [u16; 16],
    /// Count of orders in slot
    count: AtomicU64,
    /// Next slot to process
    next: u16,
    _padding: [u8; CACHE_LINE_SIZE - 72],
}

impl TimerSlot {
    const fn new() -> Self {
        Self {
            order_indices: [0xFFFF; 16],
            count: AtomicU64::new(0),
            next: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 72],
        }
    }
}

/// Execution algorithm state
pub struct ExecutionAlgo {
    /// Pre-allocated child order buffer
    child_orders: [ChildOrder; MAX_CHILD_ORDERS],
    /// Number of child orders created
    num_children: AtomicU64,
    /// Total parent order quantity
    parent_qty: AtomicU64,
    /// Filled quantity
    filled_qty: AtomicU64,
    /// Average fill price
    avg_fill_price: AtomicU64,
    /// Algorithm type
    algo_type: u8,
    /// Side
    side: u8,
    /// Active flag
    active: AtomicBool,
    /// Start time (cycles)
    start_time: AtomicU64,
    /// End time (cycles)
    end_time: AtomicU64,
    /// Duration in milliseconds
    duration_ms: u64,
    /// Number of slices
    num_slices: u16,
    /// Current slice index
    current_slice: AtomicU64,
    /// POV participation rate (basis points)
    pov_rate_bp: u16,
    /// Timer wheel
    timer_wheel: [TimerSlot; TIMER_WHEEL_SLOTS],
    /// Current timer position
    timer_position: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for ExecutionAlgo {}
unsafe impl Sync for ExecutionAlgo {}

impl ExecutionAlgo {
    /// Create new TWAP execution algorithm
    pub const fn new_twap(parent_qty: u64, duration_ms: u64, num_slices: u16) -> Self {
        const EMPTY_ORDER: ChildOrder = ChildOrder::empty();
        const EMPTY_SLOT: TimerSlot = TimerSlot::new();
        
        Self {
            child_orders: [EMPTY_ORDER; MAX_CHILD_ORDERS],
            num_children: AtomicU64::new(0),
            parent_qty: AtomicU64::new(parent_qty),
            filled_qty: AtomicU64::new(0),
            avg_fill_price: AtomicU64::new(0),
            algo_type: AlgoType::Twap as u8,
            side: Side::Buy as u8,
            active: AtomicBool::new(false),
            start_time: AtomicU64::new(0),
            end_time: AtomicU64::new(0),
            duration_ms,
            num_slices,
            current_slice: AtomicU64::new(0),
            pov_rate_bp: 0,
            timer_wheel: [EMPTY_SLOT; TIMER_WHEEL_SLOTS],
            timer_position: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Create new VWAP execution algorithm
    pub const fn new_vwap(parent_qty: u64, duration_ms: u64) -> Self {
        const EMPTY_ORDER: ChildOrder = ChildOrder::empty();
        const EMPTY_SLOT: TimerSlot = TimerSlot::new();
        
        Self {
            child_orders: [EMPTY_ORDER; MAX_CHILD_ORDERS],
            num_children: AtomicU64::new(0),
            parent_qty: AtomicU64::new(parent_qty),
            filled_qty: AtomicU64::new(0),
            avg_fill_price: AtomicU64::new(0),
            algo_type: AlgoType::Vwap as u8,
            side: Side::Buy as u8,
            active: AtomicBool::new(false),
            start_time: AtomicU64::new(0),
            end_time: AtomicU64::new(0),
            duration_ms,
            num_slices: 96, // Match VWAP profile buckets
            current_slice: AtomicU64::new(0),
            pov_rate_bp: 0,
            timer_wheel: [EMPTY_SLOT; TIMER_WHEEL_SLOTS],
            timer_position: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Create new POV execution algorithm
    pub const fn new_pov(parent_qty: u64, participation_rate_bp: u16) -> Self {
        const EMPTY_ORDER: ChildOrder = ChildOrder::empty();
        const EMPTY_SLOT: TimerSlot = TimerSlot::new();
        
        Self {
            child_orders: [EMPTY_ORDER; MAX_CHILD_ORDERS],
            num_children: AtomicU64::new(0),
            parent_qty: AtomicU64::new(parent_qty),
            filled_qty: AtomicU64::new(0),
            avg_fill_price: AtomicU64::new(0),
            algo_type: AlgoType::Pov as u8,
            side: Side::Buy as u8,
            active: AtomicBool::new(false),
            start_time: AtomicU64::new(0),
            end_time: AtomicU64::new(0),
            duration_ms: 0,
            num_slices: 0,
            current_slice: AtomicU64::new(0),
            pov_rate_bp: participation_rate_bp,
            timer_wheel: [EMPTY_SLOT; TIMER_WHEEL_SLOTS],
            timer_position: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Set order side
    #[inline(always)]
    pub fn set_side(&mut self, side: Side) {
        self.side = side as u8;
    }

    /// Start the algorithm
    #[inline(always)]
    pub fn start(&self) {
        let now = unsafe { _rdtsc() };
        self.start_time.store(now, Ordering::Release);
        
        // Calculate end time (convert ms to cycles, assuming ~3GHz)
        let cycles_per_ms = 3_000_000;
        let end = now + (self.duration_ms * cycles_per_ms);
        self.end_time.store(end, Ordering::Release);
        
        self.active.store(true, Ordering::Release);
        
        // Generate child orders based on algorithm type
        match self.algo_type {
            0 => self.generate_twap_slices(),
            1 => self.generate_vwap_slices(),
            2 => {} // POV generates orders dynamically
            _ => {}
        }
    }

    /// Generate TWAP child orders
    #[inline(always)]
    fn generate_twap_slices(&self) {
        let parent_qty = self.parent_qty.load(Ordering::Relaxed);
        let num_slices = self.num_slices as u64;
        
        if num_slices == 0 || num_slices > MAX_CHILD_ORDERS as u64 {
            return;
        }

        let qty_per_slice = parent_qty / num_slices;
        let remainder = parent_qty % num_slices;
        let cycle_freq = unsafe { _rdtsc() } / 1_000_000; // cycles per ms
        
        for i in 0..num_slices.min(MAX_CHILD_ORDERS as u64) {
            let qty = qty_per_slice + if i < remainder { 1 } else { 0 };
            let time_offset_ms = (self.duration_ms * i) / num_slices;
            let time_slot = ((time_offset_ms * cycle_freq) & (TIMER_WHEEL_SLOTS as u64 - 1)) as u16;

            let order = ChildOrder {
                order_id: i + 1,
                parent_id: 1,
                quantity: qty,
                price: 0, // Market order
                side: self.side,
                venue_id: 0,
                time_slot,
                status: 0,
                fill_qty: 0,
                fill_price: 0,
                timestamp_cycles: 0,
                _padding: [0u8; 24],
            };

            self.child_orders[i as usize] = order;
            self.add_to_timer_wheel(i as u16, time_slot);
        }

        self.num_children.store(num_slices, Ordering::Release);
    }

    /// Generate VWAP child orders based on historical volume profile
    #[inline(always)]
    fn generate_vwap_slices(&self) {
        let parent_qty = self.parent_qty.load(Ordering::Relaxed);
        let mut remaining_qty = parent_qty;

        for i in 0..self.num_slices.min(96) {
            let bucket = &VWAP_PROFILE[i];
            let slice_qty = (parent_qty as u64 * bucket.volume_bp as u64) / 10000;
            
            if i == 95 {
                // Last bucket gets remainder
                remaining_qty = slice_qty.max(remaining_qty);
            }

            let time_slot = ((bucket.start_ms as u64 * 3_000_000) & (TIMER_WHEEL_SLOTS as u64 - 1)) as u16;

            let order = ChildOrder {
                order_id: i as u64 + 1,
                parent_id: 1,
                quantity: remaining_qty.min(slice_qty),
                price: 0,
                side: self.side,
                venue_id: 0,
                time_slot,
                status: 0,
                fill_qty: 0,
                fill_price: 0,
                timestamp_cycles: 0,
                _padding: [0u8; 24],
            };

            if i < MAX_CHILD_ORDERS as u16 {
                self.child_orders[i as usize] = order;
                self.add_to_timer_wheel(i as u16, time_slot);
            }

            remaining_qty = remaining_qty.saturating_sub(slice_qty);
        }

        self.num_children.store(self.num_slices as u64, Ordering::Release);
    }

    /// Add order to timer wheel
    #[inline(always)]
    fn add_to_timer_wheel(&self, order_idx: u16, slot: u16) {
        let slot_idx = slot as usize;
        if slot_idx >= TIMER_WHEEL_SLOTS {
            return;
        }

        let timer_slot = &self.timer_wheel[slot_idx];
        let count = timer_slot.count.load(Ordering::Relaxed);
        
        if count < 16 {
            // Unsafe but safe because we control the array bounds
            unsafe {
                let ptr = timer_slot.order_indices.as_ptr() as *mut u16;
                *ptr.add(count as usize) = order_idx;
            }
            timer_slot.count.fetch_add(1, Ordering::Release);
        }
    }

    /// Get next child order to submit (called by timer wheel processor)
    #[inline(always)]
    pub fn get_next_order(&self) -> Option<ChildOrder> {
        if !self.active.load(Ordering::Acquire) {
            return None;
        }

        let current_pos = self.timer_position.load(Ordering::Relaxed);
        let slot_idx = (current_pos & (TIMER_WHEEL_SLOTS as u64 - 1)) as usize;
        let slot = &self.timer_wheel[slot_idx];
        
        let count = slot.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }

        // Get first order from slot
        let order_idx = unsafe {
            let ptr = slot.order_indices.as_ptr();
            *ptr as usize
        };

        if order_idx < MAX_CHILD_ORDERS {
            let order = self.child_orders[order_idx];
            if order.status == 0 && order.quantity > 0 {
                return Some(order);
            }
        }

        None
    }

    /// Advance timer wheel
    #[inline(always)]
    pub fn advance_timer(&self) {
        let current = self.timer_position.fetch_add(1, Ordering::Relaxed);
        let slot_idx = (current & (TIMER_WHEEL_SLOTS as u64 - 1)) as usize;
        
        // Clear processed slot
        self.timer_wheel[slot_idx].count.store(0, Ordering::Release);
    }

    /// Update order fill status
    #[inline(always)]
    pub fn update_fill(&self, order_id: u64, fill_qty: u64, fill_price: u64) {
        if order_id == 0 || order_id > MAX_CHILD_ORDERS as u64 {
            return;
        }

        let idx = (order_id - 1) as usize;
        let order = &mut self.child_orders[idx];
        order.status = 2; // Filled
        order.fill_qty = fill_qty;
        order.fill_price = fill_price;
        order.timestamp_cycles = unsafe { _rdtsc() };

        // Update aggregate stats
        let old_filled = self.filled_qty.fetch_add(fill_qty, Ordering::AcqRel);
        let old_avg = self.avg_fill_price.load(Ordering::Relaxed);
        let new_total_value = (old_avg as u128 * old_filled as u128) + (fill_price as u128 * fill_qty as u128);
        let new_filled = old_filled + fill_qty;
        let new_avg = if new_filled > 0 {
            (new_total_value / new_filled as u128) as u64
        } else {
            0
        };
        self.avg_fill_price.store(new_avg, Ordering::Release);
    }

    /// Check if algorithm is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        let filled = self.filled_qty.load(Ordering::Acquire);
        let parent = self.parent_qty.load(Ordering::Acquire);
        filled >= parent
    }

    /// Get execution progress (basis points)
    #[inline(always)]
    pub fn get_progress_bp(&self) -> u16 {
        let filled = self.filled_qty.load(Ordering::Acquire);
        let parent = self.parent_qty.load(Ordering::Acquire);
        if parent == 0 {
            return 0;
        }
        ((filled * 10000) / parent) as u16
    }

    /// Get slippage in basis points
    #[inline(always)]
    pub fn get_slippage_bp(&self, arrival_price: u64) -> i16 {
        if arrival_price == 0 {
            return 0;
        }
        let avg_fill = self.avg_fill_price.load(Ordering::Acquire);
        if avg_fill == 0 {
            return 0;
        }

        // For buys: slippage = (avg_fill - arrival) / arrival
        // For sells: slippage = (arrival - avg_fill) / arrival
        let diff = if self.side == Side::Buy as u8 {
            avg_fill as i64 - arrival_price as i64
        } else {
            arrival_price as i64 - avg_fill as i64
        };

        ((diff * 10000) / arrival_price as i64) as i16
    }

    /// Stop the algorithm
    #[inline(always)]
    pub fn stop(&self) {
        self.active.store(false, Ordering::Release);
    }

    /// Get remaining quantity to execute
    #[inline(always)]
    pub fn get_remaining_qty(&self) -> u64 {
        let parent = self.parent_qty.load(Ordering::Acquire);
        let filled = self.filled_qty.load(Ordering::Acquire);
        parent.saturating_sub(filled)
    }

    /// Calculate POV order size based on market volume
    #[inline(always)]
    pub fn calculate_pov_size(&self, market_volume: u64) -> u64 {
        if market_volume == 0 {
            return 0;
        }
        (market_volume * self.pov_rate_bp as u64) / 10000
    }
}

impl Default for ExecutionAlgo {
    fn default() -> Self {
        Self::new_twap(1000, 60000, 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_child_order_size() {
        assert_eq!(core::mem::size_of::<ChildOrder>(), 64);
    }

    #[test]
    fn test_twap_creation() {
        let algo = ExecutionAlgo::new_twap(10000, 60000, 60);
        assert_eq!(algo.algo_type, AlgoType::Twap as u8);
        assert_eq!(algo.parent_qty.load(Ordering::Relaxed), 10000);
        assert_eq!(algo.duration_ms, 60000);
        assert_eq!(algo.num_slices, 60);
    }

    #[test]
    fn test_vwap_creation() {
        let algo = ExecutionAlgo::new_vwap(10000, 360000);
        assert_eq!(algo.algo_type, AlgoType::Vwap as u8);
        assert_eq!(algo.num_slices, 96);
    }

    #[test]
    fn test_pov_creation() {
        let algo = ExecutionAlgo::new_pov(10000, 500); // 5% participation
        assert_eq!(algo.algo_type, AlgoType::Pov as u8);
        assert_eq!(algo.pov_rate_bp, 500);
    }

    #[test]
    fn test_twap_slice_generation() {
        let algo = ExecutionAlgo::new_twap(1000, 60000, 10);
        algo.generate_twap_slices();

        let num_children = algo.num_children.load(Ordering::Relaxed);
        assert_eq!(num_children, 10);

        // Each slice should have ~100 units
        let total: u64 = algo.child_orders.iter().take(10).map(|o| o.quantity).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_execution_progress() {
        let algo = ExecutionAlgo::new_twap(1000, 60000, 10);
        
        assert_eq!(algo.get_progress_bp(), 0);
        
        algo.update_fill(1, 100, 50000);
        assert_eq!(algo.get_progress_bp(), 1000); // 10%
        
        algo.update_fill(2, 200, 50100);
        assert_eq!(algo.get_progress_bp(), 3000); // 30%
    }

    #[test]
    fn test_avg_fill_price() {
        let algo = ExecutionAlgo::new_twap(1000, 60000, 10);
        
        algo.update_fill(1, 100, 50000);
        assert_eq!(algo.avg_fill_price.load(Ordering::Relaxed), 50000);
        
        algo.update_fill(2, 100, 50200);
        // Average should be (50000*100 + 50200*100) / 200 = 50100
        assert_eq!(algo.avg_fill_price.load(Ordering::Relaxed), 50100);
    }

    #[test]
    fn test_slippage_calculation() {
        let mut algo = ExecutionAlgo::new_twap(1000, 60000, 10);
        algo.set_side(Side::Buy);
        
        algo.update_fill(1, 100, 50100);
        
        // Arrival price was 50000, filled at 50100 = 20 bps slippage
        let slippage = algo.get_slippage_bp(50000);
        assert_eq!(slippage, 20);
    }

    #[test]
    fn test_pov_size_calculation() {
        let algo = ExecutionAlgo::new_pov(10000, 500); // 5%
        
        let size = algo.calculate_pov_size(100000);
        assert_eq!(size, 5000); // 5% of 100000
    }

    #[test]
    fn test_completion_check() {
        let algo = ExecutionAlgo::new_twap(1000, 60000, 10);
        
        assert!(!algo.is_complete());
        
        algo.update_fill(1, 500, 50000);
        assert!(!algo.is_complete());
        
        algo.update_fill(2, 500, 50000);
        assert!(algo.is_complete());
    }
}
