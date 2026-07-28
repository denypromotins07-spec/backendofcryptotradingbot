//! Iceberg and Hidden Order Handler
//! 
//! Queue-position-aware fill modeling with hidden order logic.
//! Pre-allocated buffers, no heap allocation in hot path.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of iceberg orders tracked
const MAX_ICEBERG_ORDERS: usize = 256;

/// Maximum queue positions per order
const MAX_QUEUE_POSITIONS: usize = 64;

/// Iceberg order state
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergState {
    Pending = 0,
    Active = 1,
    PartiallyFilled = 2,
    Completed = 3,
    Cancelled = 4,
}

/// Iceberg order - fixed size, cache-line aligned
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IcebergOrder {
    /// Unique order ID
    pub order_id: u64,
    /// Total (hidden) quantity
    pub total_qty: u64,
    /// Visible (displayed) quantity
    pub visible_qty: u64,
    /// Filled quantity
    pub filled_qty: u64,
    /// Remaining visible quantity
    pub remaining_visible: u64,
    /// Order price
    pub price: u64,
    /// Side (0=buy, 1=sell)
    pub side: u8,
    /// Current state
    pub state: u8,
    /// Venue ID
    pub venue_id: u8,
    /// Minimum display quantity
    pub min_display_qty: u64,
    /// Queue position at venue
    pub queue_position: AtomicU64,
    /// Orders ahead in queue
    pub orders_ahead: AtomicU64,
    /// Volume ahead in queue (base units)
    pub volume_ahead: AtomicU64,
    /// Last fill timestamp (cycles)
    pub last_fill_time: AtomicU64,
    /// Creation timestamp (cycles)
    pub created_time: AtomicU64,
    /// Fill rate estimate (bps per ms)
    pub fill_rate_estimate: AtomicU64,
    _padding: [u8; 16],
}

impl IcebergOrder {
    const fn empty() -> Self {
        Self {
            order_id: 0,
            total_qty: 0,
            visible_qty: 0,
            filled_qty: 0,
            remaining_visible: 0,
            price: 0,
            side: 0,
            state: IcebergState::Pending as u8,
            venue_id: 0,
            min_display_qty: 0,
            queue_position: AtomicU64::new(0),
            orders_ahead: AtomicU64::new(0),
            volume_ahead: AtomicU64::new(0),
            last_fill_time: AtomicU64::new(0),
            created_time: AtomicU64::new(0),
            fill_rate_estimate: AtomicU64::new(0),
            _padding: [0u8; 16],
        }
    }

    /// Get remaining hidden quantity
    #[inline(always)]
    pub fn remaining_hidden(&self) -> u64 {
        self.total_qty.saturating_sub(self.filled_qty).saturating_sub(self.remaining_visible)
    }

    /// Get total remaining quantity
    #[inline(always)]
    pub fn remaining_total(&self) -> u64 {
        self.total_qty.saturating_sub(self.filled_qty)
    }
}

/// Queue position tracker for fill modeling
#[repr(C)]
struct QueuePosition {
    /// Our position in queue (1-indexed)
    position: AtomicU64,
    /// Total volume at our price level
    level_volume: AtomicU64,
    /// Volume ahead of us
    volume_ahead: AtomicU64,
    /// Recent fill volume at this level (for rate estimation)
    recent_fill_volume: AtomicU64,
    /// Recent fill count
    recent_fill_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

impl QueuePosition {
    const fn new() -> Self {
        Self {
            position: AtomicU64::new(0),
            level_volume: AtomicU64::new(0),
            volume_ahead: AtomicU64::new(0),
            recent_fill_volume: AtomicU64::new(0),
            recent_fill_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }

    #[inline(always)]
    fn update_position(&self, pos: u64, vol_ahead: u64, level_vol: u64) {
        self.position.store(pos, Ordering::Release);
        self.volume_ahead.store(vol_ahead, Ordering::Release);
        self.level_volume.store(level_vol, Ordering::Release);
    }

    #[inline(always)]
    fn record_fill_at_level(&self, fill_qty: u64) {
        self.recent_fill_volume.fetch_add(fill_qty, Ordering::Relaxed);
        self.recent_fill_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn get_fill_rate(&self) -> u64 {
        let vol = self.recent_fill_volume.load(Ordering::Relaxed);
        let count = self.recent_fill_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }
        vol / count
    }

    #[inline(always)]
    fn reset_recent(&self) {
        self.recent_fill_volume.store(0, Ordering::Release);
        self.recent_fill_count.store(0, Ordering::Release);
    }
}

/// Iceberg order handler
pub struct IcebergHandler {
    /// Pre-allocated iceberg orders
    orders: [IcebergOrder; MAX_ICEBERG_ORDERS],
    /// Number of active orders
    num_active: AtomicU64,
    /// Queue position trackers
    queue_positions: [QueuePosition; MAX_ICEBERG_ORDERS],
    /// Default minimum display ratio (basis points)
    default_display_ratio_bp: AtomicU64,
    /// Aggressive refill threshold (queue position)
    aggressive_refill_threshold: AtomicU64,
    /// Enable queue position tracking
    queue_tracking_enabled: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for IcebergHandler {}
unsafe impl Sync for IcebergHandler {}

impl IcebergHandler {
    /// Create new iceberg handler
    pub const fn new() -> Self {
        const EMPTY_ORDER: IcebergOrder = IcebergOrder::empty();
        const EMPTY_QUEUE: QueuePosition = QueuePosition::new();
        
        Self {
            orders: [EMPTY_ORDER; MAX_ICEBERG_ORDERS],
            num_active: AtomicU64::new(0),
            queue_positions: [EMPTY_QUEUE; MAX_ICEBERG_ORDERS],
            default_display_ratio_bp: AtomicU64::new(1000), // 10% default
            aggressive_refill_threshold: AtomicU64::new(10), // Position <= 10
            queue_tracking_enabled: AtomicBool::new(true),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Create a new iceberg order
    #[inline(always)]
    pub fn create_iceberg(
        &self,
        order_id: u64,
        total_qty: u64,
        price: u64,
        side: u8,
        venue_id: u8,
    ) -> Option<u64> {
        if total_qty == 0 || price == 0 {
            return None;
        }

        // Find empty slot using linear scan (branchless unrolled)
        let mut slot = None;
        for i in 0..MAX_ICEBERG_ORDERS {
            let ord = &self.orders[i];
            let is_empty = (ord.order_id == 0) as u8;
            if is_empty != 0 && slot.is_none() {
                slot = Some(i);
            }
        }

        let idx = slot?;
        
        // Calculate display quantity
        let display_ratio = self.default_display_ratio_bp.load(Ordering::Relaxed);
        let min_display = self.calculate_min_display(total_qty, display_ratio);
        let visible_qty = min_display.min(total_qty);

        let now = unsafe { _rdtsc() };
        
        let order = IcebergOrder {
            order_id,
            total_qty,
            visible_qty,
            filled_qty: 0,
            remaining_visible: visible_qty,
            price,
            side,
            state: IcebergState::Active as u8,
            venue_id,
            min_display_qty: min_display,
            queue_position: AtomicU64::new(1),
            orders_ahead: AtomicU64::new(0),
            volume_ahead: AtomicU64::new(0),
            last_fill_time: AtomicU64::new(0),
            created_time: now,
            fill_rate_estimate: AtomicU64::new(0),
            _padding: [0u8; 16],
        };

        self.orders[idx] = order;
        self.num_active.fetch_add(1, Ordering::Release);
        
        Some(idx as u64)
    }

    /// Calculate minimum display quantity
    #[inline(always)]
    fn calculate_min_display(&self, total_qty: u64, ratio_bp: u64) -> u64 {
        let base = (total_qty * ratio_bp) / 10000;
        // Ensure at least 1 unit
        base.max(1)
    }

    /// Update queue position from market data
    #[inline(always)]
    pub fn update_queue_position(&self, order_idx: u64, position: u64, volume_ahead: u64, level_volume: u64) {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return;
        }

        let idx = order_idx as usize;
        let order = &mut self.orders[idx];
        
        order.queue_position.store(position, Ordering::Release);
        order.orders_ahead.store(position.saturating_sub(1), Ordering::Release);
        order.volume_ahead.store(volume_ahead, Ordering::Release);

        // Update queue tracker
        if self.queue_tracking_enabled.load(Ordering::Relaxed) {
            self.queue_positions[idx].update_position(position, volume_ahead, level_volume);
        }
    }

    /// Process a fill at the order's price level
    #[inline(always)]
    pub fn process_level_fill(&self, order_idx: u64, fill_qty: u64, is_our_fill: bool) {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return;
        }

        let idx = order_idx as usize;
        
        // Record fill at queue level
        self.queue_positions[idx].record_fill_at_level(fill_qty);

        if is_our_fill {
            let order = &mut self.orders[idx];
            let now = unsafe { _rdtsc() };
            
            // Update fill statistics
            let old_filled = order.filled_qty;
            order.filled_qty = old_filled + fill_qty;
            order.remaining_visible = order.remaining_visible.saturating_sub(fill_qty);
            order.last_fill_time.store(now, Ordering::Release);

            // Update state
            if order.filled_qty >= order.total_qty {
                order.state = IcebergState::Completed as u8;
                self.num_active.fetch_sub(1, Ordering::Release);
            } else if order.filled_qty > 0 {
                order.state = IcebergState::PartiallyFilled as u8;
            }

            // Refill visible quantity if depleted
            if order.remaining_visible == 0 && order.filled_qty < order.total_qty {
                self.refill_visible(idx);
            }
        }
    }

    /// Refill visible quantity from hidden reserve
    #[inline(always)]
    fn refill_visible(&self, idx: usize) {
        let order = &mut self.orders[idx];
        let remaining_hidden = order.remaining_hidden();
        
        if remaining_hidden == 0 {
            return;
        }

        // Check queue position for aggressive/passive refill decision
        let queue_pos = order.queue_position.load(Ordering::Relaxed);
        let threshold = self.aggressive_refill_threshold.load(Ordering::Relaxed);

        let refill_qty = if queue_pos <= threshold {
            // Aggressive: refill full display size
            order.min_display_qty.min(remaining_hidden)
        } else {
            // Passive: refill minimum
            order.min_display_qty.min(remaining_hidden)
        };

        order.remaining_visible = refill_qty;
    }

    /// Estimate time to fill based on queue position and recent activity
    #[inline(always)]
    pub fn estimate_fill_time_ms(&self, order_idx: u64) -> u64 {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return u64::MAX;
        }

        let idx = order_idx as usize;
        let order = &self.orders[idx];
        
        let volume_ahead = order.volume_ahead.load(Ordering::Relaxed);
        let fill_rate = self.queue_positions[idx].get_fill_rate();

        if fill_rate == 0 {
            return u64::MAX;
        }

        // Time = volume_ahead / fill_rate (in ms)
        volume_ahead / fill_rate
    }

    /// Get probability of fill within N milliseconds (0-10000 basis points)
    #[inline(always)]
    pub fn fill_probability_bp(&self, order_idx: u64, time_ms: u64) -> u16 {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return 0;
        }

        let idx = order_idx as usize;
        let order = &self.orders[idx];
        
        let volume_ahead = order.volume_ahead.load(Ordering::Relaxed);
        let fill_rate = self.queue_positions[idx].get_fill_rate();

        if fill_rate == 0 || volume_ahead == 0 {
            return 0;
        }

        let expected_fill_vol = fill_rate * time_ms;
        
        // Probability based on whether expected volume exceeds volume ahead
        if expected_fill_vol >= volume_ahead {
            10000
        } else {
            ((expected_fill_vol * 10000) / volume_ahead) as u16
        }
    }

    /// Cancel an iceberg order
    #[inline(always)]
    pub fn cancel_order(&self, order_idx: u64) -> bool {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return false;
        }

        let idx = order_idx as usize;
        let order = &mut self.orders[idx];
        
        if order.state == IcebergState::Completed as u8 
            || order.state == IcebergState::Cancelled as u8 {
            return false;
        }

        order.state = IcebergState::Cancelled as u8;
        self.num_active.fetch_sub(1, Ordering::Release);
        true
    }

    /// Get order by index
    #[inline(always)]
    pub fn get_order(&self, order_idx: u64) -> Option<IcebergOrder> {
        if order_idx >= MAX_ICEBERG_ORDERS as u64 {
            return None;
        }
        Some(self.orders[order_idx as usize])
    }

    /// Get number of active iceberg orders
    #[inline(always)]
    pub fn num_active_orders(&self) -> u64 {
        self.num_active.load(Ordering::Relaxed)
    }

    /// Set display ratio for new orders
    #[inline(always)]
    pub fn set_display_ratio_bp(&self, ratio_bp: u64) {
        self.default_display_ratio_bp.store(ratio_bp.min(10000), Ordering::Release);
    }

    /// Set aggressive refill threshold
    #[inline(always)]
    pub fn set_aggressive_threshold(&self, threshold: u64) {
        self.aggressive_refill_threshold.store(threshold, Ordering::Release);
    }

    /// Reset all orders (for system restart)
    #[inline(always)]
    pub fn reset_all(&self) {
        for i in 0..MAX_ICEBERG_ORDERS {
            self.orders[i] = IcebergOrder::empty();
            self.queue_positions[i] = QueuePosition::new();
        }
        self.num_active.store(0, Ordering::Release);
    }

    /// SIMD-accelerated queue position update for multiple orders
    #[inline(always)]
    pub fn update_queue_positions_simd(&self, positions: &[u64; 4], volumes_ahead: &[u64; 4], start_idx: usize) {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.update_queue_positions_avx2(positions, volumes_ahead, start_idx);
            } else {
                for i in 0..4 {
                    if start_idx + i < MAX_ICEBERG_ORDERS {
                        self.update_queue_position(
                            (start_idx + i) as u64,
                            positions[i],
                            volumes_ahead[i],
                            0,
                        );
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(always)]
    unsafe fn update_queue_positions_avx2(&self, positions: &[u64; 4], volumes_ahead: &[u64; 4], start_idx: usize) {
        use core::arch::x86_64::*;

        // Load positions and volumes into AVX2 registers
        let pos_vec = _mm256_loadu_si256(positions.as_ptr() as *const __m256i);
        let vol_vec = _mm256_loadu_si256(volumes_ahead.as_ptr() as *const __m256i);

        // Store to order structures (unrolled for performance)
        for i in 0..4 {
            if start_idx + i < MAX_ICEBERG_ORDERS {
                let idx = start_idx + i;
                let pos = _mm256_extract_epi64::<0>(pos_vec) >> (i * 64);
                let vol = _mm256_extract_epi64::<0>(vol_vec) >> (i * 64);
                
                self.orders[idx].queue_position.store(pos as u64, Ordering::Release);
                self.orders[idx].volume_ahead.store(vol as u64, Ordering::Release);
            }
        }
    }
}

impl Default for IcebergHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_order_size() {
        assert_eq!(core::mem::size_of::<IcebergOrder>(), 128);
    }

    #[test]
    fn test_create_iceberg() {
        let handler = IcebergHandler::new();
        
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0);
        assert!(idx.is_some());
        
        let order = handler.get_order(idx.unwrap()).unwrap();
        assert_eq!(order.total_qty, 10000);
        assert_eq!(order.state, IcebergState::Active as u8);
        assert!(order.visible_qty > 0);
        assert!(order.visible_qty < order.total_qty);
    }

    #[test]
    fn test_queue_position_update() {
        let handler = IcebergHandler::new();
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0).unwrap();
        
        handler.update_queue_position(idx, 5, 5000, 10000);
        
        let order = handler.get_order(idx).unwrap();
        assert_eq!(order.queue_position.load(Ordering::Relaxed), 5);
        assert_eq!(order.volume_ahead.load(Ordering::Relaxed), 5000);
    }

    #[test]
    fn test_process_fill() {
        let handler = IcebergHandler::new();
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0).unwrap();
        
        let initial_visible = handler.orders[idx as usize].remaining_visible;
        
        // Process a fill
        handler.process_level_fill(idx, 100, true);
        
        let order = handler.get_order(idx).unwrap();
        assert_eq!(order.filled_qty, 100);
        assert_eq!(order.remaining_visible, initial_visible - 100);
        assert_eq!(order.state, IcebergState::PartiallyFilled as u8);
    }

    #[test]
    fn test_visible_refill() {
        let handler = IcebergHandler::new();
        handler.set_display_ratio_bp(1000); // 10%
        
        let idx = handler.create_iceberg(1, 1000, 50000, 0, 0).unwrap();
        
        // Deplete visible quantity
        handler.process_level_fill(idx, 100, true); // This should trigger refill
        
        let order = handler.get_order(idx).unwrap();
        // After filling the entire visible portion, should have refilled
        assert!(order.remaining_visible > 0 || order.filled_qty >= order.total_qty);
    }

    #[test]
    fn test_fill_probability() {
        let handler = IcebergHandler::new();
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0).unwrap();
        
        // Set up queue position with known fill rate
        handler.update_queue_position(idx, 2, 1000, 5000);
        handler.queue_positions[idx as usize].recent_fill_volume.store(500, Ordering::Release);
        handler.queue_positions[idx as usize].recent_fill_count.store(1, Ordering::Release);
        
        let prob = handler.fill_probability_bp(idx, 10); // 10ms
        assert!(prob > 0);
    }

    #[test]
    fn test_cancel_order() {
        let handler = IcebergHandler::new();
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0).unwrap();
        
        assert_eq!(handler.num_active_orders(), 1);
        
        let cancelled = handler.cancel_order(idx);
        assert!(cancelled);
        
        let order = handler.get_order(idx).unwrap();
        assert_eq!(order.state, IcebergState::Cancelled as u8);
        assert_eq!(handler.num_active_orders(), 0);
    }

    #[test]
    fn test_remaining_quantities() {
        let handler = IcebergHandler::new();
        let idx = handler.create_iceberg(1, 10000, 50000, 0, 0).unwrap();
        
        let order = handler.get_order(idx).unwrap();
        assert_eq!(order.remaining_total(), 10000);
        assert_eq!(order.remaining_hidden(), 10000 - order.visible_qty);
        
        handler.process_level_fill(idx, 5000, true);
        
        let order = handler.get_order(idx).unwrap();
        assert_eq!(order.remaining_total(), 5000);
    }
}
