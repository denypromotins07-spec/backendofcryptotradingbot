//! Advanced Iceberg Order Slicer
//! 
//! Implements randomized child order sizing to mask true intent,
//! with lock-free circular buffers for slice tracking.
//! Uses fixed-point arithmetic and zero-copy semantics throughout.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

/// Fixed-point quantity (scaled by 1e8)
pub type FixedQty = i64;
/// Timestamp from rdtsc
pub type TscTicks = u64;

/// Maximum number of slices tracked (power of 2 for efficient modulo)
const MAX_SLICES: usize = 64;
const SLICE_MASK: usize = MAX_SLICES - 1;

/// Cache-line aligned iceberg order state (64 bytes)
#[repr(C, align(64))]
pub struct IcebergOrder {
    pub order_id: AtomicU64,
    pub total_qty: AtomicI64,
    pub remaining_qty: AtomicI64,
    pub display_qty_target: AtomicI64,
    pub current_display_qty: AtomicI64,
    pub min_slice_pct: AtomicU64, // Basis points (100 = 1%)
    pub max_slice_pct: AtomicU64, // Basis points
    pub randomization_seed: AtomicU64,
    pub slice_count: AtomicU64,
    pub filled_slices: AtomicU64,
    pub active: AtomicU8,
    _padding: [u8; 23],
}

impl IcebergOrder {
    pub const fn new(
        order_id: u64,
        total_qty: FixedQty,
        display_qty: FixedQty,
        min_pct: u64,
        max_pct: u64,
    ) -> Self {
        Self {
            order_id: AtomicU64::new(order_id),
            total_qty: AtomicI64::new(total_qty),
            remaining_qty: AtomicI64::new(total_qty),
            display_qty_target: AtomicI64::new(display_qty),
            current_display_qty: AtomicI64::new(display_qty),
            min_slice_pct: AtomicU64::new(min_pct),
            max_slice_pct: AtomicU64::new(max_pct),
            randomization_seed: AtomicU64::new(0x5DEECE66D), // LCG seed
            slice_count: AtomicU64::new(0),
            filled_slices: AtomicU64::new(0),
            active: AtomicU8::new(1),
            _padding: [0u8; 23],
        }
    }

    /// Simple LCG random number generator (lock-free, deterministic)
    #[inline]
    fn next_random(&self) -> u64 {
        let seed = self.randomization_seed.fetch_add(0x5DEECE66D, Ordering::AcqRel);
        ((seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & 0xFFFFFFFFFFFF)
    }

    /// Calculate randomized slice size within bounds
    #[inline]
    pub fn calculate_slice_size(&self, remaining: FixedQty) -> FixedQty {
        let target = self.display_qty_target.load(Ordering::Acquire);
        let min_pct = self.min_slice_pct.load(Ordering::Acquire);
        let max_pct = self.max_slice_pct.load(Ordering::Acquire);
        
        // Generate random factor between min_pct and max_pct
        let rand = self.next_random();
        let range = max_pct - min_pct;
        let factor = if range > 0 {
            min_pct + (rand % (range + 1))
        } else {
            min_pct
        };
        
        // Calculate slice: target * factor / 10000 (basis points)
        let base_slice = (target * factor as i64) / 10000;
        
        // Ensure we don't exceed remaining
        core::cmp::min(base_slice, remaining)
    }

    /// Get next display quantity (called when previous slice is filled)
    #[inline]
    pub fn refresh_display(&self) -> FixedQty {
        let remaining = self.remaining_qty.load(Ordering::Acquire);
        
        if remaining <= 0 {
            self.active.store(0, Ordering::Release);
            return 0;
        }
        
        let slice_size = self.calculate_slice_size(remaining);
        self.current_display_qty.store(slice_size, Ordering::Release);
        self.slice_count.fetch_add(1, Ordering::AcqRel);
        
        slice_size
    }

    /// Record fill of child order
    #[inline]
    pub fn record_fill(&self, fill_qty: FixedQty) -> bool {
        let remaining = self.remaining_qty.fetch_sub(fill_qty, Ordering::AcqRel) - fill_qty;
        
        if remaining <= 0 {
            self.active.store(0, Ordering::Release);
            self.remaining_qty.store(0, Ordering::Release);
            return true; // Order complete
        }
        
        false
    }

    /// Check if order needs refresh (display qty depleted)
    #[inline]
    pub fn needs_refresh(&self, filled_so_far: FixedQty) -> bool {
        let current_display = self.current_display_qty.load(Ordering::Acquire);
        filled_so_far >= current_display
    }

    /// Get statistics about the iceberg
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, FixedQty, FixedQty) {
        (
            self.slice_count.load(Ordering::Acquire),
            self.filled_slices.load(Ordering::Acquire),
            self.remaining_qty.load(Ordering::Acquire),
            self.current_display_qty.load(Ordering::Acquire),
        )
    }
}

/// Circular buffer for tracking slice history (lock-free)
#[repr(C, align(64))]
pub struct SliceHistoryBuffer {
    head: AtomicU64,
    tail: AtomicU64,
    buffer: [AtomicI64; MAX_SLICES],
    timestamps: [AtomicU64; MAX_SLICES],
    _padding: [u8; 32],
}

impl SliceHistoryBuffer {
    pub const fn new() -> Self {
        const INIT: AtomicI64 = AtomicI64::new(0);
        const TS_INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            buffer: [INIT; MAX_SLICES],
            timestamps: [TS_INIT; MAX_SLICES],
            _padding: [0u8; 32],
        }
    }

    /// Push new slice size to buffer (O(1))
    #[inline]
    pub fn push(&self, slice_qty: FixedQty) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        let head = self.head.load(Ordering::Acquire);
        let idx = (head as usize) & SLICE_MASK;
        
        self.buffer[idx].store(slice_qty, Ordering::Release);
        self.timestamps[idx].store(ts, Ordering::Release);
        
        // Check if buffer is full
        let tail = self.tail.load(Ordering::Acquire);
        if head - tail >= MAX_SLICES as u64 {
            self.tail.fetch_add(1, Ordering::AcqRel);
        }
        
        self.head.fetch_add(1, Ordering::Release);
    }

    /// Get average slice size from history
    #[inline]
    pub fn get_average_slice(&self) -> FixedQty {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let count = (head - tail) as usize;
        
        if count == 0 {
            return 0;
        }
        
        let mut sum: FixedQty = 0;
        let start = if count < MAX_SLICES {
            tail as usize
        } else {
            head as usize - MAX_SLICES
        };
        
        // Manual unroll for branch prediction
        let mut i = 0;
        while i < count {
            let idx = ((start + i) & SLICE_MASK);
            sum += self.buffer[idx].load(Ordering::Acquire);
            i += 1;
        }
        
        sum / count as FixedQty
    }

    /// Get slice size from N steps ago
    #[inline]
    pub fn get_historical_slice(&self, steps_back: usize) -> FixedQty {
        let head = self.head.load(Ordering::Acquire);
        if steps_back == 0 || steps_back > MAX_SLICES {
            return 0;
        }
        
        let idx = ((head - steps_back as u64) as usize) & SLICE_MASK;
        self.buffer[idx].load(Ordering::Acquire)
    }

    /// Get timestamp delta between last two slices (for timing analysis)
    #[inline]
    pub fn get_last_slice_delta(&self) -> TscTicks {
        let head = self.head.load(Ordering::Acquire);
        if head < 2 {
            return 0;
        }
        
        let idx1 = ((head - 1) as usize) & SLICE_MASK;
        let idx2 = ((head - 2) as usize) & SLICE_MASK;
        
        let ts1 = self.timestamps[idx1].load(Ordering::Acquire);
        let ts2 = self.timestamps[idx2].load(Ordering::Acquire);
        
        ts1.wrapping_sub(ts2)
    }
}

/// Adaptive slicer that adjusts based on market conditions
#[repr(C, align(64))]
pub struct AdaptiveIcebergSlicer {
    pub base_display_qty: AtomicI64,
    pub volatility_factor: AtomicU64, // Scaled by 1000
    pub volume_participation_rate: AtomicU64, // Basis points
    pub min_wait_ticks: AtomicU64,
    pub max_wait_ticks: AtomicU64,
    pub last_slice_ticks: AtomicU64,
    pub adjustment_counter: AtomicU64,
    _padding: [u8; 24],
}

impl AdaptiveIcebergSlicer {
    pub const fn new(base_qty: FixedQty, vol_factor: u64, participation_bps: u64) -> Self {
        Self {
            base_display_qty: AtomicI64::new(base_qty),
            volatility_factor: AtomicU64::new(vol_factor),
            volume_participation_rate: AtomicU64::new(participation_bps),
            min_wait_ticks: AtomicU64::new(1000),
            max_wait_ticks: AtomicU64::new(100000),
            last_slice_ticks: AtomicU64::new(0),
            adjustment_counter: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }

    /// Calculate adaptive display quantity based on volatility
    #[inline]
    pub fn calculate_adaptive_qty(&self, market_volume: FixedQty) -> FixedQty {
        let base = self.base_display_qty.load(Ordering::Acquire);
        let vol_factor = self.volatility_factor.load(Ordering::Acquire);
        let part_rate = self.volume_participation_rate.load(Ordering::Acquire);
        
        // Adjust base by volatility: higher vol = smaller slices
        let vol_adjusted = if vol_factor > 1000 {
            (base * 1000) / vol_factor
        } else {
            base
        };
        
        // Apply participation rate limit
        let max_by_participation = (market_volume * part_rate as i64) / 10000;
        
        core::cmp::min(vol_adjusted, max_by_participation)
    }

    /// Check if enough time has passed since last slice
    #[inline]
    pub fn can_submit_slice(&self) -> bool {
        let now = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        let last = self.last_slice_ticks.load(Ordering::Acquire);
        let min_wait = self.min_wait_ticks.load(Ordering::Acquire);
        
        now.wrapping_sub(last) >= min_wait
    }

    /// Record slice submission
    #[inline]
    pub fn record_submission(&self) {
        let now = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.last_slice_ticks.store(now, Ordering::Release);
        self.adjustment_counter.fetch_add(1, Ordering::AcqRel);
    }

    /// Update volatility factor dynamically
    #[inline]
    pub fn update_volatility(&self, new_factor: u64) {
        self.volatility_factor.store(new_factor, Ordering::Release);
    }
}

/// Shadow mode logger for testing iceberg strategies
#[repr(C, align(64))]
pub struct ShadowIcebergLogger {
    pub enabled: AtomicU8,
    pub theoretical_fills: AtomicU64,
    pub actual_fills: AtomicU64,
    pub avg_slippage: AtomicI64,
    pub sample_count: AtomicU64,
    _padding: [u8; 32],
}

impl ShadowIcebergLogger {
    pub const fn new() -> Self {
        Self {
            enabled: AtomicU8::new(0),
            theoretical_fills: AtomicU64::new(0),
            actual_fills: AtomicU64::new(0),
            avg_slippage: AtomicI64::new(0),
            sample_count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    #[inline]
    pub fn log_theoretical(&self, fill_qty: FixedQty, price: i64) {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return;
        }
        
        self.theoretical_fills.fetch_add(fill_qty as u64, Ordering::AcqRel);
        let count = self.sample_count.fetch_add(1, Ordering::AcqRel);
        
        // Running average slippage calculation
        let current_avg = self.avg_slippage.load(Ordering::Acquire);
        let new_avg = ((current_avg * count as i64) + price) / (count + 1) as i64;
        self.avg_slippage.store(new_avg, Ordering::Release);
    }

    #[inline]
    pub fn log_actual(&self, fill_qty: FixedQty) {
        self.actual_fills.fetch_add(fill_qty as u64, Ordering::AcqRel);
    }

    #[inline]
    pub fn enable(&self) {
        self.enabled.store(1, Ordering::Release);
    }

    #[inline]
    pub fn disable(&self) {
        self.enabled.store(0, Ordering::Release);
    }

    #[inline]
    pub fn get_effectiveness(&self) -> u64 {
        let actual = self.actual_fills.load(Ordering::Acquire);
        let theoretical = self.theoretical_fills.load(Ordering::Acquire);
        
        if theoretical == 0 {
            return 0;
        }
        
        (actual * 100) / theoretical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_creation() {
        let iceberg = IcebergOrder::new(12345, 10_000_000_000i64, 1_000_000_000i64, 8000, 12000);
        
        assert_eq!(iceberg.total_qty.load(Ordering::Acquire), 10_000_000_000i64);
        assert_eq!(iceberg.remaining_qty.load(Ordering::Acquire), 10_000_000_000i64);
        assert_eq!(iceberg.active.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_slice_size_randomization() {
        let iceberg = IcebergOrder::new(12346, 10_000_000_000i64, 1_000_000_000i64, 8000, 12000);
        
        let slice1 = iceberg.calculate_slice_size(10_000_000_000i64);
        let slice2 = iceberg.calculate_slice_size(10_000_000_000i64);
        
        // Slices should be different due to randomization
        // Both should be within 80%-120% of target
        assert!(slice1 >= 800_000_000i64 && slice1 <= 1_200_000_000i64);
        assert!(slice2 >= 800_000_000i64 && slice2 <= 1_200_000_000i64);
    }

    #[test]
    fn test_iceberg_refresh_cycle() {
        let iceberg = IcebergOrder::new(12347, 5_000_000_000i64, 500_000_000i64, 10000, 10000);
        
        let slice1 = iceberg.refresh_display();
        assert_eq!(slice1, 500_000_000i64);
        
        iceberg.record_fill(500_000_000i64);
        assert_eq!(iceberg.remaining_qty.load(Ordering::Acquire), 4_500_000_000i64);
        
        let slice2 = iceberg.refresh_display();
        assert_eq!(slice2, 500_000_000i64);
    }

    #[test]
    fn test_iceberg_completion() {
        let iceberg = IcebergOrder::new(12348, 1_000_000_000i64, 250_000_000i64, 10000, 10000);
        
        iceberg.refresh_display();
        iceberg.record_fill(250_000_000i64);
        iceberg.refresh_display();
        iceberg.record_fill(250_000_000i64);
        iceberg.refresh_display();
        iceberg.record_fill(250_000_000i64);
        iceberg.refresh_display();
        let done = iceberg.record_fill(250_000_000i64);
        
        assert!(done);
        assert_eq!(iceberg.active.load(Ordering::Acquire), 0);
        assert_eq!(iceberg.remaining_qty.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_slice_history_buffer() {
        let buffer = SliceHistoryBuffer::new();
        
        buffer.push(100_000_000i64);
        buffer.push(150_000_000i64);
        buffer.push(200_000_000i64);
        
        assert_eq!(buffer.get_average_slice(), 150_000_000i64);
        assert_eq!(buffer.get_historical_slice(1), 150_000_000i64);
        assert_eq!(buffer.get_historical_slice(2), 100_000_000i64);
    }

    #[test]
    fn test_adaptive_slicer() {
        let slicer = AdaptiveIcebergSlicer::new(1_000_000_000i64, 1500, 500); // 1.5x vol, 5% participation
        
        let adaptive_qty = slicer.calculate_adaptive_qty(10_000_000_000i64);
        
        // Should be reduced by volatility factor and limited by participation
        assert!(adaptive_qty <= 500_000_000i64); // 5% of 10B
    }

    #[test]
    fn test_shadow_logger() {
        let logger = ShadowIcebergLogger::new();
        logger.enable();
        
        logger.log_theoretical(100_000_000i64, 50_000_000_000i64);
        logger.log_theoretical(100_000_000i64, 50_100_000_000i64);
        logger.log_actual(180_000_000i64);
        
        assert_eq!(logger.get_effectiveness(), 90); // 180/200 = 90%
    }
}
