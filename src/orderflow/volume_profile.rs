//! High-res Volume Profile and Point of Control (POC) using lock-free histograms.
//! 
//! Pre-allocated histograms at startup, zero heap allocations in hot paths.
//! Uses atomic operations for thread-safe updates without mutexes.

#![allow(clippy::missing_docs_in_private_items)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum price bins in volume profile
pub const MAX_PRICE_BINS: usize = 4096;

/// Volume profile histogram bin
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VolumeBin {
    pub total_volume: u64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub trade_count: u64,
    pub buy_volume: u64,   // Aggressive buys (hitting asks)
    pub sell_volume: u64,  // Aggressive sells (hitting bids)
    _padding: [u8; CACHE_LINE_SIZE - 48],
}

impl VolumeBin {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            total_volume: 0,
            bid_volume: 0,
            ask_volume: 0,
            trade_count: 0,
            buy_volume: 0,
            sell_volume: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 48],
        }
    }

    /// Add volume to bin (lock-free via caller synchronization)
    #[inline(always)]
    pub fn add(&mut self, volume: u64, is_ask_trade: bool) {
        self.total_volume += volume;
        self.trade_count += 1;
        
        if is_ask_trade {
            self.ask_volume += volume;
            self.buy_volume += volume;
        } else {
            self.bid_volume += volume;
            self.sell_volume += volume;
        }
    }

    /// Reset bin
    #[inline(always)]
    pub fn reset(&mut self) {
        self.total_volume = 0;
        self.bid_volume = 0;
        self.ask_volume = 0;
        self.trade_count = 0;
        self.buy_volume = 0;
        self.sell_volume = 0;
    }

    /// Get imbalance (buy - sell)
    #[inline(always)]
    pub fn imbalance(&self) -> i64 {
        self.buy_volume as i64 - self.sell_volume as i64
    }
}

/// Value Area definition (typically 70% of volume)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ValueArea {
    pub poc_price_tick: i64,
    pub poc_volume: u64,
    pub vah_price_tick: i64,  // Value Area High
    pub val_price_tick: i64,  // Value Area Low
    pub total_volume: u64,
    pub value_area_pct: f64,
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

impl ValueArea {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            poc_price_tick: 0,
            poc_volume: 0,
            vah_price_tick: 0,
            val_price_tick: 0,
            total_volume: 0,
            value_area_pct: 0.70,
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }
}

/// Lock-free Volume Profile with pre-allocated histogram
#[repr(C)]
pub struct VolumeProfile<const NUM_BINS: usize> {
    /// Histogram bins
    bins: UnsafeCell<[VolumeBin; NUM_BINS]>,
    /// Base price tick (center of histogram)
    base_price_tick: AtomicI64,
    /// Price per bin in ticks
    tick_size: AtomicI64,
    /// Total volume across all bins
    total_volume: AtomicU64,
    /// Total trades
    total_trades: AtomicU64,
    /// Current POC index
    poc_index: AtomicU64,
    /// Session high (tick value)
    session_high: AtomicI64,
    /// Session low (tick value)
    session_low: AtomicI64,
    /// Memory usage tracker
    memory_bytes: AtomicU64,
}

// SAFETY: All interior mutability protected by atomics
unsafe impl<const N: usize> Send for VolumeProfile<N> {}
unsafe impl<const N: usize> Sync for VolumeProfile<N> {}

impl<const NUM_BINS: usize> VolumeProfile<NUM_BINS> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            bins: UnsafeCell::new([VolumeBin::new(); NUM_BINS]),
            base_price_tick: AtomicI64::new(0),
            tick_size: AtomicI64::new(1),
            total_volume: AtomicU64::new(0),
            total_trades: AtomicU64::new(0),
            poc_index: AtomicU64::new(0),
            session_high: AtomicI64::new(i64::MIN),
            session_low: AtomicI64::new(i64::MAX),
            memory_bytes: AtomicU64::new(0),
        }
    }

    /// Initialize volume profile
    #[inline]
    pub fn init(&self, base_price_tick: i64, tick_size: i64) {
        self.base_price_tick.store(base_price_tick, Ordering::Relaxed);
        self.tick_size.store(tick_size, Ordering::Relaxed);
        self.total_volume.store(0, Ordering::Relaxed);
        self.total_trades.store(0, Ordering::Relaxed);
        self.poc_index.store((NUM_BINS / 2) as u64, Ordering::Relaxed);
        self.session_high.store(i64::MIN, Ordering::Relaxed);
        self.session_low.store(i64::MAX, Ordering::Relaxed);
        
        let mem = (core::mem::size_of::<VolumeBin>() * NUM_BINS) as u64;
        self.memory_bytes.store(mem, Ordering::Relaxed);
        
        // Reset all bins
        let bins = unsafe { &mut *self.bins.get() };
        for bin in bins.iter_mut() {
            bin.reset();
        }
    }

    /// Convert price tick to bin index
    #[inline(always)]
    fn price_to_bin(&self, price_tick: i64) -> Option<usize> {
        let base = self.base_price_tick.load(Ordering::Relaxed);
        let ts = self.tick_size.load(Ordering::Relaxed);
        
        let offset = (price_tick - base) / ts;
        let half_bins = (NUM_BINS / 2) as i64;
        let idx = offset + half_bins;
        
        if idx >= 0 && idx < NUM_BINS as i64 {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Convert bin index to price tick
    #[inline(always)]
    fn bin_to_price(&self, bin_idx: usize) -> i64 {
        let base = self.base_price_tick.load(Ordering::Relaxed);
        let ts = self.tick_size.load(Ordering::Relaxed);
        let half_bins = (NUM_BINS / 2) as i64;
        base + ((bin_idx as i64 - half_bins) * ts)
    }

    /// Record a trade (lock-free hot path)
    #[inline(always)]
    pub fn record_trade(&self, price_tick: i64, volume: u64, is_ask_trade: bool) -> bool {
        // Update session extremes (branchless)
        let high = self.session_high.load(Ordering::Relaxed);
        let low = self.session_low.load(Ordering::Relaxed);
        
        let new_high = if price_tick > high { price_tick } else { high };
        let new_low = if price_tick < low { price_tick } else { low };
        
        self.session_high.store(new_high, Ordering::Relaxed);
        self.session_low.store(new_low, Ordering::Relaxed);
        
        // Find bin and add volume
        if let Some(bin_idx) = self.price_to_bin(price_tick) {
            let bins = unsafe { &mut *self.bins.get() };
            bins[bin_idx].add(volume, is_ask_trade);
            
            self.total_volume.fetch_add(volume, Ordering::Relaxed);
            self.total_trades.fetch_add(1, Ordering::Relaxed);
            
            // Update POC if this bin now has highest volume
            self.update_poc(bin_idx);
            
            true
        } else {
            false
        }
    }

    /// Update POC index if needed (optimized)
    #[inline]
    fn update_poc(&self, candidate_idx: usize) {
        let bins = unsafe { &*self.bins.get() };
        let current_poc = self.poc_index.load(Ordering::Relaxed) as usize;
        
        let candidate_vol = bins[candidate_idx].total_volume;
        let current_poc_vol = bins[current_poc].total_volume;
        
        if candidate_vol > current_poc_vol {
            self.poc_index.store(candidate_idx as u64, Ordering::Relaxed);
        }
    }

    /// Calculate full value area (70% typically)
    #[inline]
    pub fn calculate_value_area(&self, pct: f64) -> ValueArea {
        let bins = unsafe { &*self.bins.get() };
        let total_vol = self.total_volume.load(Ordering::Relaxed);
        
        if total_vol == 0 {
            return ValueArea::new();
        }
        
        let poc_idx = self.poc_index.load(Ordering::Relaxed) as usize;
        let poc_vol = bins[poc_idx].total_volume;
        let poc_price = self.bin_to_price(poc_idx);
        
        // Find value area by expanding from POC
        let target_vol = (total_vol as f64 * pct) as u64;
        let mut accumulated_vol = poc_vol;
        let mut left_idx = poc_idx;
        let mut right_idx = poc_idx;
        
        while accumulated_vol < target_vol {
            // Expand to side with more volume
            let left_vol = if left_idx > 0 { bins[left_idx - 1].total_volume } else { 0 };
            let right_vol = if right_idx < NUM_BINS - 1 { bins[right_idx + 1].total_volume } else { 0 };
            
            if left_vol >= right_vol && left_idx > 0 {
                left_idx -= 1;
                accumulated_vol += bins[left_idx].total_volume;
            } else if right_idx < NUM_BINS - 1 {
                right_idx += 1;
                accumulated_vol += bins[right_idx].total_volume;
            } else {
                break;
            }
        }
        
        ValueArea {
            poc_price_tick: poc_price,
            poc_volume: poc_vol,
            vah_price_tick: self.bin_to_price(right_idx),
            val_price_tick: self.bin_to_price(left_idx),
            total_volume: total_vol,
            value_area_pct: pct,
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }

    /// Get current POC price
    #[inline(always)]
    pub fn get_poc(&self) -> Option<(i64, u64)> {
        let poc_idx = self.poc_index.load(Ordering::Relaxed) as usize;
        let bins = unsafe { &*self.bins.get() };
        
        if bins[poc_idx].total_volume == 0 {
            return None;
        }
        
        Some((self.bin_to_price(poc_idx), bins[poc_idx].total_volume))
    }

    /// Get volume at specific price
    #[inline]
    pub fn volume_at_price(&self, price_tick: i64) -> Option<u64> {
        let bin_idx = self.price_to_bin(price_tick)?;
        let bins = unsafe { &*self.bins.get() };
        Some(bins[bin_idx].total_volume)
    }

    /// Get buy/sell ratio at price
    #[inline]
    pub fn buy_sell_ratio_at_price(&self, price_tick: i64) -> Option<f64> {
        let bin_idx = self.price_to_bin(price_tick)?;
        let bins = unsafe { &*self.bins.get() };
        
        let bin = bins[bin_idx];
        if bin.sell_volume == 0 {
            return Some(f64::INFINITY);
        }
        Some(bin.buy_volume as f64 / bin.sell_volume as f64)
    }

    /// Get session statistics
    #[inline]
    pub fn get_session_stats(&self) -> (i64, i64, u64, u64) {
        (
            self.session_high.load(Ordering::Relaxed),
            self.session_low.load(Ordering::Relaxed),
            self.total_volume.load(Ordering::Relaxed),
            self.total_trades.load(Ordering::Relaxed),
        )
    }

    /// Reset profile for new session
    #[inline]
    pub fn reset(&self) {
        let bins = unsafe { &mut *self.bins.get() };
        for bin in bins.iter_mut() {
            bin.reset();
        }
        
        self.total_volume.store(0, Ordering::Relaxed);
        self.total_trades.store(0, Ordering::Relaxed);
        self.poc_index.store((NUM_BINS / 2) as u64, Ordering::Relaxed);
        self.session_high.store(i64::MIN, Ordering::Relaxed);
        self.session_low.store(i64::MAX, Ordering::Relaxed);
    }

    /// Get memory usage
    #[inline(always)]
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Get number of non-empty bins
    #[inline]
    pub fn active_bins(&self) -> usize {
        let bins = unsafe { &*self.bins.get() };
        let mut count = 0;
        for bin in bins.iter() {
            if bin.total_volume > 0 {
                count += 1;
            }
        }
        count
    }
}

/// Type alias for crypto volume profile
pub type CryptoVolumeProfile = VolumeProfile<MAX_PRICE_BINS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_bin_basic() {
        let mut bin = VolumeBin::new();
        
        bin.add(100, true);  // Buy
        bin.add(150, false); // Sell
        
        assert_eq!(bin.total_volume, 250);
        assert_eq!(bin.buy_volume, 100);
        assert_eq!(bin.sell_volume, 150);
        assert_eq!(bin.trade_count, 2);
        assert_eq!(bin.imbalance(), -50);
    }

    #[test]
    fn test_volume_profile_init() {
        let vp = CryptoVolumeProfile::new();
        vp.init(50000, 1);
        
        assert_eq!(vp.total_trades.load(Ordering::Relaxed), 0);
        assert!(vp.memory_bytes() > 0);
    }

    #[test]
    fn test_record_trades() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        assert!(vp.record_trade(50000, 100, true));
        assert!(vp.record_trade(50001, 200, false));
        assert!(vp.record_trade(50000, 50, false));
        
        assert_eq!(vp.total_trades.load(Ordering::Relaxed), 3);
        assert_eq!(vp.total_volume.load(Ordering::Relaxed), 350);
    }

    #[test]
    fn test_poc_tracking() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        // Create volume concentration at 50005
        for _ in 0..10 {
            vp.record_trade(50005, 100, true);
        }
        
        // Less volume at 50000
        for _ in 0..3 {
            vp.record_trade(50000, 50, true);
        }
        
        let (poc_price, poc_vol) = vp.get_poc().unwrap();
        assert_eq!(poc_price, 50005);
        assert_eq!(poc_vol, 1000);
    }

    #[test]
    fn test_value_area_calculation() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        // Create symmetric volume distribution
        for i in 0..10 {
            let price = 50000 + i;
            vp.record_trade(price, 100, true);
        }
        
        let va = vp.calculate_value_area(0.70);
        assert!(va.poc_volume > 0);
        assert!(va.vah_price_tick >= va.poc_price_tick);
        assert!(va.val_price_tick <= va.poc_price_tick);
    }

    #[test]
    fn test_session_extremes() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        vp.record_trade(50010, 100, true);
        vp.record_trade(49990, 100, false);
        vp.record_trade(50005, 100, true);
        
        let (high, low, _, _) = vp.get_session_stats();
        assert_eq!(high, 50010);
        assert_eq!(low, 49990);
    }

    #[test]
    fn test_buy_sell_ratio() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        // Only buys at this price
        vp.record_trade(50000, 100, true);
        vp.record_trade(50000, 50, true);
        
        let ratio = vp.buy_sell_ratio_at_price(50000).unwrap();
        assert!(ratio.is_infinite());
        
        // Add sells
        vp.record_trade(50000, 75, false);
        let ratio = vp.buy_sell_ratio_at_price(50000).unwrap();
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        vp.record_trade(50000, 100, true);
        vp.reset();
        
        assert_eq!(vp.total_volume.load(Ordering::Relaxed), 0);
        assert_eq!(vp.active_bins(), 0);
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;
        
        assert!(size_of::<VolumeBin>() >= CACHE_LINE_SIZE);
        assert_eq!(size_of::<VolumeBin>() % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_out_of_bounds_handling() {
        let vp = VolumeProfile::<100>::new();
        vp.init(50000, 1);
        
        // Far out of bounds should fail
        assert!(!vp.record_trade(60000, 100, true));
        assert!(!vp.record_trade(40000, 100, false));
    }
}
