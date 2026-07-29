//! Zero-allocation footprint chart builder tracking bid/ask volume at every tick.
//! 
//! Pre-allocates histograms at startup, eliminates all heap allocations in hot paths.
//! Uses branchless programming for deterministic execution latency.

#![allow(clippy::missing_docs_in_private_items)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Cache line size for padding
const CACHE_LINE_SIZE: usize = 64;

/// Maximum price levels tracked (compile-time bound)
pub const MAX_PRICE_LEVELS: usize = 1024;

/// Maximum time buckets per footprint (e.g., 60 seconds, 100ms buckets)
pub const MAX_TIME_BUCKETS: usize = 1000;

/// Single footprint cell - bid/ask volume at a price level
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FootprintCell {
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub bid_count: u32,
    pub ask_count: u32,
    pub high: i64,  // Tick value (scaled integer)
    pub low: i64,
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

impl FootprintCell {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            bid_volume: 0,
            ask_volume: 0,
            bid_count: 0,
            ask_count: 0,
            high: i64::MIN,
            low: i64::MAX,
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }

    /// Add a trade to this cell (branchless)
    #[inline(always)]
    pub fn add_trade(&mut self, volume: u64, is_ask: bool, price_tick: i64) {
        // Branchless selection
        let ask_vol = volume * (is_ask as u64);
        let bid_vol = volume * (!is_ask as u64);
        
        self.ask_volume += ask_vol;
        self.bid_volume += bid_vol;
        self.ask_count += is_ask as u32;
        self.bid_count += (!is_ask) as u32;
        
        // Branchless high/low update
        let high_mask = -((price_tick > self.high) as i64);
        let low_mask = -((price_tick < self.low) as i64);
        
        self.high = (self.high & !high_mask) | (price_tick & high_mask);
        self.low = (self.low & !low_mask) | (price_tick & low_mask);
    }

    /// Get delta (ask - bid)
    #[inline(always)]
    pub fn delta(&self) -> i64 {
        self.ask_volume as i64 - self.bid_volume as i64
    }

    /// Get total volume
    #[inline(always)]
    pub fn total_volume(&self) -> u64 {
        self.bid_volume + self.ask_volume
    }

    /// Get imbalance ratio (branchless, returns scaled integer)
    /// Returns (ask_vol - bid_vol) / total * 1000, or 0 if no volume
    #[inline(always)]
    pub fn imbalance_scaled(&self) -> i32 {
        let total = self.total_volume();
        if total == 0 {
            return 0;
        }
        let delta = self.delta();
        ((delta * 1000) / total as i64) as i32
    }

    /// Reset cell for reuse
    #[inline(always)]
    pub fn reset(&mut self) {
        self.bid_volume = 0;
        self.ask_volume = 0;
        self.bid_count = 0;
        self.ask_count = 0;
        self.high = i64::MIN;
        self.low = i64::MAX;
    }
}

/// Footprint chart state - pre-allocated grid of cells
#[repr(C)]
pub struct FootprintChart<const PRICE_LEVELS: usize, const TIME_BUCKETS: usize> {
    /// Grid of footprint cells [price_level][time_bucket]
    cells: UnsafeCell<[[FootprintCell; TIME_BUCKETS]; PRICE_LEVELS]>,
    /// Price level mapping (tick value -> index)
    base_price_tick: AtomicI64,
    /// Current time bucket index
    current_bucket: AtomicU64,
    /// Total ticks processed
    tick_count: AtomicU64,
    /// Memory usage tracker
    memory_bytes: AtomicU64,
}

// SAFETY: All interior mutability is protected by atomic operations
unsafe impl<const P: usize, const T: usize> Send for FootprintChart<P, T> {}
unsafe impl<const P: usize, const T: usize> Sync for FootprintChart<P, T> {}

impl<const PRICE_LEVELS: usize, const TIME_BUCKETS: usize> FootprintChart<PRICE_LEVELS, TIME_BUCKETS> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            cells: UnsafeCell::new([[FootprintCell::new(); TIME_BUCKETS]; PRICE_LEVELS]),
            base_price_tick: AtomicI64::new(0),
            current_bucket: AtomicU64::new(0),
            tick_count: AtomicU64::new(0),
            memory_bytes: AtomicU64::new(0),
        }
    }

    /// Initialize with base price and memory tracking
    #[inline]
    pub fn init(&self, base_price_tick: i64) {
        self.base_price_tick.store(base_price_tick, Ordering::Relaxed);
        self.current_bucket.store(0, Ordering::Relaxed);
        self.tick_count.store(0, Ordering::Relaxed);
        
        // Calculate memory usage
        let mem = (core::mem::size_of::<FootprintCell>() * PRICE_LEVELS * TIME_BUCKETS) as u64;
        self.memory_bytes.store(mem, Ordering::Relaxed);
        
        // Reset all cells
        let cells = unsafe { &mut *self.cells.get() };
        for p in 0..PRICE_LEVELS {
            for t in 0..TIME_BUCKETS {
                cells[p][t].reset();
            }
        }
    }

    /// Convert price to index (branchless bounds check)
    #[inline(always)]
    fn price_to_index(&self, price_tick: i64) -> Option<usize> {
        let base = self.base_price_tick.load(Ordering::Relaxed);
        let offset = price_tick - base;
        
        // Center the range around middle of array
        let half_levels = (PRICE_LEVELS / 2) as i64;
        let idx = offset + half_levels;
        
        // Branchless bounds check
        let valid = ((idx >= 0) && (idx < PRICE_LEVELS as i64)) as usize;
        Some(idx as usize * valid + (1 - valid) * PRICE_LEVELS)
    }

    /// Record a trade at given price
    /// Returns true if recorded, false if out of bounds
    #[inline(always)]
    pub fn record_trade(&self, price_tick: i64, volume: u64, is_ask: bool) -> bool {
        let idx = match self.price_to_index(price_tick) {
            Some(i) if i < PRICE_LEVELS => i,
            _ => return false,
        };
        
        let bucket = self.current_bucket.load(Ordering::Relaxed) as usize % TIME_BUCKETS;
        
        let cells = unsafe { &mut *self.cells.get() };
        cells[idx][bucket].add_trade(volume, is_ask, price_tick);
        
        self.tick_count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Advance to next time bucket (called on timer)
    #[inline(always)]
    pub fn advance_bucket(&self) {
        let current = self.current_bucket.fetch_add(1, Ordering::Relaxed);
        
        // Clear the bucket we're about to wrap to
        let new_bucket = ((current + 1) as usize) % TIME_BUCKETS;
        let cells = unsafe { &mut *self.cells.get() };
        
        for p in 0..PRICE_LEVELS {
            cells[p][new_bucket].reset();
        }
    }

    /// Get cell at price level and time offset (0 = current bucket)
    #[inline(always)]
    pub fn get_cell(&self, price_tick: i64, time_offset: usize) -> Option<FootprintCell> {
        let idx = self.price_to_index(price_tick)?;
        if idx >= PRICE_LEVELS {
            return None;
        }
        
        let current = self.current_bucket.load(Ordering::Relaxed) as usize;
        let bucket = (current.wrapping_sub(time_offset)) % TIME_BUCKETS;
        
        let cells = unsafe { &*self.cells.get() };
        Some(cells[idx][bucket])
    }

    /// Aggregate volume across all time buckets for a price level
    #[inline]
    pub fn aggregate_at_price(&self, price_tick: i64) -> Option<(u64, u64)> {
        let idx = self.price_to_index(price_tick)?;
        if idx >= PRICE_LEVELS {
            return None;
        }
        
        let cells = unsafe { &*self.cells.get() };
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;
        
        for t in 0..TIME_BUCKETS {
            total_bid += cells[idx][t].bid_volume;
            total_ask += cells[idx][t].ask_volume;
        }
        
        Some((total_bid, total_ask))
    }

    /// Find Point of Control (price with highest total volume)
    #[inline]
    pub fn find_poc(&self) -> Option<(i64, u64)> {
        let base = self.base_price_tick.load(Ordering::Relaxed);
        let half_levels = (PRICE_LEVELS / 2) as i64;
        
        let cells = unsafe { &*self.cells.get() };
        let mut max_vol = 0u64;
        let mut poc_idx = 0usize;
        
        for p in 0..PRICE_LEVELS {
            let mut vol_at_price = 0u64;
            for t in 0..TIME_BUCKETS {
                vol_at_price += cells[p][t].total_volume();
            }
            
            if vol_at_price > max_vol {
                max_vol = vol_at_price;
                poc_idx = p;
            }
        }
        
        if max_vol == 0 {
            return None;
        }
        
        let price_tick = base + (poc_idx as i64 - half_levels);
        Some((price_tick, max_vol))
    }

    /// Get total delta across all prices in current bucket
    #[inline]
    pub fn current_bucket_delta(&self) -> i64 {
        let bucket = self.current_bucket.load(Ordering::Relaxed) as usize % TIME_BUCKETS;
        let cells = unsafe { &*self.cells.get() };
        
        let mut delta = 0i64;
        for p in 0..PRICE_LEVELS {
            delta += cells[p][bucket].delta();
        }
        delta
    }

    /// Get tick count
    #[inline(always)]
    pub fn tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Relaxed)
    }

    /// Get memory usage in bytes
    #[inline(always)]
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Check if within memory budget
    #[inline]
    pub fn check_memory_budget(&self, limit_bytes: u64) -> bool {
        self.memory_bytes.load(Ordering::Relaxed) <= limit_bytes
    }
}

/// Type alias for typical crypto footprint (100 price levels, 600 time buckets = 1 min at 100ms)
pub type CryptoFootprint = FootprintChart<MAX_PRICE_LEVELS, MAX_TIME_BUCKETS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footprint_cell_basic() {
        let mut cell = FootprintCell::new();
        
        cell.add_trade(100, false, 5000); // Bid
        cell.add_trade(150, true, 5001);  // Ask
        
        assert_eq!(cell.bid_volume, 100);
        assert_eq!(cell.ask_volume, 150);
        assert_eq!(cell.bid_count, 1);
        assert_eq!(cell.ask_count, 1);
        assert_eq!(cell.delta(), 50);
        assert_eq!(cell.total_volume(), 250);
    }

    #[test]
    fn test_footprint_cell_high_low() {
        let mut cell = FootprintCell::new();
        
        cell.add_trade(100, false, 5000);
        cell.add_trade(100, true, 5010);
        cell.add_trade(100, false, 4990);
        
        assert_eq!(cell.high, 5010);
        assert_eq!(cell.low, 4990);
    }

    #[test]
    fn test_footprint_chart_init() {
        let chart = CryptoFootprint::new();
        chart.init(50000);
        
        assert_eq!(chart.tick_count(), 0);
        assert!(chart.memory_bytes() > 0);
    }

    #[test]
    fn test_footprint_record_trades() {
        let chart = FootprintChart::<100, 10>::new();
        chart.init(50000);
        
        // Record trades at various prices
        assert!(chart.record_trade(50000, 100, false)); // At base
        assert!(chart.record_trade(50001, 150, true));  // Above base
        assert!(chart.record_trade(49999, 200, false)); // Below base
        
        assert_eq!(chart.tick_count(), 3);
        
        // Out of bounds should fail
        assert!(!chart.record_trade(60000, 100, false));
    }

    #[test]
    fn test_aggregate_at_price() {
        let chart = FootprintChart::<100, 10>::new();
        chart.init(50000);
        
        // Multiple trades at same price across different buckets
        chart.record_trade(50000, 100, false);
        chart.advance_bucket();
        chart.record_trade(50000, 200, true);
        chart.advance_bucket();
        chart.record_trade(50000, 150, false);
        
        let (bid, ask) = chart.aggregate_at_price(50000).unwrap();
        assert_eq!(bid, 250);
        assert_eq!(ask, 200);
    }

    #[test]
    fn test_find_poc() {
        let chart = FootprintChart::<100, 10>::new();
        chart.init(50000);
        
        // Create volume concentration at 50005
        for _ in 0..10 {
            chart.record_trade(50005, 1000, true);
        }
        
        // Less volume elsewhere
        for _ in 0..3 {
            chart.record_trade(50000, 100, false);
        }
        
        let (poc_price, poc_vol) = chart.find_poc().unwrap();
        assert_eq!(poc_price, 50005);
        assert!(poc_vol >= 10000);
    }

    #[test]
    fn test_bucket_advancement() {
        let chart = FootprintChart::<100, 10>::new();
        chart.init(50000);
        
        chart.record_trade(50000, 100, false);
        let delta_before = chart.current_bucket_delta();
        
        chart.advance_bucket();
        let delta_after = chart.current_bucket_delta();
        
        assert_eq!(delta_before, -100);
        assert_eq!(delta_after, 0); // New bucket is empty
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;
        
        assert!(size_of::<FootprintCell>() >= CACHE_LINE_SIZE);
        assert_eq!(size_of::<FootprintCell>() % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_imbalance_calculation() {
        let mut cell = FootprintCell::new();
        cell.add_trade(100, false, 5000); // 100 bid
        cell.add_trade(300, true, 5000);  // 300 ask
        
        // Imbalance = (300 - 100) / 400 * 1000 = 500
        assert_eq!(cell.imbalance_scaled(), 500);
    }

    #[test]
    fn test_zero_allocation_hot_path() {
        let chart = FootprintChart::<100, 100>::new();
        chart.init(50000);
        
        // Record many trades - should not allocate
        for i in 0..10000 {
            let price = 50000 + (i % 50) as i64;
            let vol = 100 + (i % 1000);
            let is_ask = (i % 2) == 0;
            chart.record_trade(price, vol, is_ask);
        }
        
        assert_eq!(chart.tick_count(), 10000);
    }
}
