//! Equal highs/lows identification and liquidity sweep detection
//! Uses lock-free queues for thread-safe liquidity pool tracking.
//! Zero-copy data structures with cache-line alignment.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::smc::structure_engine::FixedPrice;

const SCALE: i64 = 100_000_000;

/// Maximum number of liquidity pools tracked (pre-allocated)
pub const MAX_LIQUIDITY_POOLS: usize = 256;

/// Cache-line padded liquidity pool entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiquidityPool {
    /// Price level of the liquidity pool
    pub price: FixedPrice,
    /// Number of touches (equal highs/lows count)
    pub touches: u32,
    /// Total volume at this level (fixed-point)
    pub volume: i64,
    /// Timestamp of last touch (nanoseconds)
    pub last_touch_ns: u64,
    /// Is this a high (true) or low (false) pool
    pub is_high: bool,
    /// Sweep detected flag
    pub swept: bool,
    _padding: [u8; 38], // Pad to 64 bytes
}

impl Default for LiquidityPool {
    fn default() -> Self {
        Self {
            price: 0,
            touches: 0,
            volume: 0,
            last_touch_ns: 0,
            is_high: false,
            swept: false,
            _padding: [0; 38],
        }
    }
}

/// Lock-free liquidity pool tracker
#[repr(C)]
pub struct LiquidityPools {
    /// Pre-allocated pool array (no heap allocation after init)
    pub pools: [LiquidityPool; MAX_LIQUIDITY_POOLS],
    /// Head index for circular buffer
    pub head: AtomicU64,
    /// Tail index for circular buffer
    pub tail: AtomicU64,
    /// Count of active pools
    pub count: AtomicU64,
    /// Sweep detected atomic flag
    pub sweep_detected: AtomicBool,
    /// Price tolerance for equal highs/lows (in basis points)
    pub tolerance_bps: u32,
    _padding: [u8; 44], // Align to cache line
}

impl LiquidityPools {
    pub fn new(tolerance_bps: u32) -> Self {
        Self {
            pools: [LiquidityPool::default(); MAX_LIQUIDITY_POOLS],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            count: AtomicU64::new(0),
            sweep_detected: AtomicBool::new(false),
            tolerance_bps,
            _padding: [0; 44],
        }
    }

    /// Process a new high/low and update liquidity pools
    /// Returns true if a sweep was detected
    #[inline]
    pub fn process_tick(&mut self, high: FixedPrice, low: FixedPrice, volume: i64, timestamp_ns: u64) -> bool {
        let mut sweep_found = false;

        // Check existing pools for sweeps
        let count = self.count.load(Ordering::Acquire);
        for i in 0..count.min(MAX_LIQUIDITY_POOLS as u64) as usize {
            let pool = &mut self.pools[i];
            
            if pool.is_high {
                // Check if high swept this liquidity pool
                if high > pool.price && !pool.swept {
                    let threshold = self.calculate_sweep_threshold(pool.price, true);
                    if high >= threshold {
                        pool.swept = true;
                        sweep_found = true;
                    }
                }
                // Add touch if near the level
                if self.is_near_level(high, pool.price) {
                    pool.touches = pool.touches.wrapping_add(1);
                    pool.volume = pool.volume.saturating_add(volume);
                    pool.last_touch_ns = timestamp_ns;
                }
            } else {
                // Check if low swept this liquidity pool
                if low < pool.price && !pool.swept {
                    let threshold = self.calculate_sweep_threshold(pool.price, false);
                    if low <= threshold {
                        pool.swept = true;
                        sweep_found = true;
                    }
                }
                // Add touch if near the level
                if self.is_near_level(low, pool.price) {
                    pool.touches = pool.touches.wrapping_add(1);
                    pool.volume = pool.volume.saturating_add(volume);
                    pool.last_touch_ns = timestamp_ns;
                }
            }
        }

        // Add new potential liquidity pools (equal highs/lows candidates)
        if count < MAX_LIQUIDITY_POOLS as u64 {
            self.add_pool(high, volume, timestamp_ns, true);
            self.add_pool(low, volume, timestamp_ns, false);
        }

        if sweep_found {
            self.sweep_detected.store(true, Ordering::Release);
        }

        sweep_found
    }

    /// Branchless check if price is near a level within tolerance
    #[inline]
    fn is_near_level(&self, price: FixedPrice, level: FixedPrice) -> bool {
        let diff = (price - level).abs();
        let tolerance = (level * self.tolerance_bps as i64) / 10_000;
        ((diff <= tolerance) as u8) != 0
    }

    /// Calculate sweep threshold (price must exceed this to be considered a sweep)
    #[inline]
    fn calculate_sweep_threshold(&self, price: FixedPrice, is_high: bool) -> FixedPrice {
        let sweep_extension = (price * 50) / 10_000; // 0.5% extension
        if is_high {
            price + sweep_extension
        } else {
            price - sweep_extension
        }
    }

    /// Add a new liquidity pool (branchless insertion)
    #[inline]
    fn add_pool(&mut self, price: FixedPrice, volume: i64, timestamp_ns: u64, is_high: bool) {
        let idx = self.head.load(Ordering::Acquire) as usize % MAX_LIQUIDITY_POOLS;
        self.pools[idx] = LiquidityPool {
            price,
            touches: 1,
            volume,
            last_touch_ns: timestamp_ns,
            is_high,
            swept: false,
            _padding: [0; 38],
        };
        self.head.fetch_add(1, Ordering::AcqRel);
        self.count.fetch_add(1, Ordering::AcqRel);
    }

    /// Get count of equal highs (touches >= 2)
    #[inline]
    pub fn count_equal_highs(&self) -> u64 {
        let mut count = 0u64;
        let total = self.count.load(Ordering::Acquire);
        for i in 0..total.min(MAX_LIQUIDITY_POOLS as u64) as usize {
            if self.pools[i].is_high && self.pools[i].touches >= 2 {
                count = count.wrapping_add(1);
            }
        }
        count
    }

    /// Get count of equal lows (touches >= 2)
    #[inline]
    pub fn count_equal_lows(&self) -> u64 {
        let mut count = 0u64;
        let total = self.count.load(Ordering::Acquire);
        for i in 0..total.min(MAX_LIQUIDITY_POOLS as u64) as usize {
            if !self.pools[i].is_high && self.pools[i].touches >= 2 {
                count = count.wrapping_add(1);
            }
        }
        count
    }

    /// Check if sweep was detected (thread-safe)
    #[inline]
    pub fn check_sweep(&self) -> bool {
        let detected = self.sweep_detected.load(Ordering::Acquire);
        if detected {
            self.sweep_detected.store(false, Ordering::Release);
        }
        detected
    }

    /// Get strongest liquidity pool (most touches)
    #[inline]
    pub fn get_strongest_pool(&self, is_high: bool) -> Option<&LiquidityPool> {
        let mut strongest: Option<&LiquidityPool> = None;
        let total = self.count.load(Ordering::Acquire);
        
        for i in 0..total.min(MAX_LIQUIDITY_POOLS as u64) as usize {
            let pool = &self.pools[i];
            if pool.is_high == is_high && pool.touches > 1 {
                if let Some(current) = strongest {
                    if pool.touches > current.touches {
                        strongest = Some(pool);
                    }
                } else {
                    strongest = Some(pool);
                }
            }
        }
        strongest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_pools_initialization() {
        let pools = LiquidityPools::new(10);
        assert_eq!(pools.count.load(Ordering::Relaxed), 0);
        assert!(!pools.check_sweep());
    }

    #[test]
    fn test_equal_highs_detection() {
        let mut pools = LiquidityPools::new(10);
        let price = 100 * SCALE;
        let volume = 1000;

        // Process same high multiple times
        for i in 0..3 {
            pools.process_tick(price, 99 * SCALE, volume, 1000 + i);
        }

        assert!(pools.count_equal_highs() >= 1);
    }

    #[test]
    fn test_sweep_detection() {
        let mut pools = LiquidityPools::new(10);
        let price = 100 * SCALE;

        // Create liquidity pool
        pools.process_tick(price, 99 * SCALE, 1000, 1000);
        
        // Sweep above the high
        pools.process_tick(price + (SCALE / 2), 99 * SCALE, 1000, 2000);

        assert!(pools.check_sweep());
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<LiquidityPool>() >= 64);
        assert!(core::mem::align_of::<LiquidityPool>() >= 8);
    }
}
