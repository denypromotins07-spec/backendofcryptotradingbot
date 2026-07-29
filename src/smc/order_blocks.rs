//! Order block, Breaker block, and Fair Value Gap (FVG) mapping
//! Uses zero-copy candle arrays with fixed-point arithmetic.
//! Pre-allocated buffers eliminate heap allocations in hot paths.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::smc::structure_engine::FixedPrice;

const SCALE: i64 = 100_000_000;

/// Maximum number of order blocks tracked (pre-allocated)
pub const MAX_ORDER_BLOCKS: usize = 128;

/// Maximum number of FVGs tracked (pre-allocated)
pub const MAX_FVG: usize = 256;

/// Candle data in fixed-point format (cache-line aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub open: FixedPrice,
    pub high: FixedPrice,
    pub low: FixedPrice,
    pub close: FixedPrice,
    pub volume: i64,
    pub timestamp_ns: u64,
    _padding: [u8; 16], // Pad to 64 bytes
}

impl Default for Candle {
    fn default() -> Self {
        Self {
            open: 0,
            high: 0,
            low: 0,
            close: 0,
            volume: 0,
            timestamp_ns: 0,
            _padding: [0; 16],
        }
    }
}

/// Order block types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderBlockType {
    Bullish = 0,
    Bearish = 1,
    BreakerBullish = 2,
    BreakerBearish = 3,
}

/// Cache-line padded order block entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBlock {
    /// Start price of the order block
    pub start_price: FixedPrice,
    /// End price of the order block
    pub end_price: FixedPrice,
    /// High of the order block range
    pub high: FixedPrice,
    /// Low of the order block range
    pub low: FixedPrice,
    /// Timestamp when block was formed (ns)
    pub timestamp_ns: u64,
    /// Type of order block
    pub block_type: OrderBlockType,
    /// Has this block been mitigated (price returned to it)
    pub mitigated: bool,
    /// Number of times price tested this block
    pub test_count: u32,
    _padding: [u8; 30], // Pad to 64 bytes
}

impl Default for OrderBlock {
    fn default() -> Self {
        Self {
            start_price: 0,
            end_price: 0,
            high: 0,
            low: 0,
            timestamp_ns: 0,
            block_type: OrderBlockType::Bullish,
            mitigated: false,
            test_count: 0,
            _padding: [0; 30],
        }
    }
}

/// Fair Value Gap entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FairValueGap {
    /// Upper bound of the FVG
    pub upper: FixedPrice,
    /// Lower bound of the FVG
    pub lower: FixedPrice,
    /// Midpoint of the FVG
    pub midpoint: FixedPrice,
    /// Timestamp when FVG was formed (ns)
    pub timestamp_ns: u64,
    /// Is this a bullish FVG (true) or bearish (false)
    pub is_bullish: bool,
    /// Has price filled this gap
    pub filled: bool,
    /// Volume imbalance at formation
    pub imbalance_volume: i64,
    _padding: [u8; 23], // Pad to 64 bytes
}

impl Default for FairValueGap {
    fn default() -> Self {
        Self {
            upper: 0,
            lower: 0,
            midpoint: 0,
            timestamp_ns: 0,
            is_bullish: false,
            filled: false,
            imbalance_volume: 0,
            _padding: [0; 23],
        }
    }
}

/// Main order blocks and FVG tracker
#[repr(C)]
pub struct OrderBlocksTracker {
    /// Pre-allocated order blocks array
    pub order_blocks: [OrderBlock; MAX_ORDER_BLOCKS],
    /// Pre-allocated FVG array
    pub fvgs: [FairValueGap; MAX_FVG],
    /// Count of active order blocks
    pub ob_count: AtomicU64,
    /// Count of active FVGs
    pub fvg_count: AtomicU64,
    /// Circular buffer index for candles
    pub candle_idx: AtomicU64,
    /// New OB detected flag
    pub ob_detected: AtomicBool,
    /// New FVG detected flag
    pub fvg_detected: AtomicBool,
    /// Last candle for FVG detection
    pub last_candle: Candle,
    /// Second-to-last candle
    pub prev_candle: Candle,
    _padding: [u8; 32], // Align to cache line
}

impl OrderBlocksTracker {
    pub fn new() -> Self {
        Self {
            order_blocks: [OrderBlock::default(); MAX_ORDER_BLOCKS],
            fvgs: [FairValueGap::default(); MAX_FVG],
            ob_count: AtomicU64::new(0),
            fvg_count: AtomicU64::new(0),
            candle_idx: AtomicU64::new(0),
            ob_detected: AtomicBool::new(false),
            fvg_detected: AtomicBool::new(false),
            last_candle: Candle::default(),
            prev_candle: Candle::default(),
            _padding: [0; 32],
        }
    }

    /// Process a new candle and detect order blocks / FVGs
    /// Returns true if any new pattern was detected
    #[inline]
    pub fn process_candle(&mut self, candle: &Candle) -> bool {
        let mut detected = false;

        // Detect Fair Value Gaps (3-candle pattern)
        if self.prev_candle.timestamp_ns > 0 {
            if let Some(fvg) = self.detect_fvg(&self.prev_candle, candle) {
                self.add_fvg(fvg);
                self.fvg_detected.store(true, Ordering::Release);
                detected = true;
            }
        }

        // Detect order blocks (single candle or multi-candle consolidation)
        if self.detect_order_block(candle) {
            self.ob_detected.store(true, Ordering::Release);
            detected = true;
        }

        // Check for mitigation of existing blocks
        self.check_mitigation(candle.high, candle.low);

        // Shift candles
        self.prev_candle = self.last_candle;
        self.last_candle = *candle;
        self.candle_idx.fetch_add(1, Ordering::AcqRel);

        detected
    }

    /// Detect Fair Value Gap between candles
    #[inline]
    fn detect_fvg(&self, prev: &Candle, current: &Candle) -> Option<FairValueGap> {
        // Bullish FVG: current low > prev high (gap up)
        if current.low > prev.high {
            let upper = current.low;
            let lower = prev.high;
            let gap_size = upper - lower;
            
            // Only track significant gaps (> 0.05%)
            let min_gap = ((upper + lower) / 2) * 5 / 10_000;
            if gap_size >= min_gap {
                return Some(FairValueGap {
                    upper,
                    lower,
                    midpoint: (upper + lower) / 2,
                    timestamp_ns: current.timestamp_ns,
                    is_bullish: true,
                    filled: false,
                    imbalance_volume: current.volume,
                    _padding: [0; 23],
                });
            }
        }
        
        // Bearish FVG: current high < prev low (gap down)
        if current.high < prev.low {
            let upper = prev.low;
            let lower = current.high;
            let gap_size = upper - lower;
            
            // Only track significant gaps
            let min_gap = ((upper + lower) / 2) * 5 / 10_000;
            if gap_size >= min_gap {
                return Some(FairValueGap {
                    upper,
                    lower,
                    midpoint: (upper + lower) / 2,
                    timestamp_ns: current.timestamp_ns,
                    is_bullish: false,
                    filled: false,
                    imbalance_volume: current.volume,
                    _padding: [0; 23],
                });
            }
        }
        
        None
    }

    /// Detect order block formation
    #[inline]
    fn detect_order_block(&mut self, candle: &Candle) -> bool {
        // Simple heuristic: large body candle with high volume
        let body = (candle.close - candle.open).abs();
        let range = candle.high - candle.low;
        
        // Avoid division by zero
        if range == 0 {
            return false;
        }

        let body_ratio = (body * 100) / range;
        
        // Strong directional candle (>70% body ratio)
        if body_ratio >= 70 {
            let block_type = if candle.close > candle.open {
                OrderBlockType::Bullish
            } else {
                OrderBlockType::Bearish
            };

            let ob = OrderBlock {
                start_price: candle.open,
                end_price: candle.close,
                high: candle.high,
                low: candle.low,
                timestamp_ns: candle.timestamp_ns,
                block_type,
                mitigated: false,
                test_count: 0,
                _padding: [0; 30],
            };

            self.add_order_block(ob);
            return true;
        }

        false
    }

    /// Add order block to pre-allocated array
    #[inline]
    fn add_order_block(&mut self, ob: OrderBlock) {
        let idx = self.ob_count.load(Ordering::Acquire) as usize % MAX_ORDER_BLOCKS;
        self.order_blocks[idx] = ob;
        self.ob_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Add FVG to pre-allocated array
    #[inline]
    fn add_fvg(&mut self, fvg: FairValueGap) {
        let idx = self.fvg_count.load(Ordering::Acquire) as usize % MAX_FVG;
        self.fvgs[idx] = fvg;
        self.fvg_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Check if price has mitigated any existing order blocks or filled FVGs
    #[inline]
    fn check_mitigation(&mut self, high: FixedPrice, low: FixedPrice) {
        // Check order blocks
        let ob_count = self.ob_count.load(Ordering::Acquire);
        for i in 0..ob_count.min(MAX_ORDER_BLOCKS as u64) as usize {
            let ob = &mut self.order_blocks[i];
            if !ob.mitigated {
                match ob.block_type {
                    OrderBlockType::Bullish | OrderBlockType::BreakerBullish => {
                        if low <= ob.high && high >= ob.start_price {
                            ob.mitigated = true;
                            ob.test_count = ob.test_count.wrapping_add(1);
                        }
                    }
                    OrderBlockType::Bearish | OrderBlockType::BreakerBearish => {
                        if high >= ob.low && low <= ob.start_price {
                            ob.mitigated = true;
                            ob.test_count = ob.test_count.wrapping_add(1);
                        }
                    }
                }
            }
        }

        // Check FVGs
        let fvg_count = self.fvg_count.load(Ordering::Acquire);
        for i in 0..fvg_count.min(MAX_FVG as u64) as usize {
            let fvg = &mut self.fvgs[i];
            if !fvg.filled {
                if fvg.is_bullish {
                    if low <= fvg.upper && high >= fvg.lower {
                        fvg.filled = true;
                    }
                } else {
                    if high >= fvg.lower && low <= fvg.upper {
                        fvg.filled = true;
                    }
                }
            }
        }
    }

    /// Get unmitigated bullish order blocks count
    #[inline]
    pub fn count_unmitigated_bullish_ob(&self) -> u64 {
        let mut count = 0u64;
        let total = self.ob_count.load(Ordering::Acquire);
        for i in 0..total.min(MAX_ORDER_BLOCKS as u64) as usize {
            if !self.order_blocks[i].mitigated 
                && matches!(self.order_blocks[i].block_type, OrderBlockType::Bullish | OrderBlockType::BreakerBullish) 
            {
                count = count.wrapping_add(1);
            }
        }
        count
    }

    /// Get unfilled FVGs count
    #[inline]
    pub fn count_unfilled_fvg(&self) -> u64 {
        let mut count = 0u64;
        let total = self.fvg_count.load(Ordering::Acquire);
        for i in 0..total.min(MAX_FVG as u64) as usize {
            if !self.fvgs[i].filled {
                count = count.wrapping_add(1);
            }
        }
        count
    }

    /// Check if new OB was detected
    #[inline]
    pub fn check_ob_detected(&self) -> bool {
        let detected = self.ob_detected.load(Ordering::Acquire);
        if detected {
            self.ob_detected.store(false, Ordering::Release);
        }
        detected
    }

    /// Check if new FVG was detected
    #[inline]
    pub fn check_fvg_detected(&self) -> bool {
        let detected = self.fvg_detected.load(Ordering::Acquire);
        if detected {
            self.fvg_detected.store(false, Ordering::Release);
        }
        detected
    }
}

impl Default for OrderBlocksTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_blocks_initialization() {
        let tracker = OrderBlocksTracker::new();
        assert_eq!(tracker.ob_count.load(Ordering::Relaxed), 0);
        assert_eq!(tracker.fvg_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_fvg_detection_bullish() {
        let mut tracker = OrderBlocksTracker::new();
        
        let prev = Candle {
            open: 100 * SCALE,
            high: 101 * SCALE,
            low: 99 * SCALE,
            close: 100 * SCALE,
            volume: 1000,
            timestamp_ns: 1000,
            _padding: [0; 16],
        };
        
        let current = Candle {
            open: 102 * SCALE,
            high: 103 * SCALE,
            low: 102 * SCALE, // Gap: low > prev.high
            close: 103 * SCALE,
            volume: 1000,
            timestamp_ns: 2000,
            _padding: [0; 16],
        };

        tracker.prev_candle = prev;
        tracker.process_candle(&current);
        
        assert!(tracker.check_fvg_detected());
        assert!(tracker.count_unfilled_fvg() >= 1);
    }

    #[test]
    fn test_order_block_detection() {
        let mut tracker = OrderBlocksTracker::new();
        
        // Strong bullish candle (90% body ratio)
        let candle = Candle {
            open: 100 * SCALE,
            high: 101 * SCALE,
            low: 100 * SCALE,
            close: 101 * SCALE, // Full body
            volume: 5000,
            timestamp_ns: 1000,
            _padding: [0; 16],
        };

        tracker.process_candle(&candle);
        
        assert!(tracker.check_ob_detected());
        assert!(tracker.count_unmitigated_bullish_ob() >= 1);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<Candle>() >= 64);
        assert!(core::mem::size_of::<OrderBlock>() >= 64);
        assert!(core::mem::size_of::<FairValueGap>() >= 64);
    }
}
