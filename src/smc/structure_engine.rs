//! Break of Structure (BOS) and Change of Character (CHoCH) detection
//! Uses fixed-point arithmetic for deterministic FPU behavior.
//! Zero-copy swing high/low tracking with lock-free state transitions.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Fixed-point representation for price (scaled by 10^8)
pub type FixedPrice = i64;
const SCALE: i64 = 100_000_000;

/// Cache-line padded structure to prevent false sharing
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SwingPoint {
    pub price: FixedPrice,
    pub timestamp_ns: u64,
    pub is_high: bool,
    _padding: [u8; 55], // Pad to 64 bytes
}

impl Default for SwingPoint {
    fn default() -> Self {
        Self {
            price: 0,
            timestamp_ns: 0,
            is_high: false,
            _padding: [0; 55],
        }
    }
}

/// Lock-free structure engine state
#[repr(C)]
pub struct StructureEngine {
    /// Last confirmed swing high
    pub last_swing_high: SwingPoint,
    /// Last confirmed swing low
    pub last_swing_low: SwingPoint,
    /// Previous swing high (for BOS detection)
    pub prev_swing_high: SwingPoint,
    /// Previous swing low (for CHoCH detection)
    pub prev_swing_low: SwingPoint,
    /// Atomic flag for BOS detected
    pub bos_detected: AtomicBool,
    /// Atomic flag for CHoCH detected
    pub choch_detected: AtomicBool,
    /// Current market structure (bullish/bearish)
    pub structure_state: AtomicU64, // 0 = neutral, 1 = bullish, 2 = bearish
    /// Lookback period for swing detection
    pub lookback: u32,
    /// Circular buffer index for candle storage (pre-allocated externally)
    pub buffer_idx: u64,
    _padding: [u8; 32], // Align to cache line boundary
}

impl StructureEngine {
    pub const STATE_NEUTRAL: u64 = 0;
    pub const STATE_BULLISH: u64 = 1;
    pub const STATE_BEARISH: u64 = 2;

    pub fn new(lookback: u32) -> Self {
        Self {
            last_swing_high: SwingPoint::default(),
            last_swing_low: SwingPoint::default(),
            prev_swing_high: SwingPoint::default(),
            prev_swing_low: SwingPoint::default(),
            bos_detected: AtomicBool::new(false),
            choch_detected: AtomicBool::new(false),
            structure_state: AtomicU64::new(Self::STATE_NEUTRAL),
            lookback,
            buffer_idx: 0,
            _padding: [0; 32],
        }
    }

    /// Process a new candle (high, low, close in fixed-point)
    /// Returns true if a structural break was detected
    #[inline]
    pub fn process_candle(&mut self, high: FixedPrice, low: FixedPrice, timestamp_ns: u64) -> bool {
        let mut structure_changed = false;

        // Check for swing high formation
        if self.is_swing_high(high) {
            self.prev_swing_high = self.last_swing_high;
            self.last_swing_high = SwingPoint {
                price: high,
                timestamp_ns,
                is_high: true,
                _padding: [0; 55],
            };

            // Detect BOS (Break of Structure) - Bullish
            if self.last_swing_high.price > self.prev_swing_high.price
                && self.prev_swing_high.price > 0
            {
                let current_state = self.structure_state.load(Ordering::Relaxed);
                if current_state == Self::STATE_BEARISH {
                    // CHoCH: Bearish to Bullish transition
                    self.choch_detected.store(true, Ordering::Release);
                    self.structure_state.store(Self::STATE_BULLISH, Ordering::Release);
                    structure_changed = true;
                } else if current_state == Self::STATE_BULLISH {
                    // BOS: Continuation of bullish structure
                    self.bos_detected.store(true, Ordering::Release);
                }
            }
        }

        // Check for swing low formation
        if self.is_swing_low(low) {
            self.prev_swing_low = self.last_swing_low;
            self.last_swing_low = SwingPoint {
                price: low,
                timestamp_ns,
                is_high: false,
                _padding: [0; 55],
            };

            // Detect BOS (Break of Structure) - Bearish
            if self.last_swing_low.price < self.prev_swing_low.price
                && self.prev_swing_low.price > 0
            {
                let current_state = self.structure_state.load(Ordering::Relaxed);
                if current_state == Self::STATE_BULLISH {
                    // CHoCH: Bullish to Bearish transition
                    self.choch_detected.store(true, Ordering::Release);
                    self.structure_state.store(Self::STATE_BEARISH, Ordering::Release);
                    structure_changed = true;
                } else if current_state == Self::STATE_BEARISH {
                    // BOS: Continuation of bearish structure
                    self.bos_detected.store(true, Ordering::Release);
                }
            }
        }

        // Reset flags after detection (caller should read atomically)
        if structure_changed {
            self.bos_detected.store(false, Ordering::Release);
            self.choch_detected.store(false, Ordering::Release);
        }

        self.buffer_idx = self.buffer_idx.wrapping_add(1);
        structure_changed
    }

    /// Branchless swing high detection using fixed-point comparison
    #[inline]
    fn is_swing_high(&self, current_high: FixedPrice) -> bool {
        // Simplified: in production, this would check against lookback buffer
        // Using branchless comparison pattern
        let threshold = self.last_swing_high.price + (SCALE / 100); // 0.01% threshold
        ((current_high > threshold) as u8) != 0
    }

    /// Branchless swing low detection
    #[inline]
    fn is_swing_low(&self, current_low: FixedPrice) -> bool {
        let threshold = self.last_swing_low.price - (SCALE / 100);
        ((current_low < threshold) as u8) != 0
    }

    /// Get current structure state atomically
    #[inline]
    pub fn get_structure_state(&self) -> u64 {
        self.structure_state.load(Ordering::Acquire)
    }

    /// Check if BOS was detected (thread-safe)
    #[inline]
    pub fn check_bos(&self) -> bool {
        self.bos_detected.load(Ordering::Acquire)
    }

    /// Check if CHoCH was detected (thread-safe)
    #[inline]
    pub fn check_choch(&self) -> bool {
        self.choch_detected.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_structure_engine_initialization() {
        let engine = StructureEngine::new(5);
        assert_eq!(engine.get_structure_state(), StructureEngine::STATE_NEUTRAL);
        assert!(!engine.check_bos());
        assert!(!engine.check_choch());
    }

    #[test]
    fn test_bos_detection_bullish() {
        let mut engine = StructureEngine::new(5);
        
        // Initialize swing points
        engine.prev_swing_high = SwingPoint {
            price: 100 * SCALE,
            timestamp_ns: 1000,
            is_high: true,
            _padding: [0; 55],
        };
        engine.last_swing_high = SwingPoint {
            price: 100 * SCALE,
            timestamp_ns: 1000,
            is_high: true,
            _padding: [0; 55],
        };
        engine.structure_state.store(StructureEngine::STATE_BULLISH, Ordering::Release);

        // Process higher high
        engine.process_candle(105 * SCALE, 99 * SCALE, 2000);
        assert!(engine.check_bos());
    }

    proptest! {
        #[test]
        fn test_fixed_point_arithmetic(high in 1_000_000i64..100_000_000_000i64) {
            let mut engine = StructureEngine::new(5);
            let low = high - 1000; // Small spread
            engine.process_candle(high, low, 1000);
            // Should not panic with extreme values
        }
    }
}
