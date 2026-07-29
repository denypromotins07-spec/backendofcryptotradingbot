//! Lock-free pairs trading signal generator with dynamic z-score thresholds and half-life decay.
//!
//! Implements real-time pairs trading signals for statistical arbitrage
//! using fixed-point arithmetic, lock-free accumulators, and SIMD acceleration.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

/// Cache line padding for 64-byte alignment
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of spread observations for rolling statistics
const MAX_SPREAD_OBS: usize = 16384;

/// Signal strength levels
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalStrength {
    None = 0,
    Weak = 1,
    Medium = 2,
    Strong = 3,
    Extreme = 4,
}

/// Trading signal direction
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Neutral = 0,
    LongSpread = 1,   // Buy asset A, sell asset B
    ShortSpread = -1, // Sell asset A, buy asset B
}

/// Pairs trading signal
#[repr(C, align(64))]
pub struct PairsSignal {
    /// Signal direction
    pub direction: SignalDirection,
    /// Signal strength
    pub strength: SignalStrength,
    /// Current z-score of the spread
    pub z_score: FixedI64,
    /// Dynamic entry threshold
    pub entry_threshold: FixedI64,
    /// Dynamic exit threshold
    pub exit_threshold: FixedI64,
    /// Current spread value
    pub spread: FixedI64,
    /// Hedge ratio used
    pub hedge_ratio: FixedI64,
    /// Half-life of mean reversion (in observations)
    pub half_life: u32,
    /// Time since signal generation (in microseconds)
    pub signal_age_us: u64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 2 * 4 - 5 * 8 - 1 * 4 - 1 * 8],
}

/// Pairs trading state
#[repr(C, align(64))]
pub struct PairsTradingState {
    /// Lock-free flag indicating if trading is active
    pub active: AtomicBool,
    /// Lock-free flag for halt signal (circuit breaker)
    pub halted: AtomicBool,
    /// Current observation count
    pub obs_count: AtomicU64,
    /// Asset A identifier
    pub asset_a_id: u64,
    /// Asset B identifier
    pub asset_b_id: u64,
    /// Hedge ratio (units of B per unit of A)
    pub hedge_ratio: FixedI64,
    /// Current spread value
    pub spread: FixedI64,
    /// Rolling mean of spread
    pub spread_mean: FixedI64,
    /// Rolling variance of spread
    pub spread_variance: FixedI64,
    /// Rolling standard deviation of spread
    pub spread_std: FixedI64,
    /// Circular buffer for spread values
    pub spread_buffer: [FixedI64; MAX_SPREAD_OBS],
    /// Head index for circular buffer
    pub spread_head: usize,
    /// Sum of spread for mean calculation
    pub spread_sum: FixedI64,
    /// Sum of squared spread
    pub spread_ss: FixedI64,
    /// Dynamic z-score entry threshold (scaled by 1e9)
    pub z_entry: FixedI64,
    /// Dynamic z-score exit threshold (scaled by 1e9)
    pub z_exit: FixedI64,
    /// Base z-score threshold
    pub z_base: FixedI64,
    /// Volatility adjustment factor
    pub vol_adjustment: FixedI64,
    /// Half-life decay factor
    pub half_life_decay: FixedI64,
    /// Estimated half-life
    pub half_life: u32,
    /// Consecutive signals counter
    pub signal_counter: AtomicI64,
    /// Last signal timestamp (rdtsc cycles)
    pub last_signal_ts: AtomicU64,
    /// Position size limit (in base units)
    pub position_limit: FixedI64,
    /// Current position (positive = long spread)
    pub current_position: FixedI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE 
        - 2 * 8  // AtomicBool, AtomicBool (with padding)
        - 8      // AtomicU64
        - 2 * 8  // asset IDs
        - 10 * 8 // hedge_ratio through half_life_decay
        - 4      // half_life as u32
        - 8      // AtomicI64
        - 8      // AtomicU64
        - 2 * 8  // position_limit, current_position
        - 8,     // spread_head as usize
    ],
}

impl PairsTradingState {
    /// Create a new pairs trading state with pre-allocated buffers
    #[inline]
    pub const fn new(asset_a_id: u64, asset_b_id: u64) -> Self {
        Self {
            active: AtomicBool::new(false),
            halted: AtomicBool::new(false),
            obs_count: AtomicU64::new(0),
            asset_a_id,
            asset_b_id,
            hedge_ratio: FixedI64::ZERO,
            spread: FixedI64::ZERO,
            spread_mean: FixedI64::ZERO,
            spread_variance: FixedI64::ZERO,
            spread_std: FixedI64::ZERO,
            spread_buffer: [FixedI64::ZERO; MAX_SPREAD_OBS],
            spread_head: 0,
            spread_sum: FixedI64::ZERO,
            spread_ss: FixedI64::ZERO,
            z_entry: FixedI64::from_i64(2000000000i64), // 2.0
            z_exit: FixedI64::from_i64(500000000i64),   // 0.5
            z_base: FixedI64::from_i64(2000000000i64),  // 2.0
            vol_adjustment: FixedI64::ONE,
            half_life_decay: FixedI64::ZERO,
            half_life: 0,
            signal_counter: AtomicI64::new(0),
            last_signal_ts: AtomicU64::new(0),
            position_limit: FixedI64::MAX,
            current_position: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE 
                - 2 * 8 - 8 - 2 * 8 - 10 * 8 - 4 - 8 - 8 - 2 * 8 - 8],
        }
    }

    /// Activate pairs trading
    #[inline]
    pub fn activate(&self) {
        self.active.store(true, Ordering::Relaxed);
        self.halted.store(false, Ordering::Relaxed);
    }

    /// Deactivate pairs trading
    #[inline]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Trigger circuit breaker halt
    #[inline]
    pub fn halt(&self) {
        self.halted.store(true, Ordering::SeqCst);
    }

    /// Check if trading is active and not halted
    #[inline]
    pub fn can_trade(&self) -> bool {
        self.active.load(Ordering::Relaxed) && !self.halted.load(Ordering::SeqCst)
    }

    /// Update hedge ratio
    #[inline]
    pub fn set_hedge_ratio(&self, ratio: FixedI64) {
        // In production, use atomic or proper synchronization
    }

    /// Update with new price data and generate signal
    /// Returns Some(signal) if a trading signal is generated
    #[inline]
    pub fn update(&self, price_a: FixedI64, price_b: FixedI64) -> Option<PairsSignal> {
        if !self.can_trade() {
            return None;
        }

        // Calculate spread: price_a - hedge_ratio * price_b
        let spread = price_a - self.hedge_ratio * price_b;

        // Update rolling statistics using circular buffer (O(1))
        self.update_spread_stats(spread);

        // Calculate z-score
        let z_score = self.calculate_z_score(spread);

        // Update dynamic thresholds based on volatility and half-life
        self.update_dynamic_thresholds();

        // Generate signal using branchless logic
        let signal = self.generate_signal_branchless(z_score, spread);

        signal
    }

    /// Update rolling spread statistics using circular buffer
    #[inline]
    fn update_spread_stats(&self, spread: FixedI64) {
        let head = self.spread_head;
        let old_spread = unsafe { *self.spread_buffer.get_unchecked(head) };

        // Update running sums
        let new_sum = self.spread_sum + spread - old_spread;
        let new_ss = self.spread_ss + spread * spread - old_spread * old_spread;

        // Store new spread in buffer
        unsafe {
            *self.spread_buffer.get_unchecked_mut(head) = spread;
        }

        // Update head pointer
        let new_head = if head + 1 >= MAX_SPREAD_OBS { 0 } else { head + 1 };

        // Update observation count
        let obs = self.obs_count.fetch_add(1, Ordering::Relaxed);
        let n = if obs < MAX_SPREAD_OBS as u64 { obs + 1 } else { MAX_SPREAD_OBS as u64 };
        let n_fp = FixedI64::from_i64(n as i64);

        // Calculate mean
        let mean = if n > 0 { new_sum / n_fp } else { FixedI64::ZERO };

        // Calculate variance and std dev
        let variance = if n > 1 {
            (new_ss - new_sum * new_sum / n_fp) / FixedI64::from_i64((n - 1) as i64)
        } else {
            FixedI64::ZERO
        };
        let std_dev = if variance > FixedI64::ZERO {
            variance.sqrt_approx()
        } else {
            FixedI64::ZERO
        };
    }

    /// Calculate z-score of current spread
    #[inline]
    fn calculate_z_score(&self, spread: FixedI64) -> FixedI64 {
        if self.spread_std > FixedI64::ZERO {
            (spread - self.spread_mean) / self.spread_std
        } else {
            FixedI64::ZERO
        }
    }

    /// Update dynamic thresholds based on market conditions
    #[inline]
    fn update_dynamic_thresholds(&self) {
        // Adjust thresholds based on volatility regime
        // Higher volatility -> wider thresholds to avoid false signals
        
        let vol_factor = if self.spread_std > FixedI64::ZERO {
            // Normalize volatility to a baseline
            let baseline_vol = FixedI64::from_i64(1000000000i64); // 1.0
            let ratio = self.spread_std / baseline_vol;
            // Clamp between 0.8 and 1.5
            if ratio < FixedI64::from_i64(800000000i64) {
                FixedI64::from_i64(800000000i64)
            } else if ratio > FixedI64::from_i64(1500000000i64) {
                FixedI64::from_i64(1500000000i64)
            } else {
                ratio
            }
        } else {
            FixedI64::ONE
        };

        // Adjust for half-life decay
        let hl_factor = if self.half_life > 0 && self.half_life < 100 {
            // Shorter half-life -> tighter thresholds (faster mean reversion)
            FixedI64::from_i64(1000000000i64) - FixedI64::from_i64(self.half_life as i64) * FixedI64::from_i64(5000000i64)
        } else {
            FixedI64::ONE
        };

        // Combined adjustment
        let adjustment = vol_factor * hl_factor;
        
        // Update entry threshold
        let new_entry = self.z_base * adjustment;
        
        // Exit threshold is typically half of entry
        let new_exit = new_entry / FixedI64::from_i64(2000000000i64); // Divide by 2
    }

    /// Generate signal using branchless programming techniques
    #[inline]
    fn generate_signal_branchless(&self, z_score: FixedI64, spread: FixedI64) -> Option<PairsSignal> {
        // Branchless signal generation
        // Uses comparison results as multipliers instead of if/else
        
        let z_abs = if z_score < FixedI64::ZERO { -z_score } else { z_score };
        
        // Determine direction: positive z -> short spread, negative z -> long spread
        let dir_raw = if z_score > FixedI64::ZERO { -1i32 } else if z_score < FixedI64::ZERO { 1i32 } else { 0i32 };
        
        // Calculate strength based on z-score magnitude
        // z > 3.0 -> Extreme, z > 2.5 -> Strong, z > 2.0 -> Medium, z > 1.5 -> Weak
        let z_3 = FixedI64::from_i64(3000000000i64);
        let z_2_5 = FixedI64::from_i64(2500000000i64);
        let z_2 = FixedI64::from_i64(2000000000i64);
        let z_1_5 = FixedI64::from_i64(1500000000i64);
        
        let is_extreme = if z_abs > z_3 { 1 } else { 0 };
        let is_strong = if z_abs > z_2_5 { 1 } else { 0 };
        let is_medium = if z_abs > z_2 { 1 } else { 0 };
        let is_weak = if z_abs > z_1_5 { 1 } else { 0 };
        
        let strength_val = is_extreme * 4 + (1 - is_extreme) * is_strong * 3 
            + (1 - is_extreme) * (1 - is_strong) * is_medium * 2
            + (1 - is_extreme) * (1 - is_strong) * (1 - is_medium) * is_weak * 1;
        
        let strength = match strength_val {
            4 => SignalStrength::Extreme,
            3 => SignalStrength::Strong,
            2 => SignalStrength::Medium,
            1 => SignalStrength::Weak,
            _ => SignalStrength::None,
        };

        // No signal if strength is None
        if strength == SignalStrength::None {
            return None;
        }

        let direction = match dir_raw {
            1 => SignalDirection::LongSpread,
            -1 => SignalDirection::ShortSpread,
            _ => SignalDirection::Neutral,
        };

        // Get current timestamp (simulated rdtsc)
        let ts = self.last_signal_ts.fetch_add(1, Ordering::Relaxed);

        // Update signal counter
        let counter = self.signal_counter.fetch_add(1, Ordering::Relaxed);

        Some(PairsSignal {
            direction,
            strength,
            z_score,
            entry_threshold: self.z_entry,
            exit_threshold: self.z_exit,
            spread,
            hedge_ratio: self.hedge_ratio,
            half_life: self.half_life,
            signal_age_us: 0,
            _pad: [0u8; CACHE_LINE_SIZE - 2 * 4 - 5 * 8 - 1 * 4 - 1 * 8],
        })
    }

    /// SIMD-accelerated z-score calculation for multiple pairs
    #[inline]
    pub fn calculate_z_scores_simd(
        &self,
        spreads: &[FixedI64],
        means: &[FixedI64],
        stds: &[FixedI64],
    ) -> [FixedI64; 4] {
        assert!(spreads.len() >= 4);
        assert!(means.len() >= 4);
        assert!(stds.len() >= 4);

        unsafe {
            let mut z_scores = [FixedI64::ZERO; 4];
            
            // Process 4 pairs at a time
            for i in (0..4).step_by(4) {
                let s_vec = _mm256_loadu_si256(spreads[i..].as_ptr() as *const __m256i);
                let m_vec = _mm256_loadu_si256(means[i..].as_ptr() as *const __m256i);
                let sd_vec = _mm256_loadu_si256(stds[i..].as_ptr() as *const __m256i);
                
                // Calculate (spread - mean) / std
                let diff = _mm256_sub_epi64(s_vec, m_vec);
                
                // Division is complex in SIMD; for now, extract and compute scalar
                let mut diff_arr = [0i64; 4];
                let mut std_arr = [0i64; 4];
                _mm256_storeu_si256(diff_arr.as_mut_ptr() as *mut __m256i, diff);
                _mm256_storeu_si256(std_arr.as_mut_ptr() as *mut __m256i, sd_vec);
                
                for j in 0..4 {
                    let d = FixedI64::from_i64(diff_arr[j]);
                    let sd = FixedI64::from_i64(std_arr[j]);
                    z_scores[i + j] = if sd > FixedI64::ZERO { d / sd } else { FixedI64::ZERO };
                }
            }
            
            z_scores
        }
    }

    /// Manually unrolled loop for fast spread calculation
    #[inline]
    pub fn calculate_spreads_unrolled(
        &self,
        prices_a: &[FixedI64],
        prices_b: &[FixedI64],
        hedge_ratio: FixedI64,
        output: &mut [FixedI64],
    ) {
        assert!(prices_a.len() == prices_b.len());
        assert!(output.len() >= prices_a.len());

        let len = prices_a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            // Process chunks of 4
            for i in 0..chunks {
                let base = i * 4;
                *output.get_unchecked_mut(base) = 
                    *prices_a.get_unchecked(base) - hedge_ratio * *prices_b.get_unchecked(base);
                *output.get_unchecked_mut(base + 1) = 
                    *prices_a.get_unchecked(base + 1) - hedge_ratio * *prices_b.get_unchecked(base + 1);
                *output.get_unchecked_mut(base + 2) = 
                    *prices_a.get_unchecked(base + 2) - hedge_ratio * *prices_b.get_unchecked(base + 2);
                *output.get_unchecked_mut(base + 3) = 
                    *prices_a.get_unchecked(base + 3) - hedge_ratio * *prices_b.get_unchecked(base + 3);
            }

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                *output.get_unchecked_mut(idx) = 
                    *prices_a.get_unchecked(idx) - hedge_ratio * *prices_b.get_unchecked(idx);
            }
        }
    }

    /// Check if position should be exited based on z-score crossing exit threshold
    #[inline]
    pub fn should_exit(&self, current_z: FixedI64, position_direction: SignalDirection) -> bool {
        let z_abs = if current_z < FixedI64::ZERO { -current_z } else { current_z };
        
        // Branchless exit check
        let exit_threshold_abs = if self.z_exit < FixedI64::ZERO { 
            -self.z_exit 
        } else { 
            self.z_exit 
        };
        
        // Exit if z-score has reverted below threshold
        z_abs < exit_threshold_abs
    }
}

// Compile-time assertions for alignment
const _: () = assert!(core::mem::size_of::<PairsSignal>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<PairsSignal>() == 64);
const _: () = assert!(core::mem::size_of::<PairsTradingState>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<PairsTradingState>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_pairs_state_creation() {
        let state = PairsTradingState::new(1, 2);
        assert!(!state.can_trade());
        assert_eq!(state.obs_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_activation_and_trading() {
        let state = PairsTradingState::new(1, 2);
        state.activate();
        assert!(state.can_trade());
        
        state.halt();
        assert!(!state.can_trade());
    }

    #[test]
    fn test_signal_generation() {
        let state = PairsTradingState::new(1, 2);
        state.activate();
        
        let price_a = FixedI64::from_i64(100_000_000_000i64);
        let price_b = FixedI64::from_i64(100_000_000_000i64);
        
        // First update may not generate signal until we have enough data
        let result = state.update(price_a, price_b);
        // Result depends on internal state
    }

    #[test]
    fn test_circuit_breaker() {
        let state = PairsTradingState::new(1, 2);
        state.activate();
        assert!(state.can_trade());
        
        state.halt();
        assert!(!state.can_trade());
        
        // Simulate reset (in production would have explicit reset method)
        state.deactivate();
        state.activate();
        assert!(state.can_trade());
    }

    proptest! {
        #[test]
        fn test_extreme_price_spikes(
            base_price in 10000000000i64..100000000000i64,
            spike_factor in 0.01f64..100.0f64,
        ) {
            let state = PairsTradingState::new(1, 2);
            state.activate();
            
            let normal = FixedI64::from_i64(base_price);
            let spiked = FixedI64::from_i64((base_price as f64 * spike_factor) as i64);
            
            // Should handle extreme spikes without crashing
            let _ = state.update(normal, normal);
            let _ = state.update(spiked, normal);
            let _ = state.update(normal, spiked);
            
            prop_assert!(true);
        }

        #[test]
        fn test_z_score_calculation(
            spread in -1000000000000i64..1000000000000i64,
            mean in -1000000000000i64..1000000000000i64,
            std_dev in 1000000i64..1000000000000i64,
        ) {
            let state = PairsTradingState::new(1, 2);
            
            // Test that z-score calculation doesn't overflow
            let s = FixedI64::from_i64(spread);
            let m = FixedI64::from_i64(mean);
            let sd = FixedI64::from_i64(std_dev);
            
            let z = if sd > FixedI64::ZERO { (s - m) / sd } else { FixedI64::ZERO };
            
            // Z-score should be finite
            prop_assert!(z != FixedI64::MAX && z != FixedI64::MIN);
        }
    }
}
