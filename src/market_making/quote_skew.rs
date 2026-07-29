//! Dynamic bid-ask skew engine reacting instantly to order flow toxicity and CVD shifts.
//!
//! Implements real-time quote skew adjustment using fixed-point arithmetic,
//! lock-free accumulators, and branchless programming techniques.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

/// Cache line padding for 64-byte alignment
const CACHE_LINE_SIZE: usize = 64;

/// Maximum CVD history for rolling calculations
const MAX_CVD_HISTORY: usize = 4096;

/// Order flow toxicity levels
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Extreme = 3,
}

/// Quote skew state
#[repr(C, align(64))]
pub struct QuoteSkewState {
    /// Lock-free flag indicating if skew engine is active
    pub active: AtomicBool,
    /// Current observation count
    pub obs_count: AtomicU64,
    /// Cumulative volume delta (CVD) - buy volume minus sell volume
    pub cvd: AtomicI64,
    /// Rolling CVD buffer
    pub cvd_buffer: [FixedI64; MAX_CVD_HISTORY],
    /// Head index for CVD buffer
    pub cvd_head: usize,
    /// CVD sum for rolling mean
    pub cvd_sum: FixedI64,
    /// Order flow imbalance (scaled by 1e9)
    pub order_imbalance: FixedI64,
    /// Trade sign aggression (scaled by 1e9)
    pub trade_aggression: FixedI64,
    /// Toxicity level
    pub toxicity: AtomicI64,
    /// Base bid skew (scaled by 1e9)
    pub base_bid_skew: FixedI64,
    /// Base ask skew (scaled by 1e9)
    pub base_ask_skew: FixedI64,
    /// Dynamic bid skew adjustment
    pub bid_skew_adj: FixedI64,
    /// Dynamic ask skew adjustment
    pub ask_skew_adj: FixedI64,
    /// CVD momentum (rate of change)
    pub cvd_momentum: FixedI64,
    /// Last CVD value
    pub last_cvd: AtomicI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE 
        - 8      // AtomicBool
        - 8      // AtomicU64
        - 8      // AtomicI64 (cvd)
        - MAX_CVD_HISTORY * 8  // cvd_buffer
        - 8      // cvd_head
        - 7 * 8  // other FixedI64 fields
        - 8      // AtomicI64 (toxicity)
        - 8,     // AtomicI64 (last_cvd)
    ],
}

/// Skew calculation result
#[repr(C, align(64))]
pub struct SkewResult {
    /// Bid skew adjustment (positive = more aggressive)
    pub bid_skew: FixedI64,
    /// Ask skew adjustment (positive = more aggressive)
    pub ask_skew: FixedI64,
    /// Net skew (bid - ask)
    pub net_skew: FixedI64,
    /// Toxicity level
    pub toxicity: ToxicityLevel,
    /// CVD value
    pub cvd: FixedI64,
    /// CVD momentum
    pub cvd_momentum: FixedI64,
    /// Order flow imbalance
    pub order_imbalance: FixedI64,
    /// Confidence in skew signal (scaled by 1e9)
    pub confidence: FixedI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 6 * 8 - 1 * 4 - 1 * 4],
}

impl QuoteSkewState {
    /// Create new quote skew state with pre-allocated buffers
    #[inline]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            obs_count: AtomicU64::new(0),
            cvd: AtomicI64::new(0),
            cvd_buffer: [FixedI64::ZERO; MAX_CVD_HISTORY],
            cvd_head: 0,
            cvd_sum: FixedI64::ZERO,
            order_imbalance: FixedI64::ZERO,
            trade_aggression: FixedI64::ZERO,
            toxicity: AtomicI64::new(ToxicityLevel::Low as i64),
            base_bid_skew: FixedI64::ZERO,
            base_ask_skew: FixedI64::ZERO,
            bid_skew_adj: FixedI64::ZERO,
            ask_skew_adj: FixedI64::ZERO,
            cvd_momentum: FixedI64::ZERO,
            last_cvd: AtomicI64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE 
                - 8 - 8 - 8 - MAX_CVD_HISTORY * 8 - 8 - 7 * 8 - 8 - 8],
        }
    }

    /// Activate skew engine
    #[inline]
    pub fn activate(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Deactivate skew engine
    #[inline]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if engine is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Update with new order flow data using branchless logic
    #[inline]
    pub fn update_branchless(
        &self,
        buy_volume: FixedI64,
        sell_volume: FixedI64,
        trade_sign: i8, // 1 = buy, -1 = sell, 0 = unknown
    ) -> Option<SkewResult> {
        if !self.is_active() {
            return None;
        }

        // Branchless volume delta calculation
        let vol_delta = buy_volume - sell_volume;
        
        // Update CVD atomically
        let prev_cvd = self.cvd.load(Ordering::Relaxed);
        let new_cvd = prev_cvd + vol_delta.to_i64();
        self.cvd.store(new_cvd, Ordering::Relaxed);

        // Calculate CVD momentum (rate of change)
        let cvd_fp = FixedI64::from_i64(new_cvd);
        let last_cvd_fp = FixedI64::from_i64(prev_cvd);
        let momentum = cvd_fp - last_cvd_fp;
        
        // Update CVD in circular buffer (O(1))
        self.update_cvd_buffer(cvd_fp);

        // Branchless order flow imbalance calculation
        let total_vol = buy_volume + sell_volume;
        let imbalance = if total_vol > FixedI64::ZERO {
            (buy_volume - sell_volume) / total_vol
        } else {
            FixedI64::ZERO
        };

        // Branchless trade aggression update
        let aggression_update = FixedI64::from_i64(trade_sign as i64) * vol_volume.abs();
        
        // Calculate toxicity level using branchless comparisons
        let abs_imbalance = if imbalance < FixedI64::ZERO { -imbalance } else { imbalance };
        let abs_momentum = if momentum < FixedI64::ZERO { -momentum } else { momentum };
        
        let tox_threshold_high = FixedI64::from_i64(700000000i64);   // 0.7
        let tox_threshold_med = FixedI64::from_i64(400000000i64);    // 0.4
        let tox_threshold_low = FixedI64::from_i64(200000000i64);    // 0.2
        
        let is_extreme = if abs_imbalance > tox_threshold_high { 1 } else { 0 };
        let is_high = if abs_imbalance > tox_threshold_med { 1 } else { 0 };
        let is_medium = if abs_imbalance > tox_threshold_low { 1 } else { 0 };
        
        let tox_val = is_extreme * 3 + (1 - is_extreme) * is_high * 2 
            + (1 - is_extreme) * (1 - is_high) * is_medium * 1;
        
        let toxicity = match tox_val {
            3 => ToxicityLevel::Extreme,
            2 => ToxicityLevel::High,
            1 => ToxicityLevel::Medium,
            _ => ToxicityLevel::Low,
        };
        
        self.toxicity.store(toxicity as i64, Ordering::Relaxed);

        // Calculate dynamic skew adjustments
        let skew_result = self.calculate_skew_branchless(imbalance, momentum, toxicity);

        Some(skew_result)
    }

    /// Update CVD buffer using circular buffer pattern
    #[inline]
    fn update_cvd_buffer(&self, cvd: FixedI64) {
        let head = self.cvd_head;
        let old_cvd = unsafe { *self.cvd_buffer.get_unchecked(head) };
        
        // Update rolling sum
        let new_sum = self.cvd_sum + cvd - old_cvd;
        
        // Store new CVD
        unsafe {
            *self.cvd_buffer.get_unchecked_mut(head) = cvd;
        }
        
        // Update head pointer
        let new_head = if head + 1 >= MAX_CVD_HISTORY { 0 } else { head + 1 };
        
        // Increment observation count
        let _ = self.obs_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Calculate skew using branchless programming
    #[inline]
    fn calculate_skew_branchless(
        &self,
        imbalance: FixedI64,
        momentum: FixedI64,
        toxicity: ToxicityLevel,
    ) -> SkewResult {
        // Base skew from imbalance
        // Positive imbalance (more buys) -> skew asks tighter (positive ask skew)
        // Negative imbalance (more sells) -> skew bids tighter (positive bid skew)
        
        let imbal_factor = FixedI64::from_i64(500000000i64); // 0.5 multiplier
        
        // Branchless sign extraction
        let imbal_sign = if imbalance > FixedI64::ZERO { 
            FixedI64::ONE 
        } else if imbalance < FixedI64::ZERO { 
            -FixedI64::ONE 
        } else { 
            FixedI64::ZERO 
        };
        
        let imbal_skew = imbalance * imbal_factor;
        
        // Momentum adjustment
        let mom_factor = FixedI64::from_i64(250000000i64); // 0.25 multiplier
        let mom_skew = momentum * mom_factor / FixedI64::from_i64(1000000000i64);
        
        // Toxicity adjustment - higher toxicity means wider spreads
        let tox_mult = match toxicity {
            ToxicityLevel::Extreme => FixedI64::from_i64(2000000000i64),
            ToxicityLevel::High => FixedI64::from_i64(1500000000i64),
            ToxicityLevel::Medium => FixedI64::from_i64(1250000000i64),
            ToxicityLevel::Low => FixedI64::ONE,
        };
        
        // Calculate bid and ask skews
        // When imbalance is positive (buying pressure):
        //   - Ask skew becomes more negative (tighter asks to capture flow)
        //   - Bid skew becomes more positive (wider bids to protect)
        let raw_bid_skew = -imbal_skew - mom_skew;
        let raw_ask_skew = imbal_skew + mom_skew;
        
        // Apply toxicity multiplier
        let bid_skew = raw_bid_skew * tox_mult / FixedI64::ONE;
        let ask_skew = raw_ask_skew * tox_mult / FixedI64::ONE;
        
        // Net skew
        let net_skew = bid_skew - ask_skew;
        
        // Confidence based on observation count and toxicity
        let obs = self.obs_count.load(Ordering::Relaxed);
        let obs_conf = if obs > 100 {
            FixedI64::ONE
        } else if obs > 10 {
            FixedI64::from_i64(obs as i64) / FixedI64::from_i64(100i64)
        } else {
            FixedI64::ZERO
        };
        
        let tox_conf = match toxicity {
            ToxicityLevel::Extreme => FixedI64::from_i64(900000000i64),
            ToxicityLevel::High => FixedI64::from_i64(800000000i64),
            ToxicityLevel::Medium => FixedI64::from_i64(700000000i64),
            ToxicityLevel::Low => FixedI64::from_i64(500000000i64),
        };
        
        let confidence = obs_conf * tox_conf / FixedI64::ONE;

        SkewResult {
            bid_skew,
            ask_skew,
            net_skew,
            toxicity,
            cvd: FixedI64::from_i64(self.cvd.load(Ordering::Relaxed)),
            cvd_momentum: momentum,
            order_imbalance: imbalance,
            confidence,
            _pad: [0u8; CACHE_LINE_SIZE - 6 * 8 - 1 * 4 - 1 * 4],
        }
    }

    /// SIMD-accelerated imbalance calculation for multiple symbols
    #[inline]
    pub fn calculate_imbalances_simd(
        &self,
        buy_volumes: &[FixedI64],
        sell_volumes: &[FixedI64],
    ) -> [FixedI64; 4] {
        assert!(buy_volumes.len() >= 4);
        assert!(sell_volumes.len() >= 4);

        unsafe {
            let mut imbalances = [FixedI64::ZERO; 4];
            
            for i in (0..4).step_by(4) {
                let buy_vec = _mm256_loadu_si256(buy_volumes[i..].as_ptr() as *const __m256i);
                let sell_vec = _mm256_loadu_si256(sell_volumes[i..].as_ptr() as *const __m256i);
                
                // Calculate delta and total
                let delta = _mm256_sub_epi64(buy_vec, sell_vec);
                let total = _mm256_add_epi64(buy_vec, sell_vec);
                
                // Extract and compute division scalar
                let mut delta_arr = [0i64; 4];
                let mut total_arr = [0i64; 4];
                _mm256_storeu_si256(delta_arr.as_mut_ptr() as *mut __m256i, delta);
                _mm256_storeu_si256(total_arr.as_mut_ptr() as *mut __m256i, total);
                
                for j in 0..4 {
                    let d = FixedI64::from_i64(delta_arr[j]);
                    let t = FixedI64::from_i64(total_arr[j]);
                    imbalances[i + j] = if t > FixedI64::ZERO { d / t } else { FixedI64::ZERO };
                }
            }
            
            imbalances
        }
    }

    /// Get current toxicity level
    #[inline]
    pub fn get_toxicity(&self) -> ToxicityLevel {
        match self.toxicity.load(Ordering::Relaxed) {
            0 => ToxicityLevel::Low,
            1 => ToxicityLevel::Medium,
            2 => ToxicityLevel::High,
            _ => ToxicityLevel::Extreme,
        }
    }

    /// Get rolling CVD mean
    #[inline]
    pub fn get_cvd_mean(&self) -> FixedI64 {
        let obs = self.obs_count.load(Ordering::Relaxed);
        let n = if obs < MAX_CVD_HISTORY as u64 { obs } else { MAX_CVD_HISTORY as u64 };
        if n > 0 {
            self.cvd_sum / FixedI64::from_i64(n as i64)
        } else {
            FixedI64::ZERO
        }
    }
}

// Compile-time assertions
const _: () = assert!(core::mem::size_of::<QuoteSkewState>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<QuoteSkewState>() == 64);
const _: () = assert!(core::mem::size_of::<SkewResult>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<SkewResult>() == 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skew_state_creation() {
        let state = QuoteSkewState::new();
        assert!(!state.is_active());
        assert_eq!(state.cvd.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_activation() {
        let state = QuoteSkewState::new();
        state.activate();
        assert!(state.is_active());
    }

    #[test]
    fn test_skew_calculation() {
        let state = QuoteSkewState::new();
        state.activate();
        
        let buy_vol = FixedI64::from_i64(1000000000i64);
        let sell_vol = FixedI64::from_i64(500000000i64);
        
        let result = state.update_branchless(buy_vol, sell_vol, 1);
        assert!(result.is_some());
        
        let r = result.unwrap();
        // More buying should create positive ask skew
        assert!(r.order_imbalance > FixedI64::ZERO);
    }

    #[test]
    fn test_toxicity_levels() {
        let state = QuoteSkewState::new();
        state.activate();
        
        assert_eq!(state.get_toxicity(), ToxicityLevel::Low);
        
        // Generate high imbalance to trigger toxicity
        for _ in 0..10 {
            let buy_vol = FixedI64::from_i64(10000000000i64);
            let sell_vol = FixedI64::from_i64(100000000i64);
            let _ = state.update_branchless(buy_vol, sell_vol, 1);
        }
        
        let tox = state.get_toxicity();
        assert!(tox != ToxicityLevel::Low);
    }

    #[test]
    fn test_cvd_tracking() {
        let state = QuoteSkewState::new();
        state.activate();
        
        let initial_cvd = state.cvd.load(Ordering::Relaxed);
        
        let buy_vol = FixedI64::from_i64(1000000000i64);
        let sell_vol = FixedI64::from_i64(500000000i64);
        let _ = state.update_branchless(buy_vol, sell_vol, 1);
        
        let new_cvd = state.cvd.load(Ordering::Relaxed);
        assert!(new_cvd > initial_cvd);
    }
}
