//! Fear & Greed Composite Index
//! Low-latency sentiment from options skew and funding rates.
//! Branchless computation with lock-free updates.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Index valid flag
static INDEX_VALID: AtomicBool = AtomicBool::new(false);

/// Fear & Greed levels
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SentimentLevel {
    ExtremeFear = 0,
    Fear = 1,
    Neutral = 2,
    Greed = 3,
    ExtremeGreed = 4,
}

/// Fear & Greed Index - cache-line aligned
#[repr(C)]
pub struct FearGreedIndex {
    /// Options put/call ratio (scaled by 10^6)
    pub pc_ratio: i64,
    /// Options skew (25d put - 25d call IV, scaled by 10^6)
    pub options_skew: i64,
    /// Perpetual funding rate (scaled by 10^8)
    pub funding_rate: i64,
    /// Basis (futures - spot, scaled by 10^8)
    pub basis: i64,
    /// Volatility index level (scaled by 10^6)
    pub vol_index: i64,
    /// Momentum indicator (scaled by 10^6)
    pub momentum: i64,
    
    /// Component weights (scaled by 10^6), sum to 1_000_000
    pub weights: [i32; 6],
    
    /// Computed index value (0-100, scaled by 100 = 0-10000)
    pub index_value: AtomicI64,
    /// Previous index value
    pub prev_value: AtomicI64,
    /// Rate of change
    pub roc: AtomicI64,
    
    /// Extreme fear threshold
    pub fear_threshold: i64,
    /// Extreme greed threshold
    pub greed_threshold: i64,
    
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 24],
}

impl Default for FearGreedIndex {
    fn default() -> Self {
        Self {
            pc_ratio: 1_000_000, // 1.0 neutral
            options_skew: 0,
            funding_rate: 0,
            basis: 0,
            vol_index: 500_000, // 50% neutral
            momentum: 0,
            
            // Equal weights
            weights: [200_000, 200_000, 200_000, 150_000, 150_000, 100_000],
            
            index_value: AtomicI64::new(5000), // 50 neutral
            prev_value: AtomicI64::new(5000),
            roc: AtomicI64::new(0),
            
            fear_threshold: 2500,   // 25
            greed_threshold: 7500,  // 75
            
            valid: AtomicBool::new(false),
            _pad: [0u8; 24],
        }
    }
}

impl FearGreedIndex {
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Update all components and compute index
    #[inline(always)]
    pub fn update(
        &self,
        pc_ratio: i64,
        options_skew: i64,
        funding_rate: i64,
        basis: i64,
        vol_index: i64,
        momentum: i64,
    ) -> i64 {
        if !INDEX_VALID.load(Ordering::Relaxed) || !self.valid.load(Ordering::Acquire) {
            return 5000;
        }
        
        // Store raw values
        self.pc_ratio = pc_ratio;
        self.options_skew = options_skew;
        self.funding_rate = funding_rate;
        self.basis = basis;
        self.vol_index = vol_index;
        self.momentum = momentum;
        
        // Compute component scores (0-10000 each)
        let scores = [
            self.score_pc_ratio(pc_ratio),
            self.score_options_skew(options_skew),
            self.score_funding(funding_rate),
            self.score_basis(basis),
            self.score_vol(vol_index),
            self.score_momentum(momentum),
        ];
        
        // Weighted average (branchless)
        let mut weighted_sum = 0i64;
        let mut weight_sum = 0i64;
        for i in 0..6 {
            weighted_sum += scores[i] as i64 * self.weights[i] as i64;
            weight_sum += self.weights[i] as i64;
        }
        
        let index = if weight_sum > 0 {
            weighted_sum / weight_sum
        } else {
            5000
        };
        
        // Update rate of change
        let prev = self.prev_value.swap(index, Ordering::AcqRel);
        self.roc.store(index - prev, Ordering::Release);
        
        index
    }
    
    /// P/C ratio score: high ratio = fear (low score)
    #[inline(always)]
    fn score_pc_ratio(&self, ratio: i64) -> i64 {
        // ratio of 1.0 = 5000, ratio > 1.5 = low score, ratio < 0.7 = high score
        let normalized = 5000 - ((ratio - 1_000_000) * 5);
        normalized.clamp(0, 10000)
    }
    
    /// Options skew score: positive skew (puts expensive) = fear
    #[inline(always)]
    fn score_options_skew(&self, skew: i64) -> i64 {
        // skew of 0 = 5000, skew > 10% = low score
        let normalized = 5000 - (skew / 2);
        normalized.clamp(0, 10000)
    }
    
    /// Funding rate score: positive funding = greed
    #[inline(always)]
    fn score_funding(&self, rate: i64) -> i64 {
        // rate of 0 = 5000, rate > 0.1% daily = high score
        let normalized = 5000 + (rate / 10_000);
        normalized.clamp(0, 10000)
    }
    
    /// Basis score: contango = greed
    #[inline(always)]
    fn score_basis(&self, basis: i64) -> i64 {
        let normalized = 5000 + (basis / 100_000);
        normalized.clamp(0, 10000)
    }
    
    /// Volatility score: high vol = fear
    #[inline(always)]
    fn score_vol(&self, vol: i64) -> i64 {
        // vol of 50% = 5000, vol > 80% = low score
        let normalized = 5000 - ((vol - 500_000) * 10);
        normalized.clamp(0, 10000)
    }
    
    /// Momentum score: positive momentum = greed
    #[inline(always)]
    fn score_momentum(&self, mom: i64) -> i64 {
        let normalized = 5000 + (mom * 5);
        normalized.clamp(0, 10000)
    }
    
    /// Get current index value (0-100 scale)
    #[inline(always)]
    pub fn get_index(&self) -> i64 {
        self.index_value.load(Ordering::Acquire)
    }
    
    /// Get sentiment level
    #[inline(always)]
    pub fn get_level(&self) -> SentimentLevel {
        let value = self.get_index();
        
        // Branchless level detection
        let is_extreme_fear = (value < self.fear_threshold) as usize;
        let is_fear = (value >= self.fear_threshold && value < 4000) as usize;
        let is_greed = (value > self.greed_threshold && value <= 6000) as usize;
        let is_extreme_greed = (value > self.greed_threshold) as usize;
        
        match () {
            _ if is_extreme_fear != 0 => SentimentLevel::ExtremeFear,
            _ if is_fear != 0 => SentimentLevel::Fear,
            _ if is_extreme_greed != 0 && value > 6000 => SentimentLevel::ExtremeGreed,
            _ if is_greed != 0 => SentimentLevel::Greed,
            _ => SentimentLevel::Neutral,
        }
    }
    
    /// Check if in extreme fear (buy signal)
    #[inline(always)]
    pub fn is_extreme_fear(&self) -> bool {
        self.get_index() < self.fear_threshold
    }
    
    /// Check if in extreme greed (sell signal)
    #[inline(always)]
    pub fn is_extreme_greed(&self) -> bool {
        self.get_index() > self.greed_threshold
    }
    
    /// Mark index as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        INDEX_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        INDEX_VALID.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_index_bounds(
            pc_ratio in 500_000i64..2_000_000i64,
            skew in -100_000i64..100_000i64,
            funding in -100_000i64..100_000i64,
        ) {
            let fg = FearGreedIndex::new();
            fg.mark_valid();
            
            let index = fg.update(
                pc_ratio,
                skew,
                funding,
                0,
                500_000,
                0,
            );
            
            assert!(index >= 0 && index <= 10000, "Index must be in [0, 100]: {}", index);
        }
        
        #[test]
        fn test_extreme_fear_detection(pc_ratio in 1_500_000i64..3_000_000i64) {
            let fg = FearGreedIndex::new();
            fg.fear_threshold = 3000;
            fg.mark_valid();
            
            // High P/C ratio should indicate fear
            fg.update(pc_ratio, 50_000, -50_000, 0, 800_000, -100_000);
            
            assert!(fg.get_index() < 5000, "Should indicate fear");
        }
    }
    
    #[test]
    fn test_neutral_reading() {
        let fg = FearGreedIndex::new();
        fg.mark_valid();
        
        let index = fg.update(
            1_000_000, // Neutral P/C
            0,         // Neutral skew
            0,         // Neutral funding
            0,         // Neutral basis
            500_000,   // Neutral vol
            0,         // Neutral momentum
        );
        
        assert_eq!(index, 5000, "Neutral inputs should give 50");
    }
}
