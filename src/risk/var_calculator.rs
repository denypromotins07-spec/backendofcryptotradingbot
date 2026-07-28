//! Value at Risk (VaR) Calculator
//! 
//! Intraday VaR and Expected Shortfall calculator using streaming ticks.
//! Uses SIMD intrinsics for vectorized portfolio risk calculations.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of assets in portfolio
const MAX_ASSETS: usize = 128;

/// Number of buckets for histogram-based VaR
const HISTOGRAM_BUCKETS: usize = 256;

/// Streaming statistics accumulator - no heap allocation
#[repr(C)]
struct StreamingStats {
    count: AtomicU64,
    sum: AtomicI64,
    sum_sq: AtomicI64,
    min: AtomicI64,
    max: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

impl StreamingStats {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum: AtomicI64::new(0),
            sum_sq: AtomicI64::new(0),
            min: AtomicI64::new(i64::MAX),
            max: AtomicI64::new(i64::MIN),
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }

    #[inline(always)]
    fn update(&self, value: i64) {
        let count = self.count.fetch_add(1, Ordering::Relaxed);
        let _ = self.sum.fetch_add(value, Ordering::Relaxed);
        let _ = self.sum_sq.fetch_add(value * value, Ordering::Relaxed);
        
        // Atomic min/max (branchless using compare-exchange loop would be slower)
        // Using relaxed ordering for performance in hot path
        let current_min = self.min.load(Ordering::Relaxed);
        if value < current_min {
            self.min.store(value, Ordering::Relaxed);
        }
        
        let current_max = self.max.load(Ordering::Relaxed);
        if value > current_max {
            self.max.store(value, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn mean(&self) -> i64 {
        let count = self.count.load(Ordering::Acquire);
        if count == 0 {
            return 0;
        }
        self.sum.load(Ordering::Acquire) / count as i64
    }

    #[inline(always)]
    fn variance(&self) -> u64 {
        let count = self.count.load(Ordering::Acquire) as i64;
        if count <= 1 {
            return 0;
        }
        let sum = self.sum.load(Ordering::Acquire) as i128;
        let sum_sq = self.sum_sq.load(Ordering::Acquire) as i128;
        
        // Variance = E[X^2] - E[X]^2
        let mean_sq = (sum * sum) / (count * count);
        let variance = (sum_sq / count) - mean_sq;
        
        if variance < 0 {
            0
        } else {
            variance as u64
        }
    }

    #[inline(always)]
    fn std_dev(&self) -> u64 {
        let var = self.variance();
        // Integer square root using Newton's method
        if var == 0 {
            return 0;
        }
        let mut x = var;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + var / x) / 2;
        }
        x
    }

    #[inline(always)]
    fn reset(&self) {
        self.count.store(0, Ordering::Release);
        self.sum.store(0, Ordering::Release);
        self.sum_sq.store(0, Ordering::Release);
        self.min.store(i64::MAX, Ordering::Release);
        self.max.store(i64::MIN, Ordering::Release);
    }
}

/// Histogram bucket for distribution tracking
#[repr(C)]
struct HistogramBucket {
    count: AtomicU64,
    value_sum: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 16],
}

impl HistogramBucket {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            value_sum: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 16],
        }
    }
}

/// VaR calculation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VaRResult {
    pub var_95: i64,      // 95% VaR
    pub var_99: i64,      // 99% VaR
    pub expected_shortfall_95: i64,
    pub expected_shortfall_99: i64,
    pub volatility: u64,
    pub _padding: [u8; 16],
}

impl VaRResult {
    const fn empty() -> Self {
        Self {
            var_95: 0,
            var_99: 0,
            expected_shortfall_95: 0,
            expected_shortfall_99: 0,
            volatility: 0,
            _padding: [0u8; 16],
        }
    }
}

/// Per-asset risk state
#[repr(C)]
struct AssetRiskState {
    returns_stats: StreamingStats,
    position_value: AtomicI64,
    last_price: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

impl AssetRiskState {
    const fn new() -> Self {
        Self {
            returns_stats: StreamingStats::new(),
            position_value: AtomicI64::new(0),
            last_price: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// Intraday VaR Calculator with SIMD acceleration
pub struct VaRCalculator {
    /// Per-asset risk states
    assets: [AssetRiskState; MAX_ASSETS],
    /// Portfolio-level statistics
    portfolio_stats: StreamingStats,
    /// Histogram for empirical VaR calculation
    histogram: [HistogramBucket; HISTOGRAM_BUCKETS],
    /// Histogram range (min/max in basis points)
    hist_min_bp: AtomicI64,
    hist_max_bp: AtomicI64,
    /// Confidence level multipliers (pre-computed)
    z_95: i64,
    z_99: i64,
    /// Lookback window in ticks
    lookback_window: u64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for VaRCalculator {}
unsafe impl Sync for VaRCalculator {}

impl VaRCalculator {
    /// Create new VaR calculator
    pub const fn new() -> Self {
        const EMPTY_ASSET: AssetRiskState = AssetRiskState::new();
        const EMPTY_BUCKET: HistogramBucket = HistogramBucket::new();
        Self {
            assets: [EMPTY_ASSET; MAX_ASSETS],
            portfolio_stats: StreamingStats::new(),
            histogram: [EMPTY_BUCKET; HISTOGRAM_BUCKETS],
            hist_min_bp: AtomicI64::new(-10000),
            hist_max_bp: AtomicI64::new(10000),
            z_95: 1645,  // 1.645 * 1000
            z_99: 2326,  // 2.326 * 1000
            lookback_window: 10000,
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Set lookback window size
    #[inline(always)]
    pub fn set_lookback_window(&mut self, window: u64) {
        self.lookback_window = window;
    }

    /// Update price for an asset (streaming tick)
    #[inline(always)]
    pub fn update_price(&self, asset_id: usize, price: u64, position_value: i64) {
        if asset_id >= MAX_ASSETS || price == 0 {
            return;
        }

        let asset = &self.assets[asset_id];
        let last_price = asset.last_price.swap(price, Ordering::AcqRel);
        asset.position_value.store(position_value, Ordering::Release);

        if last_price > 0 {
            // Calculate return in basis points (scaled by 10000)
            let return_bp = ((price as i64 - last_price as i64) * 10000) / last_price as i64;
            asset.returns_stats.update(return_bp);

            // Update portfolio stats weighted by position
            let weighted_return = (return_bp as i128 * position_value as i128) / 1_000_000;
            self.portfolio_stats.update(weighted_return as i64);

            // Update histogram for empirical VaR
            self.update_histogram(return_bp);
        }
    }

    /// Update histogram bucket (branchless)
    #[inline(always)]
    fn update_histogram(&self, return_bp: i64) {
        let min_bp = self.hist_min_bp.load(Ordering::Relaxed);
        let max_bp = self.hist_max_bp.load(Ordering::Relaxed);
        let range = max_bp - min_bp;
        
        if range == 0 {
            return;
        }

        // Map return to bucket index (branchless clamping)
        let normalized = return_bp - min_bp;
        let bucket_idx = ((normalized * HISTOGRAM_BUCKETS as i64) / range) as usize;
        let bucket_idx = bucket_idx.min(HISTOGRAM_BUCKETS - 1);

        self.histogram[bucket_idx].count.fetch_add(1, Ordering::Relaxed);
        self.histogram[bucket_idx].value_sum.fetch_add(return_bp, Ordering::Relaxed);
    }

    /// Calculate VaR using parametric method (variance-covariance)
    #[inline(always)]
    pub fn calculate_parametric_var(&self, portfolio_value: i64) -> VaRResult {
        let volatility = self.portfolio_stats.std_dev();
        let mean = self.portfolio_stats.mean();

        // VaR = Mean - Z * StdDev
        // Using fixed-point arithmetic (values scaled by 1000)
        let var_95_bp = mean - (self.z_95 * volatility as i64) / 1000;
        let var_99_bp = mean - (self.z_99 * volatility as i64) / 1000;

        // Convert to monetary terms
        let var_95 = (var_95_bp as i128 * portfolio_value as i128) / 1_000_000;
        let var_99 = (var_99_bp as i128 * portfolio_value as i128) / 1_000_000;

        // Expected Shortfall (CVaR) approximation
        // ES ≈ VaR * (1 + 0.1 * Z) for normal distribution
        let es_factor_95 = 1000 + self.z_95 / 10;
        let es_factor_99 = 1000 + self.z_99 / 10;
        
        let es_95 = (var_95 * es_factor_95 as i128) / 1000;
        let es_99 = (var_99 * es_factor_99 as i128) / 1000;

        VaRResult {
            var_95: var_95 as i64,
            var_99: var_99 as i64,
            expected_shortfall_95: es_95 as i64,
            expected_shortfall_99: es_99 as i64,
            volatility,
            _padding: [0u8; 16],
        }
    }

    /// Calculate VaR using historical simulation (histogram-based)
    #[inline(always)]
    pub fn calculate_historical_var(&self, portfolio_value: i64) -> VaRResult {
        let total_count = self.portfolio_stats.count.load(Ordering::Acquire);
        if total_count < 100 {
            return VaRResult::empty();
        }

        // Find 95th and 99th percentile buckets
        let threshold_95 = (total_count * 5) / 100;
        let threshold_99 = (total_count * 1) / 100;

        let mut cumulative = 0u64;
        let mut var_95_bucket = 0usize;
        let mut var_99_bucket = 0usize;
        let mut found_95 = false;
        let mut found_99 = false;

        // Scan from left (worst returns) to find percentiles
        for i in 0..HISTOGRAM_BUCKETS {
            cumulative += self.histogram[i].count.load(Ordering::Relaxed);
            
            if !found_99 && cumulative >= threshold_99 {
                var_99_bucket = i;
                found_99 = true;
            }
            if !found_95 && cumulative >= threshold_95 {
                var_95_bucket = i;
                found_95 = true;
                break;
            }
        }

        // Convert bucket indices to return values
        let min_bp = self.hist_min_bp.load(Ordering::Relaxed);
        let max_bp = self.hist_max_bp.load(Ordering::Relaxed);
        let range = (max_bp - min_bp) / HISTOGRAM_BUCKETS as i64;

        let var_95_bp = min_bp + (var_95_bucket as i64 * range);
        let var_99_bp = min_bp + (var_99_bucket as i64 * range);

        // Calculate Expected Shortfall (average of tail losses)
        let mut tail_sum_95 = 0i64;
        let mut tail_count_95 = 0u64;
        for i in 0..=var_95_bucket {
            let count = self.histogram[i].count.load(Ordering::Relaxed);
            if count > 0 {
                tail_sum_95 += self.histogram[i].value_sum.load(Ordering::Relaxed);
                tail_count_95 += count;
            }
        }

        let mut tail_sum_99 = 0i64;
        let mut tail_count_99 = 0u64;
        for i in 0..=var_99_bucket {
            let count = self.histogram[i].count.load(Ordering::Relaxed);
            if count > 0 {
                tail_sum_99 += self.histogram[i].value_sum.load(Ordering::Relaxed);
                tail_count_99 += count;
            }
        }

        let es_95_bp = if tail_count_95 > 0 { tail_sum_95 / tail_count_95 as i64 } else { var_95_bp };
        let es_99_bp = if tail_count_99 > 0 { tail_sum_99 / tail_count_99 as i64 } else { var_99_bp };

        // Convert to monetary terms
        let var_95 = (var_95_bp as i128 * portfolio_value as i128) / 1_000_000;
        let var_99 = (var_99_bp as i128 * portfolio_value as i128) / 1_000_000;
        let es_95 = (es_95_bp as i128 * portfolio_value as i128) / 1_000_000;
        let es_99 = (es_99_bp as i128 * portfolio_value as i128) / 1_000_000;

        VaRResult {
            var_95: var_95 as i64,
            var_99: var_99 as i64,
            expected_shortfall_95: es_95 as i64,
            expected_shortfall_99: es_99 as i64,
            volatility: self.portfolio_stats.std_dev(),
            _padding: [0u8; 16],
        }
    }

    /// SIMD-accelerated portfolio VaR calculation
    /// Calculates VaR for multiple assets in parallel using AVX2
    #[inline(always)]
    pub fn calculate_portfolio_var_simd(&self, portfolio_value: i64) -> VaRResult {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.calculate_portfolio_var_avx2(portfolio_value)
            } else {
                self.calculate_parametric_var(portfolio_value)
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(always)]
    unsafe fn calculate_portfolio_var_avx2(&self, portfolio_value: i64) -> VaRResult {
        // Use AVX2 to process 4 assets simultaneously
        let mut total_variance = 0u64;
        let mut total_mean = 0i64;
        let mut asset_count = 0u64;

        for i in (0..MAX_ASSETS).step_by(4) {
            // Load 4 asset statistics
            let mut variances = [0u64; 4];
            let mut means = [0i64; 4];
            
            for j in 0..4 {
                if i + j < MAX_ASSETS {
                    let pos_val = self.assets[i + j].position_value.load(Ordering::Relaxed);
                    if pos_val != 0 {
                        variances[j] = self.assets[i + j].returns_stats.variance();
                        means[j] = self.assets[i + j].returns_stats.mean();
                        asset_count += 1;
                    }
                }
            }

            // Vector sum using AVX2 intrinsics
            let var_vec = _mm256_loadu_si256(variances.as_ptr() as *const __m256i);
            let mean_vec = _mm256_loadu_si256(means.as_ptr() as *const __m256i);

            // Horizontal sum
            let var_lo = _mm256_castsi256_si128(var_vec);
            let var_hi = _mm256_extracti128_si256::<1>(var_vec);
            let var_sum = _mm_add_epi64(var_lo, var_hi);
            
            let mean_lo = _mm256_castsi256_si128(mean_vec);
            let mean_hi = _mm256_extracti128_si256::<1>(mean_vec);
            let mean_sum = _mm_add_epi64(mean_lo, mean_hi);

            total_variance += _mm_extract_epi64::<0>(var_sum) as u64 
                + _mm_extract_epi64::<1>(var_sum) as u64;
            total_mean += _mm_extract_epi64::<0>(mean_sum) as i64 
                + _mm_extract_epi64::<1>(mean_sum) as i64;
        }

        if asset_count == 0 {
            return VaRResult::empty();
        }

        // Average variance and mean
        let avg_variance = total_variance / asset_count;
        let avg_mean = total_mean / asset_count as i64;

        // Calculate volatility (integer sqrt)
        let volatility = if avg_variance == 0 {
            0
        } else {
            let mut x = avg_variance;
            let mut y = (x + 1) / 2;
            while y < x {
                x = y;
                y = (x + avg_variance / x) / 2;
            }
            x
        };

        // Calculate VaR
        let var_95_bp = avg_mean - (self.z_95 * volatility as i64) / 1000;
        let var_99_bp = avg_mean - (self.z_99 * volatility as i64) / 1000;

        let var_95 = (var_95_bp as i128 * portfolio_value as i128) / 1_000_000;
        let var_99 = (var_99_bp as i128 * portfolio_value as i128) / 1_000_000;

        VaRResult {
            var_95: var_95 as i64,
            var_99: var_99 as i64,
            expected_shortfall_95: ((var_95 * 1164) / 1000) as i64,
            expected_shortfall_99: ((var_99 * 1232) / 1000) as i64,
            volatility,
            _padding: [0u8; 16],
        }
    }

    /// Get current portfolio volatility
    #[inline(always)]
    pub fn get_volatility(&self) -> u64 {
        self.portfolio_stats.std_dev()
    }

    /// Reset all statistics
    #[inline(always)]
    pub fn reset(&self) {
        self.portfolio_stats.reset();
        for i in 0..MAX_ASSETS {
            self.assets[i].returns_stats.reset();
        }
        for i in 0..HISTOGRAM_BUCKETS {
            self.histogram[i].count.store(0, Ordering::Release);
            self.histogram[i].value_sum.store(0, Ordering::Release);
        }
    }

    /// Get number of observations
    #[inline(always)]
    pub fn get_observation_count(&self) -> u64 {
        self.portfolio_stats.count.load(Ordering::Acquire)
    }
}

impl Default for VaRCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_result_size() {
        assert_eq!(core::mem::size_of::<VaRResult>(), 48);
    }

    #[test]
    fn test_streaming_stats() {
        let stats = StreamingStats::new();
        
        for i in 1..=100 {
            stats.update(i * 100);
        }

        assert_eq!(stats.count.load(Ordering::Relaxed), 100);
        assert_eq!(stats.mean(), 5050);
        assert!(stats.variance() > 0);
    }

    #[test]
    fn test_parametric_var() {
        let var_calc = VaRCalculator::new();
        
        // Simulate some price movements
        for i in 0..1000 {
            let price = 50000 + (i % 100) as u64;
            var_calc.update_price(0, price, 1_000_000);
        }

        let result = var_calc.calculate_parametric_var(1_000_000);
        assert!(result.volatility > 0);
    }

    #[test]
    fn test_historical_var() {
        let var_calc = VaRCalculator::new();
        
        // Generate enough data for historical VaR
        for i in 0..5000 {
            let price = if i % 2 == 0 {
                50000 + (i % 200) as u64
            } else {
                50000 - (i % 200) as u64
            };
            var_calc.update_price(0, price, 1_000_000);
        }

        let result = var_calc.calculate_historical_var(1_000_000);
        // Historical VaR should be calculated
        assert!(result.var_95 <= 0 || result.var_99 <= 0);
    }

    #[test]
    fn test_multi_asset_var() {
        let var_calc = VaRCalculator::new();
        
        // Update multiple assets
        for asset in 0..10 {
            for i in 0..500 {
                let price = 50000 + (i % 100) as u64 * (asset as u64 + 1);
                var_calc.update_price(asset, price, 100_000);
            }
        }

        let result = var_calc.calculate_portfolio_var_simd(1_000_000);
        assert!(result.volatility > 0);
    }

    #[test]
    fn test_reset() {
        let var_calc = VaRCalculator::new();
        
        for i in 0..100 {
            var_calc.update_price(0, 50000 + i as u64, 1_000_000);
        }

        assert!(var_calc.get_observation_count() > 0);
        
        var_calc.reset();
        
        assert_eq!(var_calc.get_observation_count(), 0);
    }
}
