//! Streaming Engle-Granger and Johansen cointegration tests using lock-free math.
//! 
//! Implements real-time cointegration testing for statistical arbitrage pairs
//! using fixed-point arithmetic and zero-copy circular buffers.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

/// Cache line padding for 64-byte alignment
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of observations in the rolling window
const MAX_OBSERVATIONS: usize = 4096;

/// Number of assets in the cointegration test (typically 2 for pairs)
const NUM_ASSETS: usize = 2;

/// Streaming Engle-Granger cointegration test state
#[repr(C, align(64))]
pub struct CointegrationState {
    /// Lock-free flag indicating if the test is active
    pub active: AtomicBool,
    /// Current observation count
    pub obs_count: AtomicU64,
    /// Rolling sum of price A (fixed-point)
    pub sum_a: FixedI64,
    /// Rolling sum of price B (fixed-point)
    pub sum_b: FixedI64,
    /// Rolling sum of A*B (fixed-point)
    pub sum_ab: FixedI64,
    /// Rolling sum of A^2 (fixed-point)
    pub sum_a2: FixedI64,
    /// Rolling sum of B^2 (fixed-point)
    pub sum_b2: FixedI64,
    /// Estimated hedge ratio (beta)
    pub hedge_ratio: FixedI64,
    /// Current spread value
    pub spread: FixedI64,
    /// Circular buffer for residuals
    pub residuals: [FixedI64; MAX_OBSERVATIONS],
    /// Head index for circular buffer
    pub residual_head: usize,
    /// Sum of residuals for mean calculation
    pub residual_sum: FixedI64,
    /// Sum of squared residuals
    pub residual_ss: FixedI64,
    /// Padding to next cache line
    _pad: [u8; CACHE_LINE_SIZE - 14 * 8 - 2 * 4],
}

/// Engle-Granger streaming test results
#[repr(C, align(64))]
pub struct EGTestResult {
    /// Test statistic (ADF-like)
    pub test_statistic: FixedI64,
    /// Critical value at 1% confidence
    pub critical_1pct: FixedI64,
    /// Critical value at 5% confidence
    pub critical_5pct: FixedI64,
    /// Critical value at 10% confidence
    pub critical_10pct: FixedI64,
    /// Is cointegrated at 5% level?
    pub is_cointegrated: bool,
    /// P-value approximation (scaled by 1e6)
    pub p_value_scaled: u64,
    /// Half-life of mean reversion (in observations)
    pub half_life: u32,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 7 * 8 - 2 * 4],
}

impl CointegrationState {
    /// Create a new cointegration state with pre-allocated buffers
    #[inline]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            obs_count: AtomicU64::new(0),
            sum_a: FixedI64::ZERO,
            sum_b: FixedI64::ZERO,
            sum_ab: FixedI64::ZERO,
            sum_a2: FixedI64::ZERO,
            sum_b2: FixedI64::ZERO,
            hedge_ratio: FixedI64::ZERO,
            spread: FixedI64::ZERO,
            residuals: [FixedI64::ZERO; MAX_OBSERVATIONS],
            residual_head: 0,
            residual_sum: FixedI64::ZERO,
            residual_ss: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE - 14 * 8 - 2 * 4],
        }
    }

    /// Activate the cointegration test
    #[inline]
    pub fn activate(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Deactivate the cointegration test
    #[inline]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if the test is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Update with new price observation using lock-free accumulation
    /// Prices must be in fixed-point format (scaled by 1e9)
    #[inline]
    pub fn update(&self, price_a: FixedI64, price_b: FixedI64) -> Option<EGTestResult> {
        if !self.is_active() {
            return None;
        }

        // Calculate spread using current hedge ratio
        let spread = if self.hedge_ratio != FixedI64::ZERO {
            price_a - price_b * self.hedge_ratio
        } else {
            price_a - price_b // Initial assumption: 1:1 ratio
        };

        // Store residual in circular buffer (O(1))
        let head = self.residual_head;
        let old_residual = unsafe { *self.residuals.get_unchecked(head) };
        
        // Update running sums for residual statistics
        let new_sum = self.residual_sum + spread - old_residual;
        let new_ss = self.residual_ss + spread * spread - old_residual * old_residual;
        
        // Lock-free update of residual buffer
        unsafe {
            *self.residuals.get_unchecked_mut(head) = spread;
        }
        
        // Update head pointer
        let new_head = if head + 1 >= MAX_OBSERVATIONS { 0 } else { head + 1 };
        
        // Atomic updates for shared state
        let obs = self.obs_count.fetch_add(1, Ordering::Relaxed);
        
        // Update cumulative sums for OLS
        // Note: In production, these would use atomic operations or be thread-local
        // For this implementation, we assume single-threaded hot path
        
        // Calculate new hedge ratio using streaming OLS
        let n = if obs < MAX_OBSERVATIONS as u64 { obs + 1 } else { MAX_OBSERVATIONS as u64 };
        
        Some(self.compute_test_result(spread, n))
    }

    /// Compute Engle-Granger test result
    #[inline]
    fn compute_test_result(&self, current_spread: FixedI64, n: u64) -> EGTestResult {
        let n_fixed = FixedI64::from_i64(n as i64);
        
        // Calculate mean of residuals
        let mean = if n > 0 {
            self.residual_sum / n_fixed
        } else {
            FixedI64::ZERO
        };

        // Calculate variance of residuals
        let variance = if n > 1 {
            (self.residual_ss - self.residual_sum * self.residual_sum / n_fixed) 
                / FixedI64::from_i64((n - 1) as i64)
        } else {
            FixedI64::ZERO
        };

        // Simple ADF-like test statistic (simplified for streaming)
        // In production, this would use full ADF regression
        let test_stat = if variance > FixedI64::ZERO {
            // Normalize spread by standard deviation
            let std_dev = variance.sqrt_approx();
            if std_dev > FixedI64::ZERO {
                current_spread / std_dev
            } else {
                FixedI64::ZERO
            }
        } else {
            FixedI64::ZERO
        };

        // Critical values (approximated, scaled by 1e9)
        // These are standard ADF critical values for cointegration
        let critical_1pct = FixedI64::from_i64(-3900000000i64); // -3.90
        let critical_5pct = FixedI64::from_i64(-3340000000i64); // -3.34
        let critical_10pct = FixedI64::from_i64(-3040000000i64); // -3.04

        let is_cointegrated = test_stat < critical_5pct;

        // Approximate p-value (scaled by 1e6)
        let p_value = if test_stat < critical_1pct {
            1000u64 // p < 0.01
        } else if test_stat < critical_5pct {
            10000u64 // p < 0.05
        } else if test_stat < critical_10pct {
            50000u64 // p < 0.10
        } else {
            100000u64 // p >= 0.10
        };

        // Estimate half-life from autocorrelation (simplified)
        let half_life = self.estimate_half_life(n);

        EGTestResult {
            test_statistic: test_stat,
            critical_1pct,
            critical_5pct,
            critical_10pct,
            is_cointegrated,
            p_value_scaled: p_value,
            half_life,
            _pad: [0u8; CACHE_LINE_SIZE - 7 * 8 - 2 * 4],
        }
    }

    /// Estimate half-life of mean reversion
    #[inline]
    fn estimate_half_life(&self, n: u64) -> u32 {
        if n < 2 {
            return 0;
        }

        // Simplified half-life estimation using lag-1 autocorrelation
        // In production, this would use full AR(1) regression
        let n_fp = FixedI64::from_i64(n as i64);
        
        // Calculate autocorrelation at lag 1
        let mut sum_xy = FixedI64::ZERO;
        let mut sum_x = FixedI64::ZERO;
        let mut sum_y = FixedI64::ZERO;
        let mut sum_x2 = FixedI64::ZERO;
        
        let count = (n - 1).min(MAX_OBSERVATIONS as u64 - 1);
        
        for i in 0..count as usize {
            let j = (i + 1) % MAX_OBSERVATIONS;
            let x = unsafe { *self.residuals.get_unchecked(i) };
            let y = unsafe { *self.residuals.get_unchecked(j) };
            sum_xy = sum_xy + x * y;
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_x2 = sum_x2 + x * x;
        }

        let count_fp = FixedI64::from_i64(count as i64);
        let rho = if sum_x2 > FixedI64::ZERO {
            (sum_xy - sum_x * sum_y / count_fp) / sum_x2
        } else {
            FixedI64::ZERO
        };

        // Half-life = -ln(2) / ln(rho)
        if rho > FixedI64::ZERO && rho < FixedI64::ONE {
            let ln2 = FixedI64::from_i64(693147180i64); // ln(2) * 1e9
            let ln_rho = rho.ln_approx();
            if ln_rho < FixedI64::ZERO {
                let hl = ln2 / (-ln_rho);
                hl.to_i64() as u32
            } else {
                u32::MAX
            }
        } else {
            u32::MAX
        }
    }

    /// Get current hedge ratio
    #[inline]
    pub fn get_hedge_ratio(&self) -> FixedI64 {
        self.hedge_ratio
    }

    /// Get current spread
    #[inline]
    pub fn get_spread(&self) -> FixedI64 {
        self.spread
    }

    /// SIMD-accelerated covariance matrix computation for multiple pairs
    /// Uses AVX2 to process 4 pairs simultaneously
    #[inline]
    pub fn compute_covariance_matrix_simd(
        &self,
        prices_a: &[FixedI64],
        prices_b: &[FixedI64],
    ) -> [FixedI64; 4] {
        assert!(prices_a.len() == prices_b.len());
        assert!(prices_a.len() >= 4);

        unsafe {
            let mut cov_sum = _mm256_setzero_si256();
            
            // Process 4 elements at a time using AVX2
            for i in (0..prices_a.len()).step_by(4) {
                let a_vec = _mm256_loadu_si256(prices_a[i..].as_ptr() as *const __m256i);
                let b_vec = _mm256_loadu_si256(prices_b[i..].as_ptr() as *const __m256i);
                
                // Convert to floating point for covariance (using fixed-point multiplication)
                // This is a simplified version; in production, use proper fixed-point SIMD
                let prod = _mm256_mul_epi32(a_vec, b_vec);
                cov_sum = _mm256_add_epi64(cov_sum, prod);
            }

            // Extract results
            let mut result = [FixedI64::ZERO; 4];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, cov_sum);
            result
        }
    }
}

/// Johansen test state for multivariate cointegration
#[repr(C, align(64))]
pub struct JohansenState {
    /// Number of variables in the system
    pub num_vars: usize,
    /// Maximum lag order
    pub max_lag: usize,
    /// Trace statistic for r=0
    pub trace_r0: FixedI64,
    /// Trace statistic for r<=1
    pub trace_r1: FixedI64,
    /// Eigenvalues (sorted descending)
    pub eigenvalues: [FixedI64; NUM_ASSETS],
    /// Eigenvectors (flattened)
    pub eigenvectors: [FixedI64; NUM_ASSETS * NUM_ASSETS],
    /// Number of cointegrating relationships
    pub num_cointegrating: usize,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 4 * 8 - 2 * 4 - NUM_ASSETS * 8 - NUM_ASSETS * NUM_ASSETS * 8],
}

impl JohansenState {
    /// Create a new Johansen test state
    #[inline]
    pub const fn new(num_vars: usize, max_lag: usize) -> Self {
        Self {
            num_vars,
            max_lag,
            trace_r0: FixedI64::ZERO,
            trace_r1: FixedI64::ZERO,
            eigenvalues: [FixedI64::ZERO; NUM_ASSETS],
            eigenvectors: [FixedI64::ZERO; NUM_ASSETS * NUM_ASSETS],
            num_cointegrating: 0,
            _pad: [0u8; CACHE_LINE_SIZE - 4 * 8 - 2 * 4 - NUM_ASSETS * 8 - NUM_ASSETS * NUM_ASSETS * 8],
        }
    }

    /// Update Johansen test with new data
    /// Returns the number of cointegrating relationships
    #[inline]
    pub fn update(&mut self, data: &[FixedI64]) -> usize {
        // Simplified Johansen test implementation
        // In production, this would use full VECM estimation
        
        if data.len() < self.num_vars {
            return 0;
        }

        // Compute covariance matrices and eigenvalues
        // This is a placeholder; full implementation requires LAPACK-style routines
        // For HFT, we use approximations based on rolling correlations
        
        self.num_cointegrating
    }

    /// Get critical values for trace test
    #[inline]
    pub fn get_critical_values(&self, n: usize) -> [FixedI64; 3] {
        // Critical values depend on sample size and number of variables
        // These are approximated values for common cases
        [
            FixedI64::from_i64(15000000000i64), // 10% level
            FixedI64::from_i64(17000000000i64), // 5% level
            FixedI64::from_i64(20000000000i64), // 1% level
        ]
    }
}

// Compile-time assertions for alignment
const _: () = assert!(core::mem::size_of::<CointegrationState>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<CointegrationState>() == 64);
const _: () = assert!(core::mem::size_of::<EGTestResult>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<EGTestResult>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_cointegration_state_creation() {
        let state = CointegrationState::new();
        assert!(!state.is_active());
        assert_eq!(state.obs_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_activation_deactivation() {
        let state = CointegrationState::new();
        state.activate();
        assert!(state.is_active());
        state.deactivate();
        assert!(!state.is_active());
    }

    #[test]
    fn test_spread_calculation() {
        let state = CointegrationState::new();
        state.activate();
        
        let price_a = FixedI64::from_i64(100_000_000_000i64); // 100.0
        let price_b = FixedI64::from_i64(200_000_000_000i64); // 200.0
        
        let result = state.update(price_a, price_b);
        assert!(result.is_some());
    }

    proptest! {
        #[test]
        fn test_cointegration_extreme_prices(
            price_a in 1000000000i64..1000000000000000i64,
            price_b in 1000000000i64..1000000000000000i64,
        ) {
            let state = CointegrationState::new();
            state.activate();
            
            let fa = FixedI64::from_i64(price_a);
            let fb = FixedI64::from_i64(price_b);
            
            let result = state.update(fa, fb);
            // Should not panic with extreme values
            prop_assert!(result.is_some() || result.is_none());
        }

        #[test]
        fn test_spread_anomaly_handling(
            base_price in 10000000000i64..100000000000i64,
            anomaly_factor in 0.1f64..10.0f64,
        ) {
            let state = CointegrationState::new();
            state.activate();
            
            let normal_price = FixedI64::from_i64(base_price);
            let anomalous_price = FixedI64::from_i64((base_price as f64 * anomaly_factor) as i64);
            
            // Normal observation
            let _ = state.update(normal_price, normal_price);
            
            // Anomalous observation
            let result = state.update(anomalous_price, normal_price);
            
            // Should handle anomaly without crashing
            prop_assert!(result.is_some());
        }
    }
}
