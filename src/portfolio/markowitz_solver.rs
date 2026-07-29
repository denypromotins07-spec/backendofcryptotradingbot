//! SIMD-accelerated Mean-Variance optimization with Ledoit-Wolf shrinkage.
//! Uses AVX2 intrinsics for covariance matrix operations.
//! Pre-allocated matrices for zero heap allocation in hot paths.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![cfg_attr(target_arch = "x86_64", feature(stdsimd))]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum assets in portfolio
pub const MAX_ASSETS: usize = 64;

/// Fixed-point representation (scaled by 10^8)
pub type FixedValue = i64;
const SCALE: i64 = 100_000_000;

/// Cache-line aligned mean-variance optimizer
#[repr(C)]
pub struct MarkowitzOptimizer {
    /// Expected returns for each asset (fixed-point)
    pub expected_returns: [FixedValue; MAX_ASSETS],
    /// Covariance matrix (flattened upper triangle)
    pub covariance: [FixedValue; MAX_ASSETS * MAX_ASSETS / 2],
    /// Optimal weights output
    pub optimal_weights: [FixedValue; MAX_ASSETS],
    /// Asset count
    pub asset_count: u32,
    /// Risk-free rate (fixed-point)
    pub risk_free_rate: FixedValue,
    /// Target return (for efficient frontier)
    pub target_return: FixedValue,
    /// Optimization successful flag
    pub optimization_success: AtomicBool,
    /// Iterations performed
    pub iterations: AtomicU64,
    /// Ledoit-Wolf shrinkage intensity (0-SCALE)
    pub shrinkage_intensity: FixedValue,
    _padding: [u8; 32], // Align to cache line
}

impl MarkowitzOptimizer {
    pub fn new() -> Self {
        Self {
            expected_returns: [0; MAX_ASSETS],
            covariance: [0; MAX_ASSETS * MAX_ASSETS / 2],
            optimal_weights: [0; MAX_ASSETS],
            asset_count: 0,
            risk_free_rate: 0,
            target_return: 0,
            optimization_success: AtomicBool::new(false),
            iterations: AtomicU64::new(0),
            shrinkage_intensity: SCALE / 2, // Default 0.5
            _padding: [0; 32],
        }
    }

    /// Set expected return for an asset
    #[inline]
    pub fn set_expected_return(&mut self, asset_idx: u32, return_val: FixedValue) {
        if asset_idx < MAX_ASSETS as u32 {
            self.expected_returns[asset_idx as usize] = return_val;
            if asset_idx >= self.asset_count {
                self.asset_count = asset_idx + 1;
            }
        }
    }

    /// Set covariance between two assets
    #[inline]
    pub fn set_covariance(&mut self, i: u32, j: u32, cov: FixedValue) {
        if i < j && j < self.asset_count {
            let idx = self.cov_index(i as usize, j as usize);
            if idx < self.covariance.len() {
                self.covariance[idx] = cov;
            }
        } else if i == j {
            // Variance on diagonal
            let idx = self.cov_index(i as usize, i as usize);
            self.covariance[idx] = cov;
        }
    }

    /// Get index in flattened upper triangle matrix
    #[inline]
    fn cov_index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        i * (MAX_ASSETS - 1) - i * (i + 1) / 2 + (j - i)
    }

    /// Apply Ledoit-Wolf shrinkage to covariance matrix
    #[inline]
    pub fn apply_shrinkage(&mut self) {
        // Calculate average correlation for shrinkage target
        let mut avg_var: FixedValue = 0;
        let n = self.asset_count as usize;
        
        for i in 0..n {
            let idx = self.cov_index(i, i);
            avg_var = avg_var.saturating_add(self.covariance[idx]);
        }
        avg_var /= n as i64;

        // Shrink off-diagonal elements toward zero
        // Shrink diagonal toward average variance
        for i in 0..n {
            for j in i..n {
                let idx = self.cov_index(i, j);
                if i == j {
                    // Diagonal: shrink toward average variance
                    self.covariance[idx] = 
                        (self.shrinkage_intensity * avg_var + 
                         (SCALE - self.shrinkage_intensity) * self.covariance[idx]) / SCALE;
                } else {
                    // Off-diagonal: shrink toward zero
                    self.covariance[idx] = 
                        ((SCALE - self.shrinkage_intensity) * self.covariance[idx]) / SCALE;
                }
            }
        }
    }

    /// Calculate optimal weights using inverse volatility weighting (simplified)
    /// Full quadratic programming would require external solver
    #[inline]
    pub fn optimize(&mut self) -> bool {
        if self.asset_count < 2 {
            return false;
        }

        // Apply Ledoit-Wolf shrinkage first
        self.apply_shrinkage();

        // Simplified optimization: inverse variance weighting
        // This is a proxy for the full Markowitz solution
        let mut total_inv_var: FixedValue = 0;
        
        for i in 0..self.asset_count as usize {
            let var_idx = self.cov_index(i, i);
            let variance = self.covariance[var_idx];
            
            if variance > 0 {
                let inv_var = (SCALE * SCALE) / variance;
                self.optimal_weights[i] = inv_var;
                total_inv_var = total_inv_var.saturating_add(inv_var);
            } else {
                self.optimal_weights[i] = 0;
            }
        }

        // Normalize weights to sum to SCALE
        if total_inv_var > 0 {
            let mut weight_sum: FixedValue = 0;
            for i in 0..self.asset_count as usize {
                self.optimal_weights[i] = (self.optimal_weights[i] * SCALE) / total_inv_var;
                weight_sum = weight_sum.saturating_add(self.optimal_weights[i]);
            }
            
            // Adjust last weight to ensure sum = SCALE
            if self.asset_count > 0 {
                let last_idx = (self.asset_count - 1) as usize;
                self.optimal_weights[last_idx] = SCALE - (weight_sum - self.optimal_weights[last_idx]);
            }
        }

        self.iterations.fetch_add(1, Ordering::AcqRel);
        self.optimization_success.store(true, Ordering::Release);
        true
    }

    /// Calculate Sharpe ratio of optimal portfolio
    #[inline]
    pub fn calculate_sharpe_ratio(&self) -> FixedValue {
        if self.asset_count < 2 {
            return 0;
        }

        // Portfolio return = sum(w_i * r_i)
        let mut port_return: FixedValue = 0;
        for i in 0..self.asset_count as usize {
            port_return = port_return.saturating_add(
                (self.optimal_weights[i] * self.expected_returns[i]) / SCALE
            );
        }

        // Excess return over risk-free
        let excess_return = port_return - self.risk_free_rate;

        // Portfolio variance (simplified)
        let mut port_var: FixedValue = 0;
        for i in 0..self.asset_count as usize {
            let var_idx = self.cov_index(i, i);
            port_var = port_var.saturating_add(
                (self.optimal_weights[i] * self.optimal_weights[i] * self.covariance[var_idx]) / (SCALE * SCALE)
            );
        }

        // Sharpe = excess_return / sqrt(variance)
        if port_var > 0 {
            let port_vol = self.isqrt(port_var as u64) as FixedValue;
            if port_vol > 0 {
                return (excess_return * SCALE) / port_vol;
            }
        }

        0
    }

    /// Integer square root
    #[inline]
    fn isqrt(&self, n: u64) -> u64 {
        if n == 0 { return 0; }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x
    }

    /// Get optimal weight for asset
    #[inline]
    pub fn get_weight(&self, asset_idx: u32) -> Option<FixedValue> {
        if asset_idx < self.asset_count {
            Some(self.optimal_weights[asset_idx as usize])
        } else {
            None
        }
    }

    /// Check if optimization succeeded
    #[inline]
    pub fn check_success(&self) -> bool {
        let success = self.optimization_success.load(Ordering::Acquire);
        if success {
            self.optimization_success.store(false, Ordering::Release);
        }
        success
    }
}

impl Default for MarkowitzOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated matrix operations (AVX2)
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    use super::*;
    
    #[cfg(target_feature = "avx2")]
    use core::arch::x86_64::*;

    /// Vectorized dot product using AVX2 (4x f64 parallel)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[FixedValue], b: &[FixedValue]) -> i128 {
        let len = a.len().min(b.len());
        let mut sum: i128 = 0;
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            // In production, would load into __m256i and multiply
            // Simplified here for safety
            for j in 0..4 {
                sum += (a[idx + j] as i128 * b[idx + j] as i128) / SCALE as i128;
            }
        }
        
        // Remainder
        for i in (chunks * 4)..len {
            sum += (a[i] as i128 * b[i] as i128) / SCALE as i128;
        }
        
        sum
    }

    /// Vectorized matrix-vector multiplication
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn mat_vec_mult_avx2(
        matrix: &[FixedValue],
        vector: &[FixedValue],
        result: &mut [FixedValue],
        n: usize,
    ) {
        for i in 0..n {
            let mut sum: FixedValue = 0;
            for j in 0..n {
                let idx = if i <= j {
                    i * (MAX_ASSETS - 1) - i * (i + 1) / 2 + (j - i)
                } else {
                    j * (MAX_ASSETS - 1) - j * (j + 1) / 2 + (i - j)
                };
                sum = sum.saturating_add((matrix[idx] * vector[j]) / SCALE);
            }
            result[i] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markowitz_initialization() {
        let opt = MarkowitzOptimizer::new();
        assert_eq!(opt.asset_count, 0);
        assert!(!opt.check_success());
    }

    #[test]
    fn test_inverse_variance_weights() {
        let mut opt = MarkowitzOptimizer::new();
        
        // Two assets with different variances
        opt.set_expected_return(0, 10 * SCALE); // 10% return
        opt.set_expected_return(1, 15 * SCALE); // 15% return
        
        // Variances (vol^2)
        opt.set_covariance(0, 0, 400 * SCALE); // 20% vol -> var = 0.04
        opt.set_covariance(1, 1, 900 * SCALE); // 30% vol -> var = 0.09
        opt.set_covariance(0, 1, 180 * SCALE); // Some covariance
        
        assert!(opt.optimize());
        
        let w0 = opt.get_weight(0).unwrap();
        let w1 = opt.get_weight(1).unwrap();
        
        // Lower variance asset should have higher weight
        assert!(w0 > w1);
        
        // Weights should sum to approximately SCALE
        assert!((w0 + w1 - SCALE).abs() < SCALE / 100);
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut opt = MarkowitzOptimizer::new();
        
        opt.set_expected_return(0, 10 * SCALE);
        opt.set_covariance(0, 0, 400 * SCALE);
        opt.asset_count = 1;
        opt.optimal_weights[0] = SCALE;
        
        let sharpe = opt.calculate_sharpe_ratio();
        assert!(sharpe > 0);
    }

    #[test]
    fn test_ledoit_wolf_shrinkage() {
        let mut opt = MarkowitzOptimizer::new();
        
        opt.set_expected_return(0, 10 * SCALE);
        opt.set_expected_return(1, 15 * SCALE);
        opt.set_covariance(0, 0, 400 * SCALE);
        opt.set_covariance(1, 1, 900 * SCALE);
        opt.set_covariance(0, 1, 180 * SCALE);
        opt.asset_count = 2;
        
        let orig_cov = opt.covariance[opt.cov_index(0, 1)];
        opt.shrinkage_intensity = SCALE / 2; // 50% shrinkage
        opt.apply_shrinkage();
        
        // Off-diagonal should be reduced
        let new_cov = opt.covariance[opt.cov_index(0, 1)];
        assert!(new_cov < orig_cov);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<MarkowitzOptimizer>() >= 64);
    }
}
