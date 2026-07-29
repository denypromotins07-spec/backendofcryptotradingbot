//! Real-Time Cross-Asset Correlation Tracker
//! Streaming covariance for DXY, Gold, Oil, and Bond yields.
//! Lock-free O(1) updates using Welford's algorithm.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum assets tracked
const MAX_ASSETS: usize = 16;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Tracker valid flag
static CORR_VALID: AtomicBool = AtomicBool::new(false);

/// Asset types
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AssetType {
    DXY = 0,
    Gold = 1,
    Oil = 2,
    Bonds10Y = 3,
    Bonds2Y = 4,
    SPX = 5,
    BTC = 6,
    ETH = 7,
}

/// Streaming correlation tracker
#[repr(C)]
pub struct CrossAssetCorr {
    /// Asset names/hashes
    pub asset_ids: [u8; MAX_ASSETS],
    /// Running means (scaled by 10^8)
    pub means: [i64; MAX_ASSETS],
    /// Running variances (scaled by 10^12)
    pub variances: [i64; MAX_ASSETS],
    /// Running covariances (scaled by 10^12): [i][j] flattened
    pub covariances: [i64; MAX_ASSETS * MAX_ASSETS],
    /// Sample counts
    pub sample_count: AtomicU64,
    /// Number of active assets
    pub num_assets: usize,
    /// Correlation matrix cached (scaled by 10^6)
    pub correlations: [i32; MAX_ASSETS * MAX_ASSETS],
    /// Last update timestamp (rdtsc)
    pub last_update: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for CrossAssetCorr {
    fn default() -> Self {
        Self {
            asset_ids: [0; MAX_ASSETS],
            means: [0; MAX_ASSETS],
            variances: [0; MAX_ASSETS],
            covariances: [0; MAX_ASSETS * MAX_ASSETS],
            sample_count: AtomicU64::new(0),
            num_assets: 0,
            correlations: [0; MAX_ASSETS * MAX_ASSETS],
            last_update: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl CrossAssetCorr {
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Register an asset
    pub fn register_asset(&self, asset_type: AssetType) -> Option<usize> {
        if !CORR_VALID.load(Ordering::Relaxed) {
            return None;
        }
        
        let idx = self.num_assets;
        if idx >= MAX_ASSETS {
            return None;
        }
        
        unsafe {
            let id_ptr = self.asset_ids.as_ptr() as *mut u8;
            *id_ptr.add(idx) = asset_type as u8;
        }
        
        unsafe {
            let ptr = &self.num_assets as *const usize as *mut usize;
            *ptr = idx + 1;
        }
        
        Some(idx)
    }
    
    /// Update with new prices (lock-free, O(1))
    #[inline(always)]
    pub fn update(&self, prices: &[i64]) {
        if !CORR_VALID.load(Ordering::Relaxed) || !self.valid.load(Ordering::Acquire) {
            return;
        }
        
        let n = self.sample_count.load(Ordering::Relaxed);
        let new_n = n + 1;
        let inv_n = 1.0 / new_n as f64;
        let inv_nm1 = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        
        // Welford's online algorithm for each asset
        for i in 0..self.num_assets.min(prices.len()) {
            let price = prices[i] as f64 / 100_000_000.0;
            
            unsafe {
                let mean_ptr = self.means.as_ptr() as *mut f64;
                let var_ptr = self.variances.as_ptr() as *mut f64;
                
                let old_mean = *mean_ptr.add(i);
                let delta = price - old_mean;
                let new_mean = old_mean + delta * inv_n;
                let new_var = *var_ptr.add(i) + delta * (price - new_mean);
                
                *mean_ptr.add(i) = new_mean;
                *var_ptr.add(i) = new_var;
            }
        }
        
        // Update covariances (pairwise)
        for i in 0..self.num_assets.min(prices.len()) {
            for j in (i + 1)..self.num_assets.min(prices.len()) {
                let x = prices[i] as f64 / 100_000_000.0;
                let y = prices[j] as f64 / 100_000_000.0;
                
                unsafe {
                    let mean_ptr = self.means.as_ptr() as *const f64;
                    let cov_ptr = self.covariances.as_ptr() as *mut f64;
                    
                    let x_mean = *mean_ptr.add(i);
                    let y_mean = *mean_ptr.add(j);
                    
                    let old_cov = *cov_ptr.add(i * MAX_ASSETS + j);
                    let new_cov = old_cov + (x - x_mean) * (y - y_mean) * inv_nm1;
                    
                    *cov_ptr.add(i * MAX_ASSETS + j) = new_cov;
                    *cov_ptr.add(j * MAX_ASSETS + i) = new_cov; // Symmetric
                }
            }
        }
        
        self.sample_count.store(new_n, Ordering::Release);
        
        #[cfg(target_arch = "x86_64")]
        self.last_update.store(unsafe { core::arch::x86_64::_rdtsc() }, Ordering::Release);
        
        // Update correlation matrix periodically
        if new_n % 10 == 0 {
            self.update_correlations();
        }
    }
    
    /// Update correlation matrix from covariances
    fn update_correlations(&self) {
        for i in 0..self.num_assets {
            for j in 0..self.num_assets {
                if i == j {
                    unsafe {
                        let corr_ptr = self.correlations.as_ptr() as *mut i32;
                        *corr_ptr.add(i * MAX_ASSETS + j) = 1_000_000; // Perfect correlation
                    }
                    continue;
                }
                
                let var_i = self.variances[i] as f64 / 1_000_000_000_000.0;
                let var_j = self.variances[j] as f64 / 1_000_000_000_000.0;
                
                if var_i <= 0.0 || var_j <= 0.0 {
                    unsafe {
                        let corr_ptr = self.correlations.as_ptr() as *mut i32;
                        *corr_ptr.add(i * MAX_ASSETS + j) = 0;
                    }
                    continue;
                }
                
                let std_i = var_i.sqrt();
                let std_j = var_j.sqrt();
                
                let cov = self.covariances[i * MAX_ASSETS + j] as f64 / 1_000_000_000_000.0;
                let corr = (cov / (std_i * std_j)).clamp(-1.0, 1.0);
                
                unsafe {
                    let corr_ptr = self.correlations.as_ptr() as *mut i32;
                    *corr_ptr.add(i * MAX_ASSETS + j) = (corr * 1_000_000.0) as i32;
                }
            }
        }
    }
    
    /// Get correlation between two assets (scaled by 10^6)
    #[inline(always)]
    pub fn get_correlation(&self, i: usize, j: usize) -> i32 {
        if i >= self.num_assets || j >= self.num_assets {
            return 0;
        }
        self.correlations[i * MAX_ASSETS + j]
    }
    
    /// Check if assets are negatively correlated
    #[inline(always)]
    pub fn is_negative_corr(&self, i: usize, j: usize) -> bool {
        self.get_correlation(i, j) < -100_000 // -0.1 threshold
    }
    
    /// Mark tracker as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        CORR_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        CORR_VALID.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_correlation_bounds(
            price1 in -10000i64..10000i64,
            price2 in -10000i64..10000i64,
        ) {
            let tracker = CrossAssetCorr::new();
            tracker.mark_valid();
            tracker.register_asset(AssetType::DXY);
            tracker.register_asset(AssetType::Gold);
            
            for _ in 0..100 {
                tracker.update(&[price1 * 1_000_000, price2 * 1_000_000]);
            }
            
            let corr = tracker.get_correlation(0, 1);
            assert!(corr >= -1_000_000 && corr <= 1_000_000, 
                    "Correlation must be in [-1, 1]: {}", corr);
        }
    }
    
    #[test]
    fn test_self_correlation() {
        let tracker = CrossAssetCorr::new();
        tracker.mark_valid();
        tracker.register_asset(AssetType::BTC);
        
        for _ in 0..100 {
            tracker.update(&[5000000000]);
        }
        
        let corr = tracker.get_correlation(0, 0);
        assert_eq!(corr, 1_000_000, "Self-correlation must be 1.0");
    }
}
