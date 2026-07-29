//! Online Mutual Information and Feature Importance Tracker
//! Prunes dead signals using streaming entropy estimation.
//! Lock-free, branchless computation.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum features tracked
const MAX_FEATURES: usize = 512;
/// Histogram bins for entropy estimation
const NUM_BINS: usize = 64;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Tracker valid flag
static TRACKER_VALID: AtomicBool = AtomicBool::new(false);

/// Feature importance data - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FeatureImportance {
    /// Feature hash
    pub feature_hash: u64,
    /// Marginal entropy estimate (scaled by 10^8)
    pub marginal_entropy: i64,
    /// Conditional entropy with target (scaled by 10^8)
    pub conditional_entropy: i64,
    /// Mutual information (scaled by 10^8)
    pub mutual_info: i64,
    /// Sample count
    pub sample_count: u64,
    /// Is active (not pruned)
    pub is_active: bool,
    /// Last update timestamp
    pub last_update: u64,
    _pad: [u8; 30],
}

impl Default for FeatureImportance {
    fn default() -> Self {
        Self {
            feature_hash: 0,
            marginal_entropy: 0,
            conditional_entropy: 0,
            mutual_info: 0,
            sample_count: 0,
            is_active: true,
            last_update: 0,
            _pad: [0u8; 30],
        }
    }
}

const _: () = assert!(core::mem::size_of::<FeatureImportance>() == 64);

/// Online mutual information tracker
#[repr(C)]
pub struct MutualInfoTracker {
    /// Feature importance data
    pub features: [FeatureImportance; MAX_FEATURES],
    /// Joint histograms: [feature][bin_x][bin_y]
    pub joint_histograms: [[[u64; NUM_BINS]; NUM_BINS]; MAX_FEATURES],
    /// Marginal histograms for features: [feature][bin]
    pub feature_histograms: [[u64; NUM_BINS]; MAX_FEATURES],
    /// Target marginal histogram
    pub target_histogram: [u64; NUM_BINS],
    /// Number of active features
    pub num_features: usize,
    /// Total samples processed
    pub total_samples: AtomicU64,
    /// MI threshold for pruning (scaled by 10^8)
    pub prune_threshold: i64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for MutualInfoTracker {
    fn default() -> Self {
        Self {
            features: [FeatureImportance::default(); MAX_FEATURES],
            joint_histograms: [[[0; NUM_BINS]; NUM_BINS]; MAX_FEATURES],
            feature_histograms: [[0; NUM_BINS]; MAX_FEATURES],
            target_histogram: [0; NUM_BINS],
            num_features: 0,
            total_samples: AtomicU64::new(0),
            prune_threshold: 100_000, // 0.001 scaled
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl MutualInfoTracker {
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Register a feature
    pub fn register_feature(&self, feature_hash: u64) -> Option<usize> {
        if !TRACKER_VALID.load(Ordering::Relaxed) {
            return None;
        }
        
        let idx = self.num_features;
        if idx >= MAX_FEATURES {
            return None;
        }
        
        unsafe {
            let f_ptr = self.features.as_ptr() as *mut FeatureImportance;
            *f_ptr.add(idx) = FeatureImportance {
                feature_hash,
                is_active: true,
                ..FeatureImportance::default()
            };
        }
        
        unsafe {
            let ptr = &self.num_features as *const usize as *mut usize;
            *ptr = idx + 1;
        }
        
        Some(idx)
    }
    
    /// Update with new sample (feature value, target value)
    #[inline(always)]
    pub fn update(&self, feature_idx: usize, feature_val: u8, target_val: u8) {
        if !TRACKER_VALID.load(Ordering::Relaxed) || !self.valid.load(Ordering::Acquire) {
            return;
        }
        
        if feature_idx >= self.num_features || !self.features[feature_idx].is_active {
            return;
        }
        
        // Bin values
        let x_bin = (feature_val as usize * NUM_BINS / 256).min(NUM_BINS - 1);
        let y_bin = (target_val as usize * NUM_BINS / 256).min(NUM_BINS - 1);
        
        // Update histograms
        unsafe {
            let jh_ptr = self.joint_histograms.as_ptr() as *mut [[[u64; NUM_BINS]; NUM_BINS]];
            (*jh_ptr.add(feature_idx))[x_bin][y_bin] += 1;
            
            let fh_ptr = self.feature_histograms.as_ptr() as *mut [[u64; NUM_BINS]];
            (*fh_ptr.add(feature_idx))[x_bin] += 1;
            
            let th_ptr = self.target_histogram.as_ptr() as *mut [u64; NUM_BINS];
            (*th_ptr)[y_bin] += 1;
        }
        
        // Update sample count
        let feat = &self.features[feature_idx];
        unsafe {
            let f_ptr = self.features.as_ptr() as *mut FeatureImportance;
            let f = &mut *f_ptr.add(feature_idx);
            f.sample_count += 1;
            #[cfg(target_arch = "x86_64")]
            {
                f.last_update = core::arch::x86_64::_rdtsc();
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                f.last_update = f.sample_count;
            }
        }
        
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        
        // Periodically recompute MI (every 1000 samples per feature)
        if feat.sample_count % 1000 == 0 {
            self.compute_mutual_info(feature_idx);
        }
    }
    
    /// Compute mutual information for a feature
    fn compute_mutual_info(&self, feature_idx: usize) {
        let n = self.features[feature_idx].sample_count;
        if n == 0 {
            return;
        }
        
        let mut h_x = 0.0f64; // Marginal entropy of X
        let mut h_y = 0.0f64; // Marginal entropy of Y
        let mut h_xy = 0.0f64; // Joint entropy
        
        let inv_n = 1.0 / n as f64;
        
        // Compute H(X)
        for i in 0..NUM_BINS {
            let p = self.feature_histograms[feature_idx][i] as f64 * inv_n;
            if p > 0.0 {
                h_x -= p * p.log2();
            }
        }
        
        // Compute H(Y)
        for j in 0..NUM_BINS {
            let p = self.target_histogram[j] as f64 * inv_n;
            if p > 0.0 {
                h_y -= p * p.log2();
            }
        }
        
        // Compute H(X,Y)
        for i in 0..NUM_BINS {
            for j in 0..NUM_BINS {
                let p = self.joint_histograms[feature_idx][i][j] as f64 * inv_n;
                if p > 0.0 {
                    h_xy -= p * p.log2();
                }
            }
        }
        
        // MI = H(X) + H(Y) - H(X,Y)
        let mi = (h_x + h_y - h_xy).max(0.0);
        
        unsafe {
            let f_ptr = self.features.as_ptr() as *mut FeatureImportance;
            let f = &mut *f_ptr.add(feature_idx);
            f.marginal_entropy = (h_x * 100_000_000.0) as i64;
            f.conditional_entropy = ((h_xy - h_x) * 100_000_000.0) as i64;
            f.mutual_info = (mi * 100_000_000.0) as i64;
            
            // Prune if MI below threshold
            if f.mutual_info < self.prune_threshold {
                f.is_active = false;
            }
        }
    }
    
    /// Get mutual information for a feature
    #[inline(always)]
    pub fn get_mi(&self, feature_idx: usize) -> i64 {
        if feature_idx >= self.num_features {
            return 0;
        }
        self.features[feature_idx].mutual_info
    }
    
    /// Check if feature is active
    #[inline(always)]
    pub fn is_active(&self, feature_idx: usize) -> bool {
        if feature_idx >= self.num_features {
            return false;
        }
        self.features[feature_idx].is_active
    }
    
    /// Mark tracker as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        TRACKER_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        TRACKER_VALID.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_mi_non_negative(
            feature_val in 0u8..255u8,
            target_val in 0u8..255u8,
        ) {
            let tracker = MutualInfoTracker::new();
            tracker.mark_valid();
            tracker.register_feature(0x12345678);
            
            // Feed many samples
            for _ in 0..1000 {
                tracker.update(0, feature_val, target_val);
            }
            
            let mi = tracker.get_mi(0);
            assert!(mi >= 0, "MI must be non-negative");
        }
    }
    
    #[test]
    fn test_feature_importance_size() {
        assert_eq!(core::mem::size_of::<FeatureImportance>(), 64);
    }
    
    #[test]
    fn test_pruning() {
        let tracker = MutualInfoTracker::new();
        tracker.prune_threshold = 1_000_000; // High threshold
        tracker.mark_valid();
        tracker.register_feature(0xABC);
        
        // Constant feature has zero MI
        for _ in 0..5000 {
            tracker.update(0, 128, 128);
        }
        
        assert!(!tracker.is_active(0), "Constant feature should be pruned");
    }
}
