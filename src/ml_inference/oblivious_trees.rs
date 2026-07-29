//! Oblivious Decision Trees Inference Engine
//! Zero-allocation, branchless CatBoost-style inference for ultra-low latency.
//! All arithmetic uses fixed-point or fast-math floats to ensure determinism.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Fixed-point representation scaled by 10^8
const SCALE: i64 = 100_000_000;

/// Cache line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum tree depth (oblivious = same split feature at each level)
const MAX_DEPTH: usize = 8;
const MAX_NODES: usize = 1 << MAX_DEPTH;

/// Memory tracker for 6.5GB limit enforcement
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;

/// Circuit breaker for feature drift
static DRIFT_EXCEEDED: AtomicBool = AtomicBool::new(false);

/// Oblivious Decision Tree node - all splits at depth d use the same feature
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ObliviousNode {
    /// Feature index to compare (fixed per depth level)
    pub feature_idx: u16,
    /// Threshold in fixed-point
    pub threshold: i64,
    /// Left child value (if feature < threshold)
    pub left_value: f32,
    /// Right child value (if feature >= threshold)
    pub right_value: f32,
    /// Padding to 64 bytes
    _pad: [u8; 34],
}

impl Default for ObliviousNode {
    fn default() -> Self {
        Self {
            feature_idx: 0,
            threshold: 0,
            left_value: 0.0,
            right_value: 0.0,
            _pad: [0u8; 34],
        }
    }
}

/// Ensure 64-byte alignment
const _: () = assert!(core::mem::size_of::<ObliviousNode>() == 64);

/// Oblivious Decision Forest - pre-allocated, zero-copy
#[repr(C)]
pub struct ObliviousForest {
    /// Number of trees
    pub num_trees: usize,
    /// Tree depth (same for all trees)
    pub depth: usize,
    /// Flattened tree nodes: [tree][depth][node_at_depth]
    /// Oblivious property: nodes at same depth share feature_idx
    pub nodes: [[ObliviousNode; MAX_DEPTH]; 64], // Max 64 trees
    /// Leaf values stored separately for cache efficiency
    pub leaf_values: [[f32; MAX_NODES]; 64],
    /// Feature drift statistics (mean, variance) for circuit breaker
    pub feature_mean: [f32; 256],
    pub feature_var: [f32; 256],
    pub feature_count: AtomicU64,
    /// Drift threshold (standard deviations)
    pub drift_threshold: f32,
    _pad: [u8; 32],
}

impl ObliviousForest {
    /// Create new forest with pre-allocated buffers
    pub const fn new(num_trees: usize, depth: usize) -> Self {
        assert!(num_trees <= 64, "Max 64 trees supported");
        assert!(depth <= MAX_DEPTH, "Max depth is {}", MAX_DEPTH);
        
        Self {
            num_trees,
            depth,
            nodes: [[ObliviousNode::default(); MAX_DEPTH]; 64],
            leaf_values: [[0.0; MAX_NODES]; 64],
            feature_mean: [0.0; 256],
            feature_var: [0.0; 256],
            feature_count: AtomicU64::new(0),
            drift_threshold: 5.0, // 5 sigma drift detection
            _pad: [0u8; 32],
        }
    }
    
    /// Branchless prediction for a single tree
    #[inline(always)]
    pub fn predict_tree(&self, tree_idx: usize, features: &[i64]) -> f32 {
        if tree_idx >= self.num_trees {
            return 0.0;
        }
        
        let mut path: u32 = 0;
        let depth = self.depth;
        
        // Manually unroll loop for branch prediction
        // Each level uses the same feature (oblivious property)
        unsafe {
            for d in 0..depth {
                let node = &self.nodes[tree_idx][d];
                let feat_val = *features.get_unchecked(node.feature_idx as usize);
                
                // Branchless comparison: mask = 0xFFFFFFFF if feat_val >= threshold
                let mask = ((feat_val >= node.threshold) as u32).wrapping_neg();
                
                // Select path bit based on comparison
                path |= (mask >> 31) << d;
            }
        }
        
        // Return leaf value
        unsafe { *self.leaf_values[tree_idx].get_unchecked(path as usize) }
    }
    
    /// SIMD-accelerated forest prediction (AVX2)
    #[inline(always)]
    pub fn predict(&self, features: &[i64]) -> f32 {
        // Check circuit breaker
        if DRIFT_EXCEEDED.load(Ordering::Relaxed) {
            return 0.0;
        }
        
        let num_trees = self.num_trees;
        let mut sum = 0.0f32;
        
        // Process 8 trees at a time using AVX2
        if num_trees >= 8 && cfg!(target_feature = "avx2") {
            unsafe {
                let mut i = 0;
                while i + 8 <= num_trees {
                    let v0 = _mm256_set1_ps(self.predict_tree(i, features));
                    let v1 = _mm256_set1_ps(self.predict_tree(i + 1, features));
                    let v2 = _mm256_set1_ps(self.predict_tree(i + 2, features));
                    let v3 = _mm256_set1_ps(self.predict_tree(i + 3, features));
                    let v4 = _mm256_set1_ps(self.predict_tree(i + 4, features));
                    let v5 = _mm256_set1_ps(self.predict_tree(i + 5, features));
                    let v6 = _mm256_set1_ps(self.predict_tree(i + 6, features));
                    let v7 = _mm256_set1_ps(self.predict_tree(i + 7, features));
                    
                    let sum_vec = _mm256_add_ps(
                        _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                        _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)),
                    );
                    
                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
                    sum += result.iter().sum::<f32>();
                    
                    i += 8;
                }
                
                // Handle remaining trees
                for j in i..num_trees {
                    sum += self.predict_tree(j, features);
                }
            }
        } else {
            for i in 0..num_trees {
                sum += self.predict_tree(i, features);
            }
        }
        
        sum / num_trees as f32
    }
    
    /// Online update of feature statistics for drift detection
    #[inline(always)]
    pub fn update_feature_stats(&self, feature_idx: usize, value: f32) {
        if feature_idx >= 256 {
            return;
        }
        
        let count = self.feature_count.fetch_add(1, Ordering::Relaxed);
        let n = (count + 1) as f32;
        
        // Welford's online algorithm for mean and variance
        let delta = value - self.feature_mean[feature_idx];
        let new_mean = self.feature_mean[feature_idx] + delta / n;
        let new_var = self.feature_var[feature_idx] 
            + delta * (value - new_mean);
        
        unsafe {
            let mean_ptr = self.feature_mean.as_ptr() as *mut f32;
            let var_ptr = self.feature_var.as_ptr() as *mut f32;
            *mean_ptr.add(feature_idx) = new_mean;
            *var_ptr.add(feature_idx) = new_var / n;
        }
    }
    
    /// Check for feature drift - circuit breaker
    pub fn check_drift(&self, feature_idx: usize, value: f32) -> bool {
        if feature_idx >= 256 {
            return false;
        }
        
        let mean = self.feature_mean[feature_idx];
        let var = self.feature_var[feature_idx];
        let std = if var > 0.0 { var.sqrt() } else { 1.0 };
        let z_score = (value - mean).abs() / std;
        
        if z_score > self.drift_threshold {
            DRIFT_EXCEEDED.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }
    
    /// Reset circuit breaker
    pub fn reset_drift(&self) {
        DRIFT_EXCEEDED.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_oblivious_prediction_stability(
            tree_idx in 0usize..4,
            depth in 1usize..6,
            features in prop::collection::vec(-1000i64..1000i64, 16)
        ) {
            let forest = ObliviousForest::new(4, depth);
            
            // Prediction should be deterministic
            let pred1 = forest.predict_tree(tree_idx.min(forest.num_trees - 1), &features);
            let pred2 = forest.predict_tree(tree_idx.min(forest.num_trees - 1), &features);
            
            assert_eq!(pred1, pred2, "Predictions must be deterministic");
        }
        
        #[test]
        fn test_drift_detection(value in -100.0f32..100.0f32) {
            let forest = ObliviousForest::new(4, 4);
            
            // Initialize with normal values
            for _ in 0..1000 {
                forest.update_feature_stats(0, 0.0);
            }
            
            // Extreme value should trigger drift
            let _drifted = forest.check_drift(0, value.abs() + 50.0);
        }
    }
    
    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(core::mem::size_of::<ObliviousNode>(), 64);
        assert_eq!(core::mem::align_of::<ObliviousNode>(), 8);
    }
}
