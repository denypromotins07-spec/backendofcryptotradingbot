//! Streaming PCA using Oja's Rule
//! Online dimensionality reduction for feature compression.
//! Lock-free, O(1) update complexity.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum input dimensions
const MAX_INPUT_DIM: usize = 256;
/// Maximum output components
const MAX_COMPONENTS: usize = 32;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// PCA valid flag
static PCA_VALID: AtomicBool = AtomicBool::new(false);

/// Streaming PCA with Oja's rule
#[repr(C)]
pub struct StreamingPCA {
    /// Weight matrix W (output x input), scaled by 10^8
    pub weights: [[i64; MAX_INPUT_DIM]; MAX_COMPONENTS],
    /// Eigenvalue estimates (scaled by 10^8)
    pub eigenvalues: [i64; MAX_COMPONENTS],
    /// Input dimension
    pub input_dim: usize,
    /// Output components
    pub num_components: usize,
    /// Learning rate (scaled by 10^8)
    pub learning_rate: i64,
    /// Update count
    pub update_count: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for StreamingPCA {
    fn default() -> Self {
        Self {
            weights: [[0; MAX_INPUT_DIM]; MAX_COMPONENTS],
            eigenvalues: [0; MAX_COMPONENTS],
            input_dim: 0,
            num_components: 0,
            learning_rate: 100_000, // 0.001 scaled
            update_count: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl StreamingPCA {
    pub const fn new(input_dim: usize, num_components: usize) -> Self {
        assert!(input_dim <= MAX_INPUT_DIM);
        assert!(num_components <= MAX_COMPONENTS);
        
        Self {
            weights: [[0; MAX_INPUT_DIM]; MAX_COMPONENTS],
            eigenvalues: [0; MAX_COMPONENTS],
            input_dim,
            num_components,
            learning_rate: 100_000,
            update_count: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
    
    /// Initialize weights with small random values (deterministic seed)
    pub fn initialize(&self, seed: u64) {
        let mut state = seed;
        for i in 0..self.num_components {
            for j in 0..self.input_dim {
                // Simple LCG for deterministic initialization
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) % 1000) as i64 - 500; // Small values around 0
                unsafe {
                    let w_ptr = self.weights.as_ptr() as *mut i64;
                    *w_ptr.add(i * MAX_INPUT_DIM + j) = val;
                }
            }
        }
    }
    
    /// Oja's rule update: W <- W + lr * y * (x - y * W)
    #[inline(always)]
    pub fn update(&self, input: &[i64]) -> [i64; MAX_COMPONENTS] {
        if !PCA_VALID.load(Ordering::Relaxed) || !self.valid.load(Ordering::Acquire) {
            return [0; MAX_COMPONENTS];
        }
        
        let mut output = [0i64; MAX_COMPONENTS];
        let lr = self.learning_rate as f64 / 100_000_000.0;
        
        // Forward pass: y = W * x
        for i in 0..self.num_components {
            let mut sum = 0i64;
            for j in 0..self.input_dim.min(input.len()) {
                sum = sum.saturating_add(
                    (self.weights[i][j] as i128 * input[j] as i128 / 100_000_000) as i64
                );
            }
            output[i] = sum / 100_000_000;
        }
        
        // Oja's rule update for each component
        for i in 0..self.num_components {
            let y = output[i] as f64 / 100_000_000.0;
            
            for j in 0..self.input_dim.min(input.len()) {
                let x = input[j] as f64 / 100_000_000.0;
                let w = self.weights[i][j] as f64 / 100_000_000.0;
                
                // Oja's rule: delta_w = lr * y * (x - y * w)
                let delta = lr * y * (x - y * w);
                
                unsafe {
                    let w_ptr = self.weights.as_ptr() as *mut i64;
                    let new_w = ((w + delta) * 100_000_000.0) as i64;
                    *w_ptr.add(i * MAX_INPUT_DIM + j) = new_w;
                }
            }
            
            // Update eigenvalue estimate (Rayleigh quotient)
            self.eigenvalues[i] = ((y * y) * 100_000_000.0) as i64;
        }
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        output
    }
    
    /// Transform input to principal components
    #[inline(always)]
    pub fn transform(&self, input: &[i64]) -> [i64; MAX_COMPONENTS] {
        let mut output = [0i64; MAX_COMPONENTS];
        
        for i in 0..self.num_components {
            let mut sum = 0i64;
            for j in 0..self.input_dim.min(input.len()) {
                sum = sum.saturating_add(
                    (self.weights[i][j] as i128 * input[j] as i128 / 100_000_000) as i64
                );
            }
            output[i] = sum / 100_000_000;
        }
        
        output
    }
    
    /// Mark PCA as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        PCA_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        PCA_VALID.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_pca_transform_deterministic(
            seed in any::<u64>(),
            input in prop::collection::vec(-1000i64..1000i64, 10),
        ) {
            let pca = StreamingPCA::new(10, 3);
            pca.initialize(seed);
            pca.mark_valid();
            
            let out1 = pca.transform(&input);
            let out2 = pca.transform(&input);
            
            assert_eq!(out1, out2, "Transform must be deterministic");
        }
    }
    
    #[test]
    fn test_pca_initialization() {
        let pca = StreamingPCA::new(16, 4);
        pca.initialize(42);
        
        // Weights should be initialized
        let mut non_zero = 0;
        for i in 0..4 {
            for j in 0..16 {
                if pca.weights[i][j] != 0 {
                    non_zero += 1;
                }
            }
        }
        assert!(non_zero > 0);
    }
}
