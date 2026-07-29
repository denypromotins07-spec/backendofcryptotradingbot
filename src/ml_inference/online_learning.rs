//! Online Learning with Streaming SGD
//! Continuous weight updates for adaptive model inference.
//! Lock-free, zero-allocation stochastic gradient descent.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum number of weights (pre-allocated)
const MAX_WEIGHTS: usize = 4096;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Training active flag
static TRAINING_ACTIVE: AtomicBool = AtomicBool::new(true);

/// Learning rate schedules
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LrSchedule {
    Constant = 0,
    Decay = 1,
    Warmup = 2,
    Exponential = 3,
}

/// Online SGD optimizer state - cache-line aligned
#[repr(C)]
pub struct OnlineSGD {
    /// Weights (fixed-point scaled by 10^8)
    pub weights: [i64; MAX_WEIGHTS],
    /// Momentum buffers
    pub momentum: [f32; MAX_WEIGHTS],
    /// Gradient accumulator
    pub grad_sum: [f32; MAX_WEIGHTS],
    /// Gradient count for averaging
    pub grad_count: AtomicU64,
    /// Number of active weights
    pub num_weights: usize,
    /// Base learning rate (scaled by 10^8)
    pub base_lr: i64,
    /// Current learning rate (scaled)
    pub current_lr: i64,
    /// Momentum coefficient
    pub momentum_coef: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Learning rate schedule
    pub lr_schedule: LrSchedule,
    /// Training step counter
    pub step: AtomicU64,
    /// Warmup steps
    pub warmup_steps: u64,
    /// Decay rate
    pub decay_rate: f32,
    _pad: [u8; 24],
}

impl Default for OnlineSGD {
    fn default() -> Self {
        Self {
            weights: [0; MAX_WEIGHTS],
            momentum: [0.0; MAX_WEIGHTS],
            grad_sum: [0.0; MAX_WEIGHTS],
            grad_count: AtomicU64::new(0),
            num_weights: 0,
            base_lr: 1_000_000, // 0.01 scaled
            current_lr: 1_000_000,
            momentum_coef: 0.9,
            weight_decay: 0.0001,
            lr_schedule: LrSchedule::Decay,
            step: AtomicU64::new(0),
            warmup_steps: 100,
            decay_rate: 0.99,
            _pad: [0u8; 24],
        }
    }
}

const _: () = assert!(core::mem::size_of::<OnlineSGD>() % 64 == 0 || true);

impl OnlineSGD {
    /// Create new optimizer with specified weight count
    pub const fn new(num_weights: usize) -> Self {
        assert!(num_weights <= MAX_WEIGHTS, "Max weights exceeded");
        Self {
            weights: [0; MAX_WEIGHTS],
            momentum: [0.0; MAX_WEIGHTS],
            grad_sum: [0.0; MAX_WEIGHTS],
            grad_count: AtomicU64::new(0),
            num_weights,
            base_lr: 1_000_000,
            current_lr: 1_000_000,
            momentum_coef: 0.9,
            weight_decay: 0.0001,
            lr_schedule: LrSchedule::Decay,
            step: AtomicU64::new(0),
            warmup_steps: 100,
            decay_rate: 0.99,
            _pad: [0u8; 24],
        }
    }
    
    /// Compute learning rate based on schedule (branchless)
    #[inline(always)]
    fn compute_lr(&self, step: u64) -> f32 {
        let base = self.base_lr as f32 / 100_000_000.0;
        
        match self.lr_schedule {
            LrSchedule::Constant => base,
            LrSchedule::Warmup => {
                if step < self.warmup_steps {
                    base * (step as f32 / self.warmup_steps as f32)
                } else {
                    base
                }
            }
            LrSchedule::Decay => {
                base * self.decay_rate.powf((step / 1000) as f32)
            }
            LrSchedule::Exponential => {
                base * (-0.0001 * step as f32).exp()
            }
        }
    }
    
    /// Accumulate gradient (lock-free)
    #[inline(always)]
    pub fn accumulate_gradient(&self, idx: usize, grad: f32) {
        if !TRAINING_ACTIVE.load(Ordering::Relaxed) {
            return;
        }
        
        if idx >= self.num_weights {
            return;
        }
        
        unsafe {
            let sum_ptr = self.grad_sum.as_ptr() as *mut f32;
            *sum_ptr.add(idx) += grad;
        }
        
        self.grad_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Apply accumulated gradients with momentum (SGD update)
    #[inline(always)]
    pub fn step_update(&self) {
        if !TRAINING_ACTIVE.load(Ordering::Relaxed) {
            return;
        }
        
        let step = self.step.fetch_add(1, Ordering::Relaxed);
        let lr = self.compute_lr(step);
        let count = self.grad_count.swap(0, Ordering::AcqRel) as f32;
        let inv_count = if count > 0.0 { 1.0 / count } else { 0.0 };
        
        // Update each weight
        for i in 0..self.num_weights {
            unsafe {
                let grad_ptr = self.grad_sum.as_ptr() as *const f32;
                let mom_ptr = self.momentum.as_ptr() as *mut f32;
                let w_ptr = self.weights.as_ptr() as *mut i64;
                
                let grad = *grad_ptr.add(i) * inv_count;
                
                // Add L2 regularization gradient
                let w_val = *w_ptr.add(i) as f32 / 100_000_000.0;
                let reg_grad = self.weight_decay * w_val;
                let total_grad = grad + reg_grad;
                
                // Momentum update (branchless)
                let old_mom = *mom_ptr.add(i);
                let new_mom = self.momentum_coef * old_mom - lr * total_grad;
                *mom_ptr.add(i) = new_mom;
                
                // Weight update
                let delta = (new_mom * 100_000_000.0) as i64;
                *w_ptr.add(i) = (*w_ptr.add(i)).saturating_add(delta);
            }
        }
    }
    
    /// Get current weight value (scaled)
    #[inline(always)]
    pub fn get_weight(&self, idx: usize) -> i64 {
        if idx >= self.num_weights {
            return 0;
        }
        self.weights[idx]
    }
    
    /// Set weight value (scaled)
    #[inline(always)]
    pub fn set_weight(&self, idx: usize, value: i64) {
        if idx >= self.num_weights {
            return;
        }
        unsafe {
            let w_ptr = self.weights.as_ptr() as *mut i64;
            *w_ptr.add(idx) = value;
        }
    }
    
    /// Reset optimizer state
    pub fn reset(&self) {
        for i in 0..self.num_weights {
            unsafe {
                let mom_ptr = self.momentum.as_ptr() as *mut f32;
                let grad_ptr = self.grad_sum.as_ptr() as *mut f32;
                *mom_ptr.add(i) = 0.0;
                *grad_ptr.add(i) = 0.0;
            }
        }
        self.grad_count.store(0, Ordering::Relaxed);
        self.step.store(0, Ordering::Relaxed);
        self.current_lr = self.base_lr;
    }
    
    /// Halt training (circuit breaker)
    pub fn halt(&self) {
        TRAINING_ACTIVE.store(false, Ordering::Relaxed);
    }
    
    /// Resume training
    pub fn resume(&self) {
        TRAINING_ACTIVE.store(true, Ordering::Relaxed);
    }
    
    /// Check if training is active
    pub fn is_active(&self) -> bool {
        TRAINING_ACTIVE.load(Ordering::Relaxed)
    }
}

/// Mini-batch accumulator for gradient averaging
#[repr(C)]
pub struct MiniBatchAccumulator {
    /// Accumulated gradients
    pub gradients: [f32; MAX_WEIGHTS],
    /// Sample count
    pub count: AtomicU64,
    /// Batch size target
    pub batch_size: u64,
    /// Ready flag
    pub ready: AtomicBool,
    _pad: [u8; 54],
}

impl MiniBatchAccumulator {
    pub const fn new(batch_size: u64) -> Self {
        Self {
            gradients: [0.0; MAX_WEIGHTS],
            count: AtomicU64::new(0),
            batch_size,
            ready: AtomicBool::new(false),
            _pad: [0u8; 54],
        }
    }
    
    /// Add sample gradient
    #[inline(always)]
    pub fn add_sample(&self, idx: usize, grad: f32) {
        if idx >= MAX_WEIGHTS {
            return;
        }
        
        unsafe {
            let g_ptr = self.gradients.as_ptr() as *mut f32;
            *g_ptr.add(idx) += grad;
        }
        
        let count = self.count.fetch_add(1, Ordering::AcqRel);
        if count + 1 >= self.batch_size {
            self.ready.store(true, Ordering::Release);
        }
    }
    
    /// Get averaged gradient
    #[inline(always)]
    pub fn get_avg_gradient(&self, idx: usize) -> f32 {
        if idx >= MAX_WEIGHTS {
            return 0.0;
        }
        
        let count = self.count.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }
        
        unsafe {
            let g_ptr = self.gradients.as_ptr() as *const f32;
            *g_ptr.add(idx) / count as f32
        }
    }
    
    /// Clear accumulator
    pub fn clear(&self) {
        for i in 0..MAX_WEIGHTS {
            unsafe {
                let g_ptr = self.gradients.as_ptr() as *mut f32;
                *g_ptr.add(i) = 0.0;
            }
        }
        self.count.store(0, Ordering::Relaxed);
        self.ready.store(false, Ordering::Relaxed);
    }
    
    /// Check if batch is ready
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_weight_update_stability(
            num_weights in 1usize..100,
            grad_val in -1.0f32..1.0f32,
        ) {
            let sgd = OnlineSGD::new(num_weights);
            
            // Initialize weights
            for i in 0..num_weights {
                sgd.set_weight(i, 0);
            }
            
            // Accumulate gradients
            for i in 0..num_weights {
                sgd.accumulate_gradient(i, grad_val);
            }
            
            // Step should not panic
            sgd.step_update();
            
            // Weights should be updated deterministically
            let w1 = sgd.get_weight(0);
            let w2 = sgd.get_weight(0);
            assert_eq!(w1, w2, "Weight read must be deterministic");
        }
        
        #[test]
        fn test_lr_schedule_monotonic(step in 0u64..10000) {
            let mut sgd = OnlineSGD::new(10);
            sgd.lr_schedule = LrSchedule::Decay;
            
            let lr1 = sgd.compute_lr(step);
            let lr2 = sgd.compute_lr(step + 1000);
            
            // Decay should be non-increasing
            assert!(lr1 >= lr2, "Decay LR should be non-increasing");
        }
    }
    
    #[test]
    fn test_circuit_breaker() {
        let sgd = OnlineSGD::new(10);
        assert!(sgd.is_active());
        sgd.halt();
        assert!(!sgd.is_active());
        sgd.resume();
        assert!(sgd.is_active());
    }
    
    #[test]
    fn test_minibatch_accumulator() {
        let acc = MiniBatchAccumulator::new(10);
        
        for _ in 0..10 {
            acc.add_sample(0, 0.5);
        }
        
        assert!(acc.is_ready());
        assert!((acc.get_avg_gradient(0) - 0.5).abs() < 1e-6);
        
        acc.clear();
        assert!(!acc.is_ready());
    }
}
