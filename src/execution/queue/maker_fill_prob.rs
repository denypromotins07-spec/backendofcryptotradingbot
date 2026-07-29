//! Probability model for maker fills based on order book depletion.
//! 
//! Uses a Bayesian approach to estimate fill probability:
//! - Prior: historical fill rate at this price level
//! - Likelihood: current book depletion rate, trade flow imbalance
//! - Posterior: updated fill probability
//! 
//! All calculations use fixed-point arithmetic and SIMD where applicable.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Number of bins for book depth histogram
const DEPTH_BINS: usize = 32;

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Book state snapshot - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BookState {
    pub bid_depth: [i64; DEPTH_BINS],
    pub ask_depth: [i64; DEPTH_BINS],
    pub best_bid: i64,
    pub best_ask: i64,
    pub spread_bps: i64,          // Spread in basis points (fixed-point)
    pub imbalance: i64,           // Order flow imbalance (fixed-point)
    pub timestamp_cycles: u64,
    _padding: [u8; 16],           // Pad to 64 bytes
}

impl Default for BookState {
    fn default() -> Self {
        Self {
            bid_depth: [0; DEPTH_BINS],
            ask_depth: [0; DEPTH_BINS],
            best_bid: 0,
            best_ask: 0,
            spread_bps: 0,
            imbalance: 0,
            timestamp_cycles: 0,
            _padding: [0; 16],
        }
    }
}

/// Historical fill statistics - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FillHistory {
    pub total_orders: u64,
    pub filled_orders: u64,
    pub partial_fills: u64,
    pub avg_fill_time_us: u64,
    pub fill_rate_at_level: [i64; DEPTH_BINS], // Per-level fill rates
    _padding: [u8; 24],            // Pad to 64 bytes
}

impl Default for FillHistory {
    fn default() -> Self {
        Self {
            total_orders: 0,
            filled_orders: 0,
            partial_fills: 0,
            avg_fill_time_us: 0,
            fill_rate_at_level: [0; DEPTH_BINS],
            _padding: [0; 24],
        }
    }
}

/// Lock-free maker fill probability estimator
#[repr(C)]
pub struct MakerFillProbability {
    book_state: BookState,
    history: FillHistory,
    
    // Depletion tracking
    depletion_sum: AtomicU64,      // Running sum of depletion events
    depletion_count: AtomicU64,    // Count of depletion samples
    
    // Probability state
    prior_prob: AtomicU64,         // Prior fill probability (fixed-point)
    posterior_prob: AtomicU64,     // Current posterior (fixed-point)
    
    // Model parameters
    alpha: AtomicU64,              // Beta distribution alpha parameter
    beta: AtomicU64,               // Beta distribution beta parameter
    
    // Flags
    is_active: AtomicBool,
    regime_shift: AtomicBool,      // Set if market regime changed
    
    _padding: [u8; 32],            // Pad to cache line
}

impl MakerFillProbability {
    /// Create new maker fill probability estimator
    pub const fn new() -> Self {
        Self {
            book_state: BookState {
                bid_depth: [0; DEPTH_BINS],
                ask_depth: [0; DEPTH_BINS],
                best_bid: 0,
                best_ask: 0,
                spread_bps: 0,
                imbalance: 0,
                timestamp_cycles: 0,
                _padding: [0; 16],
            },
            history: FillHistory {
                total_orders: 0,
                filled_orders: 0,
                partial_fills: 0,
                avg_fill_time_us: 0,
                fill_rate_at_level: [0; DEPTH_BINS],
                _padding: [0; 24],
            },
            depletion_sum: AtomicU64::new(0),
            depletion_count: AtomicU64::new(0),
            prior_prob: AtomicU64::new(FIXED_SCALE as u64 / 2), // 50% prior
            posterior_prob: AtomicU64::new(FIXED_SCALE as u64 / 2),
            alpha: AtomicU64::new(1),
            beta: AtomicU64::new(1),
            is_active: AtomicBool::new(true),
            regime_shift: AtomicBool::new(false),
            _padding: [0; 32],
        }
    }
    
    /// Update book state
    #[inline(always)]
    pub fn update_book_state(&mut self, state: &BookState) {
        let cycles = unsafe { x86_64::_rdtsc() };
        
        // SIMD-like batch copy for depth arrays
        unsafe {
            core::ptr::copy_nonoverlapping(
                state.bid_depth.as_ptr(),
                self.book_state.bid_depth.as_mut_ptr(),
                DEPTH_BINS,
            );
            core::ptr::copy_nonoverlapping(
                state.ask_depth.as_ptr(),
                self.book_state.ask_depth.as_mut_ptr(),
                DEPTH_BINS,
            );
        }
        
        self.book_state.best_bid = state.best_bid;
        self.book_state.best_ask = state.best_ask;
        self.book_state.spread_bps = state.spread_bps;
        self.book_state.imbalance = state.imbalance;
        self.book_state.timestamp_cycles = cycles;
    }
    
    /// Record a depletion event (trade eating into the book)
    #[inline(always)]
    pub fn record_depletion(&self, volume: i64, level: usize) {
        if level >= DEPTH_BINS {
            return;
        }
        
        // Update running statistics
        let current_sum = self.depletion_sum.load(Ordering::Relaxed);
        self.depletion_sum.store(current_sum.saturating_add(volume as u64), Ordering::Relaxed);
        
        let current_count = self.depletion_count.load(Ordering::Relaxed);
        self.depletion_count.store(current_count + 1, Ordering::Relaxed);
        
        // Update per-level fill rate using exponential moving average
        let old_rate = self.history.fill_rate_at_level[level];
        let alpha = 0.1 * FIXED_SCALE as f64; // EMA smoothing factor
        let contribution = ((volume as f64 / 1000.0) * FIXED_SCALE as f64) as i64;
        let new_rate = ((old_rate as f64 * (FIXED_SCALE as f64 - alpha)) + (contribution as f64 * alpha)) as i64 / FIXED_SCALE;
        
        unsafe {
            *self.history.fill_rate_at_level.get_unchecked_mut(level) = new_rate.clamp(0, FIXED_SCALE);
        }
    }
    
    /// Calculate fill probability using Bayesian update
    #[inline(always)]
    pub fn calculate_probability(&self, price_level: usize, order_size: i64) -> i64 {
        if price_level >= DEPTH_BINS || !self.is_active.load(Ordering::Acquire) {
            return 0;
        }
        
        // Get book depth at our level
        let depth_at_level = unsafe { *self.book_state.bid_depth.get_unchecked(price_level) };
        
        // Branchless: if no depth, probability is 0
        let has_depth = (depth_at_level > 0) as i64;
        
        // Calculate base probability from depth ratio
        let depth_ratio = if depth_at_level > 0 {
            ((order_size.min(depth_at_level) as i128 * FIXED_SCALE as i128) / depth_at_level as i128) as i64
        } else {
            0
        };
        
        // Adjust for imbalance (branchless)
        let imbalance_factor = FIXED_SCALE + (self.book_state.imbalance / 100);
        let adjusted_prob = ((depth_ratio as i128 * imbalance_factor as i128) / FIXED_SCALE as i128) as i64;
        
        // Combine with historical rate at this level
        let hist_rate = unsafe { *self.history.fill_rate_at_level.get_unchecked(price_level) };
        
        // Weighted average: 60% current state, 40% history
        let combined = (adjusted_prob as i128 * 6 + hist_rate as i128 * 4) / 10;
        
        // Apply Bayesian posterior adjustment
        let prior = self.prior_prob.load(Ordering::Acquire) as i128;
        let posterior = self.posterior_prob.load(Ordering::Acquire) as i128;
        
        // Final probability: weighted blend of model and posterior
        let final_prob = ((combined * 7 + posterior * 3) / 10) as i64;
        
        // Clamp and apply depth gate
        (final_prob.clamp(0, FIXED_SCALE)) & -(has_depth)
    }
    
    /// Update Bayesian posterior based on observed fills
    #[inline(always)]
    pub fn update_posterior(&self, was_filled: bool) {
        let mut alpha = self.alpha.load(Ordering::Acquire);
        let mut beta = self.beta.load(Ordering::Acquire);
        
        // Branchless update
        let fill_flag = was_filled as u64;
        alpha = alpha + fill_flag;
        beta = beta + (1 - fill_flag);
        
        self.alpha.store(alpha, Ordering::Release);
        self.beta.store(beta, Ordering::Release);
        
        // Calculate new posterior mean: alpha / (alpha + beta)
        let total = alpha + beta;
        let prob = if total > 0 {
            ((alpha as u128 * FIXED_SCALE as u128) / total as u128) as u64
        } else {
            FIXED_SCALE as u64 / 2
        };
        
        self.posterior_prob.store(prob, Ordering::Release);
        
        // Update history
        self.history.total_orders += 1;
        self.history.filled_orders += fill_flag;
    }
    
    /// Detect regime shift (sudden change in fill dynamics)
    #[inline(always)]
    pub fn detect_regime_shift(&self, threshold: i64) -> bool {
        let prior = self.prior_prob.load(Ordering::Acquire) as i64;
        let posterior = self.posterior_prob.load(Ordering::Acquire) as i64;
        
        let diff = (posterior - prior).abs();
        let shifted = (diff > threshold) as u8;
        
        // Branchless flag setting
        self.regime_shift.store(shifted != 0, Ordering::Release);
        
        shifted != 0
    }
    
    /// Reset after regime shift
    #[inline(always)]
    pub fn reset_regime(&mut self) {
        self.prior_prob.store(self.posterior_prob.load(Ordering::Acquire), Ordering::Release);
        self.alpha.store(1, Ordering::Release);
        self.beta.store(1, Ordering::Release);
        self.regime_shift.store(false, Ordering::Release);
    }
    
    /// Get current posterior probability
    #[inline(always)]
    pub fn get_posterior(&self) -> i64 {
        self.posterior_prob.load(Ordering::Acquire) as i64
    }
    
    /// Get average depletion rate
    #[inline(always)]
    pub fn avg_depletion_rate(&self) -> i64 {
        let sum = self.depletion_sum.load(Ordering::Acquire);
        let count = self.depletion_count.load(Ordering::Acquire);
        
        if count == 0 {
            return 0;
        }
        
        (sum / count) as i64
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<BookState>() == 64);
        assert!(core::mem::size_of::<FillHistory>() == 64);
        assert!(core::mem::size_of::<MakerFillProbability>() % 64 == 0);
    }
    
    #[test]
    fn test_probability_calculation() {
        let mut estimator = MakerFillProbability::new();
        
        // Set up book state
        let mut state = BookState::default();
        state.bid_depth[0] = 1000;
        state.bid_depth[1] = 500;
        state.imbalance = 10_000_000; // Positive imbalance
        
        estimator.update_book_state(&state);
        
        // Calculate probability for order at level 0
        let prob = estimator.calculate_probability(0, 100);
        assert!(prob > 0);
        assert!(prob <= FIXED_SCALE);
    }
    
    #[test]
    fn test_bayesian_update() {
        let estimator = MakerFillProbability::new();
        
        // Initial posterior should be 50%
        assert_eq!(estimator.get_posterior(), FIXED_SCALE / 2);
        
        // Simulate several fills
        for _ in 0..5 {
            estimator.update_posterior(true);
        }
        for _ in 0..2 {
            estimator.update_posterior(false);
        }
        
        // Posterior should now favor fills (> 50%)
        assert!(estimator.get_posterior() > FIXED_SCALE / 2);
    }
}
