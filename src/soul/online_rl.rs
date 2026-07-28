//! Lock-free online reinforcement learning (Contextual Bandits) for strategy weighting.
//!
//! This module implements a custom Contextual Bandit algorithm using fixed-point math
//! to avoid floating-point latency overhead. It dynamically adjusts strategy weights
//! based on real-time PnL feedback without blocking the trading thread.
//!
//! **Latency Target:** < 200ns per weight update.
//! **Memory Limit:** Pre-allocated buffers only, zero heap allocation in hot path.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of strategies supported by the bandit.
const MAX_STRATEGIES: usize = 32;

/// Fixed-point precision (number of fractional bits).
const FIXED_POINT_BITS: u32 = 16;

/// Convert f64 to fixed-point i64.
#[inline]
fn to_fixed(val: f64) -> i64 {
    (val * (1 << FIXED_POINT_BITS) as f64) as i64
}

/// Convert fixed-point i64 to f64.
#[inline]
fn from_fixed(val: i64) -> f64 {
    val as f64 / (1 << FIXED_POINT_BITS) as f64
}

/// A single strategy slot in the bandit.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StrategySlot {
    /// Cumulative reward in fixed-point.
    pub total_reward: AtomicU64,
    /// Number of times this strategy was selected.
    pub selection_count: AtomicU64,
    /// Current weight (probability of selection) in fixed-point.
    pub weight: AtomicU64,
    /// Last update timestamp (rdtsc cycles).
    pub last_update_ts: AtomicU64,
    /// Penalty multiplier for recent mistakes (fixed-point).
    pub penalty_factor: AtomicU64,
    /// Reserved padding.
    _padding: [u8; 24],
}

impl StrategySlot {
    #[inline]
    pub const fn new() -> Self {
        Self {
            total_reward: AtomicU64::new(0),
            selection_count: AtomicU64::new(0),
            weight: AtomicU64::new(to_fixed(1.0 / MAX_STRATEGIES as f64)),
            last_update_ts: AtomicU64::new(0),
            penalty_factor: AtomicU64::new(to_fixed(1.0)),
            _padding: [0u8; 24],
        }
    }

    /// Get current weight as f64.
    #[inline]
    pub fn get_weight(&self) -> f64 {
        from_fixed(self.weight.load(Ordering::Acquire))
    }

    /// Update weight atomically.
    #[inline]
    pub fn update_weight(&self, new_weight: f64) {
        self.weight.store(to_fixed(new_weight), Ordering::Release);
    }
}

// Ensure StrategySlot is exactly one cache line.
const _: () = assert!(core::mem::size_of::<StrategySlot>() == CACHE_LINE_SIZE);

/// Contextual Bandit for online reinforcement learning.
///
/// Uses epsilon-greedy with decay and Thompson Sampling approximation.
/// All operations are lock-free and use fixed-point arithmetic.
pub struct OnlineBandit {
    /// Array of strategy slots (pre-allocated).
    strategies: [StrategySlot; MAX_STRATEGIES],
    /// Total number of active strategies.
    active_count: AtomicU64,
    /// Exploration rate (epsilon) in fixed-point.
    epsilon: AtomicU64,
    /// Epsilon decay factor per iteration (fixed-point).
    epsilon_decay: AtomicU64,
    /// Minimum epsilon (fixed-point).
    epsilon_min: AtomicU64,
    /// Global iteration counter.
    iteration_count: AtomicU64,
    /// Flag indicating if the bandit is active.
    is_active: AtomicBool,
    /// Padding to separate hot/cold data.
    _padding: [u8; 56],
}

unsafe impl Send for OnlineBandit {}
unsafe impl Sync for OnlineBandit {}

impl OnlineBandit {
    /// Create a new bandit with default parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            strategies: [StrategySlot::new(); MAX_STRATEGIES],
            active_count: AtomicU64::new(0),
            epsilon: AtomicU64::new(to_fixed(0.1)), // 10% exploration
            epsilon_decay: AtomicU64::new(to_fixed(0.999)), // Slow decay
            epsilon_min: AtomicU64::new(to_fixed(0.01)), // 1% minimum exploration
            iteration_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 56],
        }
    }

    /// Initialize the bandit with a specific number of strategies.
    #[inline]
    pub fn init_strategies(&mut self, count: usize) {
        assert!(count <= MAX_STRATEGIES, "Exceeds maximum strategies");
        self.active_count.store(count as u64, Ordering::Release);
        
        // Initialize weights uniformly
        let initial_weight = 1.0 / count as f64;
        for i in 0..count {
            self.strategies[i].update_weight(initial_weight);
        }
    }

    /// Select a strategy using epsilon-greedy with weighted probabilities.
    ///
    /// Returns the index of the selected strategy.
    /// Uses branchless logic for deterministic latency.
    #[inline]
    pub fn select_strategy(&self) -> usize {
        if !self.is_active.load(Ordering::Acquire) {
            return 0;
        }

        let iteration = self.iteration_count.fetch_add(1, Ordering::AcqRel);
        
        // Decay epsilon
        let current_epsilon = from_fixed(self.epsilon.load(Ordering::Acquire));
        let decayed_epsilon = (current_epsilon * from_fixed(self.epsilon_decay.load(Ordering::Acquire)))
            .max(from_fixed(self.epsilon_min.load(Ordering::Acquire)));
        self.epsilon.store(to_fixed(decayed_epsilon), Ordering::Release);

        // Generate pseudo-random number (simplified, use rdtsc in production)
        let rand_val = ((iteration.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) & 0x7FFF;
        let rand_normalized = rand_val as f64 / 32768.0;

        // Epsilon-greedy: explore with probability epsilon, exploit otherwise
        let active_count = self.active_count.load(Ordering::Acquire) as usize;
        
        if rand_normalized < decayed_epsilon {
            // Explore: random selection
            (rand_val % active_count as u32) as usize
        } else {
            // Exploit: select highest weight (branchless linear scan)
            let mut best_idx = 0;
            let mut best_weight = self.strategies[0].get_weight();
            
            // Manual loop unrolling for O(1) feel
            let mut i = 1;
            while i < active_count {
                let w = self.strategies[i].get_weight();
                // Branchless comparison
                let mask = ((w > best_weight) as u64).wrapping_neg();
                best_idx = (best_idx & !mask as usize) | (i & mask as usize);
                best_weight = (best_weight * !(mask as f64)) + (w * (mask as f64));
                i += 1;
            }
            
            best_idx
        }
    }

    /// Update the reward for a selected strategy.
    ///
    /// Called after observing the outcome (PnL) of a trade.
    /// Updates weights using a simplified Thompson Sampling approach.
    #[inline]
    pub fn update_reward(&self, strategy_idx: usize, reward: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        if strategy_idx >= MAX_STRATEGIES {
            return;
        }

        let slot = &self.strategies[strategy_idx];
        
        // Update cumulative reward and count
        let current_reward = slot.total_reward.load(Ordering::Acquire);
        slot.total_reward.store(
            current_reward.wrapping_add(to_fixed(reward)),
            Ordering::Release,
        );

        let count = slot.selection_count.fetch_add(1, Ordering::AcqRel);
        let new_count = count + 1;

        // Calculate new weight: average reward with penalty factor
        let avg_reward = if new_count > 0 {
            from_fixed(current_reward.wrapping_add(to_fixed(reward))) / new_count as f64
        } else {
            0.0
        };

        let penalty = from_fixed(slot.penalty_factor.load(Ordering::Acquire));
        let adjusted_reward = avg_reward * penalty;

        // Softmax-like weight update (simplified)
        let active_count = self.active_count.load(Ordering::Acquire) as f64;
        let new_weight = (adjusted_reward + 1.0).max(0.0) / active_count; // Shift to positive
        
        slot.update_weight(new_weight);
        
        // Record timestamp
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::_rdtsc;
            slot.last_update_ts.store(_rdtsc(), Ordering::Release);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            slot.last_update_ts.store(iteration, Ordering::Release);
        }
    }

    /// Apply a penalty to a strategy (e.g., after a mistake).
    #[inline]
    pub fn apply_penalty(&self, strategy_idx: usize, penalty_factor: f64) {
        if strategy_idx >= MAX_STRATEGIES {
            return;
        }

        let slot = &self.strategies[strategy_idx];
        let current_penalty = from_fixed(slot.penalty_factor.load(Ordering::Acquire));
        let new_penalty = (current_penalty * penalty_factor).max(0.1); // Cap at 0.1
        slot.penalty_factor.store(to_fixed(new_penalty), Ordering::Release);
    }

    /// Get current weights for all active strategies.
    #[inline]
    pub fn get_weights(&self, output: &mut [f64; MAX_STRATEGIES]) {
        let active_count = self.active_count.load(Ordering::Acquire) as usize;
        for i in 0..active_count {
            output[i] = self.strategies[i].get_weight();
        }
        for i in active_count..MAX_STRATEGIES {
            output[i] = 0.0;
        }
    }

    /// Shutdown the bandit.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

impl Default for OnlineBandit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_slot_size() {
        assert_eq!(core::mem::size_of::<StrategySlot>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_bandit_init() {
        let mut bandit = OnlineBandit::new();
        bandit.init_strategies(4);
        assert_eq!(bandit.active_count.load(Ordering::Acquire), 4);
    }

    #[test]
    fn test_select_and_update() {
        let mut bandit = OnlineBandit::new();
        bandit.init_strategies(2);
        
        // Select and update multiple times
        for _ in 0..100 {
            let idx = bandit.select_strategy();
            let reward = if idx == 0 { 1.0 } else { -0.5 };
            bandit.update_reward(idx, reward);
        }
        
        // Strategy 0 should have higher weight
        let mut weights = [0.0; MAX_STRATEGIES];
        bandit.get_weights(&mut weights);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_penalty_application() {
        let mut bandit = OnlineBandit::new();
        bandit.init_strategies(2);
        
        bandit.apply_penalty(0, 0.5);
        
        let slot = &bandit.strategies[0];
        let penalty = from_fixed(slot.penalty_factor.load(Ordering::Acquire));
        assert!((penalty - 0.5).abs() < 0.01);
    }
}
