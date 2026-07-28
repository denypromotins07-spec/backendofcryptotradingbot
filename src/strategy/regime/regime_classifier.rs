//! Hidden Markov Model (HMM) for Volatility Regime Switching
//! 
//! Implements a 2-state HMM for detecting volatility regime transitions.
//! Uses fixed-point arithmetic and pre-computed transition matrices.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

#[repr(C, align(64))]
pub struct PaddedAtomicBool { value: AtomicBool, _padding: [u8; 63] }
impl PaddedAtomicBool {
    #[inline(always)] pub const fn new(v: bool) -> Self { Self { value: AtomicBool::new(v), _padding: [0u8; 63] } }
    #[inline(always)] pub fn set(&self, v: bool) { self.value.store(v, Ordering::Relaxed); }
    #[inline(always)] pub fn get(&self) -> bool { self.value.load(Ordering::Relaxed) }
}

#[repr(C, align(64))]
pub struct PaddedAtomicU64 { value: AtomicU64, _padding: [u8; 56] }
impl PaddedAtomicU64 {
    #[inline(always)] pub const fn new(v: u64) -> Self { Self { value: AtomicU64::new(v), _padding: [0u8; 56] } }
    #[inline(always)] pub fn load(&self) -> u64 { self.value.load(Ordering::Relaxed) }
    #[inline(always)] pub fn store(&self, v: u64) { self.value.store(v, Ordering::Relaxed); }
}

/// Volatility regime: 0=low, 1=high
#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
pub enum Regime { Low = 0, High = 1 }

/// HMM state - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct HMMState {
    /// Current regime probability P(high) in Q16.16
    pub prob_high: i32,
    /// Current regime classification
    pub current_regime: u8,
    /// Transition count
    pub transitions: u32,
    /// Last update timestamp (TSC)
    pub last_update_tsc: u64,
    /// Log-likelihood (Q32.32)
    pub log_likelihood: i64,
    _padding: [u8; 36],
}
const _: () = assert!(core::mem::size_of::<HMMState>() == 64);

impl HMMState {
    #[inline(always)] pub const fn new() -> Self {
        Self { prob_high: 32768, current_regime: 0, transitions: 0, last_update_tsc: 0, log_likelihood: 0, _padding: [0u8; 36] }
    }
}

/// 2-state HMM for volatility regime detection
#[repr(C, align(64))]
pub struct VolatilityHMM {
    /// Transition matrix [low->low, low->high, high->low, high->high] in Q16.16
    trans_matrix: [i32; 4],
    /// Emission means for each regime (volatility levels) in Q32.32
    emission_mean: [i64; 2],
    /// Emission std devs in Q32.32
    emission_std: [i64; 2],
    pub state: HMMState,
    /// Update counter
    pub update_count: PaddedAtomicU64,
    pub circuit_breaker: PaddedAtomicBool,
    _padding: [u8; 32],
}

impl VolatilityHMM {
    #[inline(always)]
    pub const fn new(
        trans_low_low: f64, trans_low_high: f64,
        trans_high_low: f64, trans_high_high: f64,
        emit_low: f64, emit_high: f64,
        std_low: f64, std_high: f64,
    ) -> Self {
        Self {
            trans_matrix: [
                (trans_low_low * 65536.0) as i32,
                (trans_low_high * 65536.0) as i32,
                (trans_high_low * 65536.0) as i32,
                (trans_high_high * 65536.0) as i32,
            ],
            emission_mean: [(emit_low * 4294967296.0) as i64, (emit_high * 4294967296.0) as i64],
            emission_std: [(std_low * 4294967296.0) as i64, (std_high * 4294967296.0) as i64],
            state: HMMState::new(),
            update_count: PaddedAtomicU64::new(0),
            circuit_breaker: PaddedAtomicBool::new(false),
            _padding: [0u8; 32],
        }
    }
    
    /// Update HMM with new volatility observation
    #[inline(always)]
    pub fn update(&mut self, observed_vol: i64, ts: u64) {
        if self.circuit_breaker.get() { return; }
        
        let p_high = self.state.prob_high;
        let p_low = 65536 - p_high;
        
        // Calculate emission probabilities (simplified Gaussian)
        let e_low = self.emission_prob(observed_vol, 0);
        let e_high = self.emission_prob(observed_vol, 1);
        
        // Forward step: calculate new probabilities
        // P(low) = P(low|low)*e_low*P(low) + P(low|high)*e_low*P(high)
        let tm = &self.trans_matrix;
        let new_low = ((tm[0] as i64 * e_low as i64 * p_low as i64) >> 32)
                    + ((tm[2] as i64 * e_low as i64 * p_high as i64) >> 32);
        let new_high = ((tm[1] as i64 * e_high as i64 * p_low as i64) >> 32)
                     + ((tm[3] as i64 * e_high as i64 * p_high as i64) >> 32);
        
        // Normalize
        let total = new_low + new_high;
        if total > 0 {
            self.state.prob_high = ((new_high << 16) / total) as i32;
        }
        
        // Determine regime
        self.state.current_regime = if self.state.prob_high > 32768 { 1 } else { 0 };
        
        // Track transitions
        let prev_regime = if p_high > 32768 { 1u32 } else { 0 };
        if prev_regime != self.state.current_regime as u32 {
            self.state.transitions += 1;
        }
        
        self.state.last_update_tsc = ts;
        self.update_count.value.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Simplified emission probability (inverse distance)
    #[inline(always)]
    fn emission_prob(&self, obs: i64, regime: usize) -> i64 {
        let mean = self.emission_mean[regime];
        let std = self.emission_std[regime].max(1);
        let diff = (obs - mean).abs();
        // Inverse: higher prob when closer to mean
        (1_000_000_000 / (diff / std + 1)).min(65535) as i64
    }
    
    #[inline(always)] pub fn is_high_vol(&self) -> bool { self.state.current_regime == 1 }
    #[inline(always)] pub fn prob_high_vol(&self) -> f64 { self.state.prob_high as f64 / 65536.0 }
    #[inline(always)] pub fn halt(&self) { self.circuit_breaker.set(true); }
    #[inline(always)] pub fn resume(&self) { self.circuit_breaker.set(false); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_size() { assert_eq!(core::mem::size_of::<HMMState>(), 64); }
}
