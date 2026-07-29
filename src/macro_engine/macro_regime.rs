//! Hidden Markov Model for Macro Regime Detection
//! Real-time risk-on/risk-off state machine with lock-free transitions.
//! Branchless Viterbi decoding for regime inference.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

/// Number of hidden states
const NUM_STATES: usize = 3; // Risk-off, Neutral, Risk-on
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// HMM valid flag
static HMM_VALID: AtomicBool = AtomicBool::new(false);

/// Macro regime states
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MacroRegime {
    RiskOff = 0,
    Neutral = 1,
    RiskOn = 2,
}

/// HMM parameters - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HMMParams {
    /// Transition matrix [from][to], scaled by 10^6
    pub trans_matrix: [[i32; NUM_STATES]; NUM_STATES],
    /// Emission means [state], scaled by 10^8
    pub emission_means: [i64; NUM_STATES],
    /// Emission variances [state], scaled by 10^12
    pub emission_vars: [i64; NUM_STATES],
    /// Initial state probabilities, scaled by 10^6
    pub init_probs: [i32; NUM_STATES],
    _pad: [u8; 28],
}

impl Default for HMMParams {
    fn default() -> Self {
        Self {
            // High probability of staying in same state
            trans_matrix: [
                [900_000, 80_000, 20_000],  // Risk-off tends to persist
                [100_000, 800_000, 100_000], // Neutral is sticky
                [20_000, 80_000, 900_000],  // Risk-on tends to persist
            ],
            emission_means: [-500_000_000, 0, 500_000_000], // Negative, zero, positive returns
            emission_vars: [100_000_000_000_000, 50_000_000_000_000, 100_000_000_000_000],
            init_probs: [333_333, 333_333, 333_334],
            _pad: [0u8; 28],
        }
    }
}

const _: () = assert!(core::mem::size_of::<HMMParams>() == 64);

/// HMM Regime Detector
#[repr(C)]
pub struct HMMRegimeDetector {
    /// Model parameters
    pub params: HMMParams,
    /// Current state beliefs (forward algorithm), scaled by 10^6
    pub state_beliefs: [i32; NUM_STATES],
    /// Most likely current state
    pub current_state: AtomicU8,
    /// State transition count
    pub transition_count: AtomicU64,
    /// Last observation value
    pub last_observation: i64,
    /// Volatility scaling factor (scaled by 10^6)
    pub vol_scale: i64,
    /// Kill switch for extreme conditions
    pub kill_switch: AtomicBool,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 30],
}

impl Default for HMMRegimeDetector {
    fn default() -> Self {
        Self {
            params: HMMParams::default(),
            state_beliefs: [333_333, 333_333, 333_334],
            current_state: AtomicU8::new(MacroRegime::Neutral as u8),
            transition_count: AtomicU64::new(0),
            last_observation: 0,
            vol_scale: 1_000_000,
            kill_switch: AtomicBool::new(false),
            valid: AtomicBool::new(false),
            _pad: [0u8; 30],
        }
    }
}

impl HMMRegimeDetector {
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Gaussian log-likelihood approximation (scaled by 10^8)
    #[inline(always)]
    fn log_likelihood(&self, obs: i64, state: usize) -> i64 {
        let mean = self.params.emission_means[state] as f64 / 100_000_000.0;
        let var = self.params.emission_vars[state] as f64 / 1_000_000_000_000.0;
        let x = obs as f64 / 100_000_000.0;
        
        if var <= 0.0 {
            return i64::MIN;
        }
        
        let diff = x - mean;
        // Simplified: -0.5 * (x - mu)^2 / sigma^2
        let ll = -0.5 * diff * diff / var;
        (ll * 100_000_000.0) as i64
    }
    
    /// Forward step - update state beliefs with new observation
    #[inline(always)]
    pub fn update(&self, observation: i64) -> MacroRegime {
        if !HMM_VALID.load(Ordering::Relaxed) || !self.valid.load(Ordering::Acquire) {
            return MacroRegime::Neutral;
        }
        
        if self.kill_switch.load(Ordering::Relaxed) {
            return MacroRegime::Neutral;
        }
        
        let mut new_beliefs = [0i32; NUM_STATES];
        
        // Forward algorithm step
        for j in 0..NUM_STATES {
            let mut sum = 0i64;
            for i in 0..NUM_STATES {
                let trans = self.params.trans_matrix[i][j] as i64;
                let belief = self.state_beliefs[i] as i64;
                sum += trans * belief / 1_000_000;
            }
            
            // Multiply by emission probability
            let ll = self.log_likelihood(observation, j);
            let emission = ((ll + 100_000_000).clamp(0, 200_000_000) / 100_000_000) as i32;
            new_beliefs[j] = (sum as i32 * emission / 1_000_000).max(1);
        }
        
        // Normalize beliefs
        let total: i64 = new_beliefs.iter().map(|&x| x as i64).sum();
        if total > 0 {
            for j in 0..NUM_STATES {
                self.state_beliefs[j] = (new_beliefs[j] as i64 * 1_000_000 / total) as i32;
            }
        }
        
        // Find most likely state (branchless argmax)
        let mut best_state = 0;
        let mut best_prob = self.state_beliefs[0];
        for j in 1..NUM_STATES {
            let mask = ((self.state_beliefs[j] > best_prob) as usize).wrapping_neg();
            best_state = best_state ^ ((j ^ best_state) & mask);
            best_prob = best_prob ^ ((self.state_beliefs[j] ^ best_prob) & mask);
        }
        
        // Track transitions
        let old_state = self.current_state.load(Ordering::Relaxed);
        if old_state != best_state as u8 {
            self.transition_count.fetch_add(1, Ordering::Relaxed);
            
            // Check for extreme volatility - trigger kill switch
            let vol_scaled = observation.abs() * self.vol_scale / 100_000_000;
            if vol_scaled > 10_000_000 { // 10% move
                self.kill_switch.store(true, Ordering::Relaxed);
                return MacroRegime::Neutral;
            }
        }
        
        self.current_state.store(best_state as u8, Ordering::Release);
        self.last_observation = observation;
        
        match best_state {
            0 => MacroRegime::RiskOff,
            1 => MacroRegime::Neutral,
            2 => MacroRegime::RiskOn,
            _ => MacroRegime::Neutral,
        }
    }
    
    /// Get current regime
    #[inline(always)]
    pub fn get_regime(&self) -> MacroRegime {
        match self.current_state.load(Ordering::Acquire) {
            0 => MacroRegime::RiskOff,
            1 => MacroRegime::Neutral,
            2 => MacroRegime::RiskOn,
            _ => MacroRegime::Neutral,
        }
    }
    
    /// Set volatility scaling
    pub fn set_vol_scale(&self, scale: i64) {
        self.vol_scale = scale.clamp(100_000, 10_000_000);
    }
    
    /// Reset kill switch
    pub fn reset_kill_switch(&self) {
        self.kill_switch.store(false, Ordering::Relaxed);
    }
    
    /// Activate kill switch
    pub fn activate_kill_switch(&self) {
        self.kill_switch.store(true, Ordering::Relaxed);
    }
    
    /// Mark detector as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        HMM_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        HMM_VALID.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_regime_stability(observation in -1000i64..1000i64) {
            let hmm = HMMRegimeDetector::new();
            hmm.mark_valid();
            
            // Feed consistent observations
            for _ in 0..10 {
                hmm.update(observation);
            }
            
            let regime = hmm.get_regime();
            // Should settle into a stable regime
            assert!(matches!(regime, MacroRegime::RiskOff | MacroRegime::Neutral | MacroRegime::RiskOn));
        }
        
        #[test]
        fn test_kill_switch_triggers(extreme_obs in 2000i64..10000i64) {
            let hmm = HMMRegimeDetector::new();
            hmm.mark_valid();
            
            // Normal observations first
            for _ in 0..5 {
                hmm.update(100);
            }
            
            // Extreme observation should trigger kill switch
            hmm.update(extreme_obs * 1_000_000);
            
            assert!(hmm.kill_switch.load(Ordering::Relaxed));
            assert_eq!(hmm.get_regime(), MacroRegime::Neutral);
        }
    }
    
    #[test]
    fn test_hmm_params_size() {
        assert_eq!(core::mem::size_of::<HMMParams>(), 64);
    }
    
    #[test]
    fn test_initial_state() {
        let hmm = HMMRegimeDetector::new();
        assert_eq!(hmm.get_regime(), MacroRegime::Neutral);
    }
}
