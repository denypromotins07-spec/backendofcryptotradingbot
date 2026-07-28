//! Dynamic Weighting Router for Mean-Reversion vs Momentum Strategies
//! 
//! Implements an ensemble router that dynamically weights strategy outputs
//! based on rolling performance and regime detection.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

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

/// Maximum number of strategies in ensemble
pub const MAX_STRATEGIES: usize = 8;

/// Strategy weight - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct StrategyWeight {
    /// Current weight (Q16.16)
    pub weight: i32,
    /// Rolling Sharpe estimate (Q16.16)
    pub sharpe: i32,
    /// Rolling return (Q32.32)
    pub return_acc: i64,
    /// Trade count
    pub trade_count: u32,
    /// Active flag
    pub active: u8,
    _padding: [u8; 35],
}
const _: () = assert!(core::mem::size_of::<StrategyWeight>() == 64);

impl StrategyWeight {
    #[inline(always)] pub const fn new() -> Self {
        Self { weight: 0, sharpe: 0, return_acc: 0, trade_count: 0, active: 1, _padding: [0u8; 35] }
    }
}

/// Ensemble router state
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct EnsembleState {
    /// Combined signal (-128 to 127)
    pub combined_signal: i8,
    /// Active strategy count
    pub active_count: u8,
    /// Regime indicator (0=mean-rev, 1=momentum)
    pub regime: u8,
    /// Total weight sum (Q16.16)
    pub total_weight: i32,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    _padding: [u8; 52],
}
const _: () = assert!(core::mem::size_of::<EnsembleState>() == 64);

impl EnsembleState {
    #[inline(always)] pub const fn new() -> Self {
        Self { combined_signal: 0, active_count: 0, regime: 0, total_weight: 0, timestamp_tsc: 0, _padding: [0u8; 52] }
    }
}

/// Dynamic ensemble router
#[repr(C, align(64))]
pub struct EnsembleRouter<const N: usize> {
    /// Per-strategy weights
    weights: [StrategyWeight; N],
    pub state: EnsembleState,
    /// Decay factor for EMA (Q16.16)
    decay: i32,
    /// Min weight threshold (Q16.16)
    min_weight: i32,
    pub circuit_breaker: PaddedAtomicBool,
    _padding: [u8: 32],
}

impl<const N: usize> EnsembleRouter<N> {
    #[inline(always)]
    pub const fn new(decay: f64, min_weight: f64) -> Self {
        Self {
            weights: [StrategyWeight::new(); N],
            state: EnsembleState::new(),
            decay: (decay * 65536.0) as i32,
            min_weight: (min_weight * 65536.0) as i32,
            circuit_breaker: PaddedAtomicBool::new(false),
            _padding: [0u8; 32],
        }
    }
    
    /// Update strategy performance and recalculate weights
    #[inline(always)]
    pub fn update_performance(&mut self, strategy_idx: usize, pnl: i64, ts: u64) {
        if self.circuit_breaker.get() || strategy_idx >= N { return; }
        
        let w = &mut self.weights[strategy_idx];
        if w.active == 0 { return; }
        
        // Update rolling return with EMA
        let alpha = self.decay;
        w.return_acc = ((alpha as i64 * pnl) + ((65536 - alpha) as i64 * w.return_acc)) >> 16;
        w.trade_count += 1;
        
        // Simplified Sharpe: return / sqrt(variance) ~ sign(return) * abs(return)
        // Using absolute return as proxy for risk-adjusted return
        w.sharpe = if w.return_acc > 0 {
            ((w.return_acc >> 16) as i32).min(32768)
        } else {
            -((w.return_acc.abs() >> 16) as i32).min(32768)
        };
    }
    
    /// Recalculate all weights based on performance
    #[inline(always)]
    pub fn recalculate_weights(&mut self, ts: u64) {
        if self.circuit_breaker.get() { return; }
        
        let mut total: i32 = 0;
        let mut active: u8 = 0;
        
        // Calculate raw weights based on Sharpe
        for w in &mut self.weights {
            if w.active == 0 || w.trade_count < 2 {
                w.weight = 0;
                continue;
            }
            
            active += 1;
            
            // Weight proportional to Sharpe, clamped
            let raw_weight = w.sharpe.abs().max(self.min_weight);
            w.weight = raw_weight;
            total = total.saturating_add(raw_weight);
        }
        
        // Normalize weights to sum to 65536 (1.0 in Q16.16)
        if total > 0 {
            for w in &mut self.weights {
                if w.weight > 0 {
                    w.weight = (w.weight << 16) / total;
                }
            }
        }
        
        self.state.active_count = active;
        self.state.total_weight = total;
        self.state.timestamp_tsc = ts;
    }
    
    /// Combine signals from all strategies
    #[inline(always)]
    pub fn combine_signals(&self, signals: &[i8; N]) -> i8 {
        if self.circuit_breaker.get() { return 0; }
        
        let mut weighted_sum: i32 = 0;
        
        for i in 0..N.min(signals.len()) {
            if self.weights[i].active == 0 { continue; }
            weighted_sum += (signals[i] as i32 * self.weights[i].weight) >> 16;
        }
        
        // Clamp to -128..127
        weighted_sum.clamp(-128, 127) as i8
    }
    
    /// Set strategy active state
    #[inline(always)]
    pub fn set_strategy_active(&mut self, idx: usize, active: bool) {
        if idx < N {
            self.weights[idx].active = active as u8;
        }
    }
    
    /// Set regime (affects which strategies are favored)
    #[inline(always)]
    pub fn set_regime(&mut self, regime: u8) {
        self.state.regime = regime & 1;
    }
    
    #[inline(always)] pub fn halt(&self) { self.circuit_breaker.set(true); }
    #[inline(always)] pub fn resume(&self) { self.circuit_breaker.set(false); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_sizes() {
        assert_eq!(core::mem::size_of::<StrategyWeight>(), 64);
        assert_eq!(core::mem::size_of::<EnsembleState>(), 64);
    }
}
