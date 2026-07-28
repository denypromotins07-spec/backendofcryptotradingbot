//! Options Gamma-Exposure (GEX) and Delta-Hedging Signal Approximator
//! 
//! Estimates dealer gamma exposure and its impact on spot volatility.
//! Uses fixed-point arithmetic for deterministic calculations.

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

/// GEX state - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct GEXState {
    /// Net gamma exposure (Q32.32, in notional terms)
    pub net_gamma: i64,
    /// Call gamma (Q32.32)
    pub call_gamma: i64,
    /// Put gamma (Q32.32)
    pub put_gamma: i64,
    /// Zero-gamma level estimate (Q32.32 price)
    pub zero_gamma_level: i64,
    /// Dealer hedging signal (-128 to 127)
    pub hedge_signal: i8,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    _padding: [u8; 34],
}
const _: () = assert!(core::mem::size_of::<GEXState>() == 64);

impl GEXState {
    #[inline(always)] pub const fn new() -> Self {
        Self { net_gamma: 0, call_gamma: 0, put_gamma: 0, zero_gamma_level: 0, hedge_signal: 0, timestamp_tsc: 0, _padding: [0u8; 34] }
    }
}

/// Options GEX calculator
#[repr(C, align(64))]
pub struct OptionsGEX {
    /// Total call open interest (Q32.32)
    call_oi: i64,
    /// Total put open interest (Q32.32)
    put_oi: i64,
    /// ATM strike (Q32.32)
    atm_strike: i64,
    /// Current spot (Q32.32)
    spot: i64,
    pub state: GEXState,
    pub circuit_breaker: PaddedAtomicBool,
    _padding: [u8; 40],
}

impl OptionsGEX {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            call_oi: 0, put_oi: 0, atm_strike: 0, spot: 0,
            state: GEXState::new(),
            circuit_breaker: PaddedAtomicBool::new(false),
            _padding: [0u8; 40],
        }
    }
    
    /// Update with new options data
    #[inline(always)]
    pub fn update(&mut self, call_oi: i64, put_oi: i64, atm: i64, spot: i64, ts: u64) {
        if self.circuit_breaker.get() { return; }
        
        self.call_oi = call_oi;
        self.put_oi = put_oi;
        self.atm_strike = atm;
        self.spot = spot;
        
        // Simplified gamma calculation
        // Gamma ~ OI / (strike * sqrt(T) * sigma)
        // Assuming constant T and sigma for speed
        let scale = 1000i64; // Simplified scaling factor
        
        // Call gamma positive, put gamma negative (from dealer perspective)
        let call_g = (call_oi * scale) >> 16;
        let put_g = -(put_oi * scale) >> 16;
        
        self.state.call_gamma = call_g;
        self.state.put_gamma = put_g;
        self.state.net_gamma = call_g + put_g;
        
        // Estimate zero-gamma level
        // When spot moves such that call gamma = -put gamma
        let total_oi = call_oi + put_oi;
        self.state.zero_gamma_level = if total_oi > 0 {
            atm + ((call_oi - put_oi) * 100 / total_oi)
        } else { atm };
        
        // Generate hedging signal
        // Positive gamma = dealers sell into strength, buy weakness (stabilizing)
        // Negative gamma = dealers buy strength, sell weakness (destabilizing)
        self.state.hedge_signal = if self.state.net_gamma > 0 {
            // Long gamma: fade moves
            if spot > atm { -50i8 } else { 50i8 }
        } else {
            // Short gamma: chase moves
            if spot > atm { 50i8 } else { -50i8 }
        };
        
        self.state.timestamp_tsc = ts;
    }
    
    #[inline(always)] pub fn is_long_gamma(&self) -> bool { self.state.net_gamma > 0 }
    #[inline(always)] pub fn net_gamma_f64(&self) -> f64 { self.state.net_gamma as f64 / 4294967296.0 }
    #[inline(always)] pub fn halt(&self) { self.circuit_breaker.set(true); }
    #[inline(always)] pub fn resume(&self) { self.circuit_breaker.set(false); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_size() { assert_eq!(core::mem::size_of::<GEXState>(), 64); }
}
