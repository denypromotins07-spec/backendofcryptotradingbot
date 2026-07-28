//! Real-time Liquidation Cascade and Stop-Hunt Detector
//! 
//! Detects liquidation cascades and stop-hunt patterns using order book dynamics.
//! Uses lock-free state and fixed-point arithmetic for microsecond detection.

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

/// Liquidation cascade state - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct LiquidationState {
    /// Cascade intensity (0-255)
    pub intensity: u8,
    /// Direction: 0=none, 1=long liq (down), 2=short liq (up)
    pub direction: u8,
    /// Estimated cascade size (Q32.32)
    pub est_size_i64: i64,
    /// Price impact rate (Q16.16 bps per unit)
    pub impact_rate: i32,
    /// Detection timestamp (TSC)
    pub detected_tsc: u64,
    /// Cumulative liquidations (Q32.32)
    pub total_liq: i64,
    _padding: [u8; 30],
}
const _: () = assert!(core::mem::size_of::<LiquidationState>() == 64);

impl LiquidationState {
    #[inline(always)] pub const fn new() -> Self {
        Self { intensity: 0, direction: 0, est_size_i64: 0, impact_rate: 0, detected_tsc: 0, total_liq: 0, _padding: [0u8; 30] }
    }
}

/// Stop-hunt detector
#[repr(C, align(64))]
pub struct LiquidationDetector<const N: usize> {
    /// Rolling liquidation volumes
    liq_buffer: [i64; N],
    head: u64,
    pub state: LiquidationState,
    /// Intensity threshold
    threshold: u8,
    /// Lookback count
    lookback: u32,
    pub circuit_breaker: PaddedAtomicBool,
    _padding: [u8: 40],
}

impl<const N: usize> LiquidationDetector<N> {
    #[inline(always)]
    pub const fn new(threshold: u8, lookback: u32) -> Self {
        Self {
            liq_buffer: [0; N], head: 0, state: LiquidationState::new(),
            threshold, lookback, circuit_breaker: PaddedAtomicBool::new(false),
            _padding: [0u8; 40],
        }
    }
    
    #[inline(always)]
    pub fn update(&mut self, liq_volume: i64, price_delta: i64, ts: u64) {
        if self.circuit_breaker.get() { return; }
        
        // Add to rolling buffer
        let idx = (self.head % N as u64) as usize;
        self.liq_buffer[idx] = liq_volume;
        self.head += 1;
        
        // Calculate rolling sum
        let mut sum: i128 = 0;
        let count = self.lookback.min(N as u32) as usize;
        for i in 0..count {
            let iidx = ((self.head - 1 - i as u64) % N as u64) as usize;
            sum += self.liq_buffer[iidx] as i128;
        }
        
        // Update total liquidations
        self.state.total_liq = self.state.total_liq.saturating_add(liq_volume);
        
        // Calculate intensity based on rolling volume
        let avg = (sum / count as i128) as i64;
        let intensity = if avg > 0 && liq_volume > avg * 3 {
            ((liq_volume * 256 / avg).min(255)) as u8
        } else { 0 };
        
        // Determine direction from price delta
        let direction = if price_delta < 0 { 1u8 } else if price_delta > 0 { 2u8 } else { 0u8 };
        
        // Detect cascade
        if intensity >= self.threshold {
            self.state.intensity = intensity;
            self.state.direction = direction;
            self.state.est_size_i64 = sum as i64;
            self.state.impact_rate = ((price_delta.abs() << 16) / avg.max(1)) as i32;
            self.state.detected_tsc = ts;
        }
    }
    
    #[inline(always)] pub fn halt(&self) { self.circuit_breaker.set(true); }
    #[inline(always)] pub fn resume(&self) { self.circuit_breaker.set(false); }
    #[inline(always)] pub fn is_cascade(&self) -> bool { self.state.intensity >= self.threshold; }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_size() { assert_eq!(core::mem::size_of::<LiquidationState>(), 64); }
}
