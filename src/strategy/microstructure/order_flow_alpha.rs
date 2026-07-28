//! CVD Divergence and Order-Flow Imbalance Alpha Extractor
//! 
//! Implements cumulative volume delta (CVD) analysis and order flow imbalance
//! detection for alpha generation. Uses lock-free circular buffers and SIMD.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Cache-line padded structures
#[repr(C, align(64))]
pub struct PaddedAtomicBool {
    value: AtomicBool,
    _padding: [u8; 63],
}

impl PaddedAtomicBool {
    #[inline(always)]
    pub const fn new(val: bool) -> Self {
        Self { value: AtomicBool::new(val), _padding: [0u8; 63] }
    }
    #[inline(always)]
    pub fn set(&self, val: bool) { self.value.store(val, Ordering::Relaxed); }
    #[inline(always)]
    pub fn get(&self) -> bool { self.value.load(Ordering::Relaxed) }
}

#[repr(C, align(64))]
pub struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 56],
}

impl PaddedAtomicU64 {
    #[inline(always)]
    pub const fn new(val: u64) -> Self {
        Self { value: AtomicU64::new(val), _padding: [0u8; 56] }
    }
    #[inline(always)]
    pub fn load(&self) -> u64 { self.value.load(Ordering::Relaxed) }
    #[inline(always)]
    pub fn store(&self, val: u64) { self.value.store(val, Ordering::Relaxed); }
}

/// Order flow imbalance state - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct OrderFlowState {
    /// Cumulative volume delta (Q32.32)
    pub cvd_i64: i64,
    /// Order flow imbalance ratio (Q16.16, -1 to 1)
    pub ofi_ratio: i32,
    /// Price-CVD divergence (Q16.16)
    pub divergence: i32,
    /// Aggressive buyer volume (Q32.32)
    pub buy_volume: i64,
    /// Aggressive seller volume (Q32.32)
    pub sell_volume: i64,
    /// Signal strength (-128 to 127)
    pub signal: i8,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    _padding: [u8; 35],
}

const _: () = assert!(core::mem::size_of::<OrderFlowState>() == 64);

impl OrderFlowState {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            cvd_i64: 0, ofi_ratio: 0, divergence: 0,
            buy_volume: 0, sell_volume: 0, signal: 0,
            timestamp_tsc: 0, _padding: [0u8; 35],
        }
    }
}

/// Circular buffer for rolling CVD calculation
#[repr(C, align(64))]
pub struct CVDBuffer<const N: usize> {
    data: [i64; N],
    head: u64,
    sum: i128,
    count: u64,
    _padding: [u8; 24],
}

impl<const N: usize> CVDBuffer<N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self { data: [0; N], head: 0, sum: 0, count: 0, _padding: [0u8; 24] }
    }
    
    #[inline(always)]
    pub fn push(&mut self, value: i64) {
        let idx = (self.head % N as u64) as usize;
        let old = self.data[idx];
        self.sum = self.sum - old as i128 + value as i128;
        self.data[idx] = value;
        self.head += 1;
        if self.count < N as u64 { self.count += 1; }
    }
    
    #[inline(always)]
    pub fn mean(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        (self.sum / self.count as i128) as f64 / 4294967296.0
    }
}

/// Order flow alpha extractor
#[repr(C, align(64))]
pub struct OrderFlowAlpha<const N: usize> {
    cvd_buffer: CVDBuffer<N>,
    pub state: OrderFlowState,
    /// Lookback window for divergence
    lookback: u32,
    /// Signal threshold
    threshold: i32,
    pub circuit_breaker: PaddedAtomicBool,
    _padding: [u8; 48],
}

impl<const N: usize> OrderFlowAlpha<N> {
    #[inline(always)]
    pub const fn new(lookback: u32, threshold: f64) -> Self {
        Self {
            cvd_buffer: CVDBuffer::new(),
            state: OrderFlowState::new(),
            lookback,
            threshold: (threshold * 65536.0) as i32,
            circuit_breaker: PaddedAtomicBool::new(false),
            _padding: [0u8; 48],
        }
    }
    
    #[inline(always)]
    pub fn update(&mut self, is_buy: bool, volume: i64, price_change: i64, ts: u64) {
        if self.circuit_breaker.get() { return; }
        
        // Update volumes
        if is_buy {
            self.state.buy_volume = self.state.buy_volume.saturating_add(volume);
        } else {
            self.state.sell_volume = self.state.sell_volume.saturating_add(volume);
        }
        
        // Update CVD (positive for buys, negative for sells)
        let cvd_delta = if is_buy { volume } else { -volume };
        self.state.cvd_i64 = self.state.cvd_i64.saturating_add(cvd_delta);
        self.cvd_buffer.push(self.state.cvd_i64);
        
        // Calculate OFI ratio
        let total = self.state.buy_volume + self.state.sell_volume;
        self.state.ofi_ratio = if total > 0 {
            (((self.state.buy_volume - self.state.sell_volume) << 16) / total) as i32
        } else { 0 };
        
        // Calculate price-CVD divergence
        let cvd_mean = self.cvd_buffer.mean();
        let cvd_z = if cvd_mean != 0.0 {
            ((price_change as f64 / 4294967296.0) / cvd_mean * 65536.0) as i32
        } else { 0 };
        self.state.divergence = cvd_z;
        
        // Generate signal based on divergence and OFI
        let div_signal = (self.state.divergence.abs() > self.threshold) as i8;
        let ofi_sign = if self.state.ofi_ratio > 0 { 1i8 } else { -1 };
        self.state.signal = div_signal * ofi_sign;
        self.state.timestamp_tsc = ts;
    }
    
    #[inline(always)]
    pub fn halt(&self) { self.circuit_breaker.set(true); }
    #[inline(always)]
    pub fn resume(&self) { self.circuit_breaker.set(false); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_size() { assert_eq!(core::mem::size_of::<OrderFlowState>(), 64); }
}
