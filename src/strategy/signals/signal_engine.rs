//! Multi-Asset Signal Engine & Statistical Models
//! 
//! This module implements the core signal generation infrastructure for BTC/ETH/SOL/USDT pairs.
//! All calculations use fixed-point arithmetic or fast-math floats to avoid FPU non-determinism.
//! Memory is pre-allocated at startup; zero heap allocations occur in the hot path.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache-line size for padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Padded atomic flag for lock-free state transitions
#[repr(C, align(64))]
pub struct PaddedAtomicBool {
    pub value: AtomicBool,
    _padding: [u8; 63],
}

impl PaddedAtomicBool {
    #[inline(always)]
    pub const fn new(val: bool) -> Self {
        Self {
            value: AtomicBool::new(val),
            _padding: [0u8; 63],
        }
    }
    
    #[inline(always)]
    pub fn set(&self, val: bool) {
        self.value.store(val, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn get(&self) -> bool {
        self.value.load(Ordering::Relaxed)
    }
}

/// Signal types supported by the engine
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    Long = 0,
    Short = 1,
    Neutral = 2,
    CloseLong = 3,
    CloseShort = 4,
}

/// Multi-asset signal structure - fits in single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct MarketSignal {
    /// Asset identifier (fixed-point encoding)
    pub asset_id: u64,
    /// Signal type
    pub signal_type: u8,
    /// Signal strength (0-255, fixed-point)
    pub strength: u8,
    /// Timestamp in TSC cycles
    pub timestamp_ns: u64,
    /// Z-score of signal (fixed-point Q16.16)
    pub z_score_i32: i32,
    /// Padding to reach 64 bytes
    _padding: [u8; 39],
}

// Compile-time assertion: MarketSignal must be exactly 64 bytes
const _: () = assert!(core::mem::size_of::<MarketSignal>() == 64);

impl MarketSignal {
    #[inline(always)]
    pub const fn new(asset_id: u64, signal_type: SignalType, strength: u8, timestamp_ns: u64, z_score: f64) -> Self {
        // Convert f64 z-score to Q16.16 fixed-point
        let z_score_i32 = (z_score * 65536.0) as i32;
        Self {
            asset_id,
            signal_type: signal_type as u8,
            strength,
            timestamp_ns,
            z_score_i32,
            _padding: [0u8; 39],
        }
    }
    
    #[inline(always)]
    pub fn z_score(&self) -> f64 {
        self.z_score_i32 as f64 / 65536.0
    }
}

/// Signal engine dispatcher for multi-asset routing
#[repr(C, align(64))]
pub struct SignalEngine {
    /// Active flag for circuit breaker
    pub active: PaddedAtomicBool,
    /// Number of assets being tracked
    pub asset_count: AtomicU64,
    /// Signal buffer index (atomic for lock-free updates)
    pub write_index: AtomicU64,
    /// Read index for consumers
    pub read_index: AtomicU64,
    /// Pre-allocated signal ring buffer (capacity defined elsewhere)
    /// In production, this points to a memory-mapped region
    pub signal_ptr: *mut MarketSignal,
    /// Shadow mode flag - logs theoretical fills without execution
    pub shadow_mode: PaddedAtomicBool,
    _padding: [u8; 32],
}

// Safety: SignalEngine is only accessed from a single thread in hot path
unsafe impl Send for SignalEngine {}
unsafe impl Sync for SignalEngine {}

impl SignalEngine {
    /// Create a new signal engine with pre-allocated buffer
    #[inline(always)]
    pub fn new(signal_ptr: *mut MarketSignal, asset_count: u64) -> Self {
        Self {
            active: PaddedAtomicBool::new(true),
            asset_count: AtomicU64::new(asset_count),
            write_index: AtomicU64::new(0),
            read_index: AtomicU64::new(0),
            signal_ptr,
            shadow_mode: PaddedAtomicBool::new(false),
            _padding: [0u8; 32],
        }
    }
    
    /// Dispatch a signal to the ring buffer (lock-free, single-producer)
    #[inline(always)]
    pub fn dispatch_signal(&self, signal: MarketSignal) -> bool {
        if !self.active.get() {
            return false;
        }
        
        // Branchless capacity check would go here in production
        let idx = self.write_index.fetch_add(1, Ordering::AcqRel);
        
        unsafe {
            // Direct memory write - zero-copy
            core::ptr::write(self.signal_ptr.add((idx % 1024) as usize), signal);
        }
        
        true
    }
    
    /// Enable/disable shadow mode for theoretical fill logging
    #[inline(always)]
    pub fn set_shadow_mode(&self, enabled: bool) {
        self.shadow_mode.set(enabled);
    }
    
    /// Circuit breaker - disable all signal generation
    #[inline(always)]
    pub fn trigger_circuit_breaker(&self) {
        self.active.set(false);
    }
    
    /// Reset circuit breaker
    #[inline(always)]
    pub fn reset_circuit_breaker(&self) {
        self.active.set(true);
    }
}

/// Asset identifiers for BTC/ETH/SOL/USDT
pub mod assets {
    pub const BTC: u64 = 0x4254430000000000; // "BTC" in hex
    pub const ETH: u64 = 0x4554480000000000; // "ETH" in hex
    pub const SOL: u64 = 0x534F4C0000000000; // "SOL" in hex
    pub const USDT: u64 = 0x5553445400000000; // "USDT" in hex
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_size() {
        assert_eq!(core::mem::size_of::<MarketSignal>(), 64);
    }
    
    #[test]
    fn test_z_score_conversion() {
        let signal = MarketSignal::new(assets::BTC, SignalType::Long, 128, 1000, 2.5);
        assert!((signal.z_score() - 2.5).abs() < 0.0001);
    }
}
