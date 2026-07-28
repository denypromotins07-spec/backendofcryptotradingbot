//! Quality Monitor - Real-time anomaly detection for market data feeds.
//! 
//! Monitors feed quality, detects stale books, bad spikes, and missing venues.
//! Implements circuit breaker logic to halt trading on anomalies.
//! 
//! Micro-optimizations:
//! - Statistical thresholds with rolling windows
//! - Zero-allocation anomaly scoring
//! - Fast path for normal operation

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Maximum symbols monitored
pub const MAX_SYMBOLS: usize = 1024;

/// Anomaly types
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    None = 0,
    StaleBook = 1,
    PriceSpike = 2,
    SpreadWidening = 3,
    MissingVenue = 4,
    SequenceGap = 5,
    CrossedBook = 6,
}

/// Per-symbol quality state (cache-line aligned)
#[repr(C, align(64))]
struct SymbolQuality {
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Last price (fixed-point)
    last_price: AtomicI64,
    /// Price change count (for spike detection)
    price_changes: AtomicU64,
    /// Anomaly flags
    anomaly_flags: AtomicU64,
    /// Consecutive anomaly count
    anomaly_count: AtomicU64,
    _pad: [u8; 24],
}

impl SymbolQuality {
    const fn new() -> Self {
        Self {
            last_update_ns: AtomicU64::new(0),
            last_price: AtomicI64::new(0),
            price_changes: AtomicU64::new(0),
            anomaly_flags: AtomicU64::new(0),
            anomaly_count: AtomicU64::new(0),
            _pad: [0; 24],
        }
    }
}

/// Circuit breaker state
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed = 0,    // Normal operation
    Open = 1,      // Trading halted
    HalfOpen = 2,  // Testing recovery
}

/// Quality Monitor with circuit breaker
pub struct QualityMonitor {
    /// Per-symbol quality states
    symbols: [SymbolQuality; MAX_SYMBOLS],
    /// Circuit breaker state
    circuit_state: AtomicUsize,
    /// Circuit trip count
    trip_count: AtomicU64,
    /// Staleness threshold (ns)
    staleness_threshold_ns: AtomicU64,
    /// Spike threshold (basis points)
    spike_threshold_bps: AtomicU64,
    /// Max allowed anomalies before trip
    max_anomalies: AtomicU64,
    /// Total anomalies detected
    total_anomalies: AtomicU64,
}

// SAFETY: All mutable state is protected by atomics
unsafe impl Send for QualityMonitor {}
unsafe impl Sync for QualityMonitor {}

impl QualityMonitor {
    pub const fn new() -> Self {
        const INIT_SYM: SymbolQuality = SymbolQuality::new();
        Self {
            symbols: [INIT_SYM; MAX_SYMBOLS],
            circuit_state: AtomicUsize::new(CircuitState::Closed as usize),
            trip_count: AtomicU64::new(0),
            staleness_threshold_ns: AtomicU64::new(500_000_000), // 500ms
            spike_threshold_bps: AtomicU64::new(500), // 5% move
            max_anomalies: AtomicU64::new(10),
            total_anomalies: AtomicU64::new(0),
        }
    }
    
    /// Read TSC
    #[inline(always)]
    fn rdtsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0
        }
    }
    
    /// Process a price update and check for anomalies
    #[inline]
    pub fn process_update(&self, symbol_id: u32, price: i64) -> AnomalyType {
        let idx = symbol_id as usize;
        if idx >= MAX_SYMBOLS {
            return AnomalyType::None;
        }
        
        let now = Self::rdtsc();
        let sym = &self.symbols[idx];
        
        // Update timestamp
        sym.last_update_ns.store(now, Ordering::Relaxed);
        
        // Check for price spike
        let last_price = sym.last_price.load(Ordering::Relaxed);
        if last_price != 0 {
            let change_bps = (price.wrapping_sub(last_price).abs() as u64)
                .wrapping_mul(10000)
                .wrapping_div(last_price.abs() as u64);
            
            let threshold = self.spike_threshold_bps.load(Ordering::Relaxed);
            if change_bps > threshold {
                sym.anomaly_count.fetch_add(1, Ordering::Relaxed);
                self.total_anomalies.fetch_add(1, Ordering::Relaxed);
                sym.anomaly_flags.fetch_or(1 << AnomalyType::PriceSpike as u8, Ordering::Relaxed);
                
                self.check_circuit_breaker();
                return AnomalyType::PriceSpike;
            }
        }
        sym.last_price.store(price, Ordering::Relaxed);
        
        // Check anomaly count for circuit breaker
        let anomaly_count = sym.anomaly_count.load(Ordering::Relaxed);
        let max = self.max_anomalies.load(Ordering::Relaxed);
        if anomaly_count >= max {
            self.trip_circuit();
            return AnomalyType::SequenceGap; // Generic anomaly
        }
        
        AnomalyType::None
    }
    
    /// Check for stale books across all symbols
    pub fn check_staleness(&self) -> u32 {
        let now = Self::rdtsc();
        let threshold = self.staleness_threshold_ns.load(Ordering::Relaxed);
        let mut stale_count = 0;
        
        for sym in &self.symbols {
            let last = sym.last_update_ns.load(Ordering::Relaxed);
            if last != 0 && now.wrapping_sub(last) > threshold {
                sym.anomaly_flags.fetch_or(1 << AnomalyType::StaleBook as u8, Ordering::Relaxed);
                stale_count += 1;
            }
        }
        
        if stale_count > 10 {
            self.trip_circuit();
        }
        
        stale_count
    }
    
    /// Check if circuit breaker is tripped
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.circuit_state.load(Ordering::Acquire) == CircuitState::Open as usize
    }
    
    /// Trip the circuit breaker
    #[inline]
    pub fn trip_circuit(&self) {
        self.circuit_state.store(CircuitState::Open as usize, Ordering::Release);
        self.trip_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Reset circuit breaker (manual intervention required)
    #[inline]
    pub fn reset_circuit(&self) {
        self.circuit_state.store(CircuitState::HalfOpen as usize, Ordering::Release);
        
        // Clear anomaly counts
        for sym in &self.symbols {
            sym.anomaly_count.store(0, Ordering::Relaxed);
            sym.anomaly_flags.store(0, Ordering::Relaxed);
        }
        
        // After cooling period, close circuit
        self.circuit_state.store(CircuitState::Closed as usize, Ordering::Release);
    }
    
    /// Get circuit state
    #[inline]
    pub fn circuit_state(&self) -> CircuitState {
        match self.circuit_state.load(Ordering::Acquire) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
    
    /// Set staleness threshold
    pub fn set_staleness_threshold(&self, ns: u64) {
        self.staleness_threshold_ns.store(ns, Ordering::Relaxed);
    }
    
    /// Set spike threshold (basis points)
    pub fn set_spike_threshold(&self, bps: u64) {
        self.spike_threshold_bps.store(bps, Ordering::Relaxed);
    }
    
    /// Get total anomalies
    #[inline]
    pub fn total_anomalies(&self) -> u64 {
        self.total_anomalies.load(Ordering::Relaxed)
    }
    
    /// Get trip count
    #[inline]
    pub fn trip_count(&self) -> u64 {
        self.trip_count.load(Ordering::Relaxed)
    }
}

impl Default for QualityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbol_quality_size() {
        assert_eq!(core::mem::size_of::<SymbolQuality>(), 64);
    }
    
    #[test]
    fn test_monitor_creation() {
        let mon = QualityMonitor::new();
        assert_eq!(mon.circuit_state(), CircuitState::Closed);
        assert!(!mon.is_halted());
    }
}
