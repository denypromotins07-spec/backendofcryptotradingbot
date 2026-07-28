//! Latency Probe - Nanosecond-precision tick-to-trade RTT measurement.
//! 
//! Injects probes into the trading pipeline to measure round-trip latency
//! from market data receipt to order execution confirmation.
//! 
//! Micro-optimizations:
//! - Lock-free probe injection and collection
//! - Pre-allocated probe slots to avoid allocation
//! - TSC-based timestamps for nanosecond precision

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};

/// Maximum concurrent probes
pub const MAX_PROBES: usize = 1024;

/// Probe state
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProbeState {
    Empty = 0,
    Injected = 1,
    MarketDataReceived = 2,
    StrategyProcessed = 3,
    OrderSent = 4,
    ExecutionReceived = 5,
    Completed = 6,
}

/// Single latency probe (cache-line aligned)
#[repr(C, align(64))]
pub struct LatencyProbe {
    /// Unique probe ID
    pub id: u64,
    /// Current state
    pub state: AtomicUsize,
    /// Timestamp: probe injection
    pub t_inject: AtomicU64,
    /// Timestamp: market data received
    pub t_md_recv: AtomicU64,
    /// Timestamp: strategy decision
    pub t_strategy: AtomicU64,
    /// Timestamp: order sent
    pub t_order_sent: AtomicU64,
    /// Timestamp: execution received
    pub t_exec_recv: AtomicU64,
    /// Symbol ID being probed
    pub symbol_id: u32,
    /// Venue ID
    pub venue_id: u8,
    _pad: [u8; 11],
}

impl LatencyProbe {
    pub const fn new() -> Self {
        Self {
            id: 0,
            state: AtomicUsize::new(ProbeState::Empty as usize),
            t_inject: AtomicU64::new(0),
            t_md_recv: AtomicU64::new(0),
            t_strategy: AtomicU64::new(0),
            t_order_sent: AtomicU64::new(0),
            t_exec_recv: AtomicU64::new(0),
            symbol_id: 0,
            venue_id: 0,
            _pad: [0; 11],
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
    
    /// Mark probe as received market data
    #[inline]
    pub fn mark_md_received(&self) {
        self.t_md_recv.store(Self::rdtsc(), Ordering::Relaxed);
        self.state.store(ProbeState::MarketDataReceived as usize, Ordering::Release);
    }
    
    /// Mark probe as processed by strategy
    #[inline]
    pub fn mark_strategy_done(&self) {
        self.t_strategy.store(Self::rdtsc(), Ordering::Relaxed);
        self.state.store(ProbeState::StrategyProcessed as usize, Ordering::Release);
    }
    
    /// Mark probe as order sent
    #[inline]
    pub fn mark_order_sent(&self) {
        self.t_order_sent.store(Self::rdtsc(), Ordering::Relaxed);
        self.state.store(ProbeState::OrderSent as usize, Ordering::Release);
    }
    
    /// Mark probe as execution received (complete)
    #[inline]
    pub fn mark_execution_received(&self) {
        self.t_exec_recv.store(Self::rdtsc(), Ordering::Relaxed);
        self.state.store(ProbeState::Completed as usize, Ordering::Release);
    }
    
    /// Calculate tick-to-trade latency in cycles
    #[inline]
    pub fn tick_to_trade(&self) -> u64 {
        let t_md = self.t_md_recv.load(Ordering::Relaxed);
        let t_order = self.t_order_sent.load(Ordering::Relaxed);
        if t_order > t_md {
            t_order.wrapping_sub(t_md)
        } else {
            0
        }
    }
    
    /// Calculate full round-trip latency
    #[inline]
    pub fn round_trip(&self) -> u64 {
        let t_start = self.t_inject.load(Ordering::Relaxed);
        let t_end = self.t_exec_recv.load(Ordering::Relaxed);
        if t_end > t_start {
            t_end.wrapping_sub(t_start)
        } else {
            0
        }
    }
}

/// Latency Probe Manager
pub struct LatencyProbeManager {
    /// Probe slots
    probes: [LatencyProbe; MAX_PROBES],
    /// Next probe ID
    next_id: AtomicU64,
    /// Active probe count
    active_count: AtomicUsize,
    /// Completed probe count
    completed_count: AtomicUsize,
    /// Is probing enabled?
    enabled: AtomicBool,
    /// Running average of tick-to-trade (exponential moving average)
    avg_tick_to_trade: AtomicU64,
}

// SAFETY: All mutable state is protected by atomics
unsafe impl Send for LatencyProbeManager {}
unsafe impl Sync for LatencyProbeManager {}

impl LatencyProbeManager {
    /// Create a new probe manager
    pub const fn new() -> Self {
        const INIT_PROBE: LatencyProbe = LatencyProbe::new();
        Self {
            probes: [INIT_PROBE; MAX_PROBES],
            next_id: AtomicU64::new(1),
            active_count: AtomicUsize::new(0),
            completed_count: AtomicUsize::new(0),
            enabled: AtomicBool::new(false),
            avg_tick_to_trade: AtomicU64::new(0),
        }
    }
    
    /// Enable probing
    #[inline]
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }
    
    /// Disable probing
    #[inline]
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }
    
    /// Check if probing is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
    
    /// Inject a new probe, returns pointer to probe or None if full
    #[inline]
    pub fn inject(&self, symbol_id: u32, venue_id: u8) -> Option<&LatencyProbe> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        
        // Find an empty slot (linear search - could optimize with bitmap)
        for probe in &self.probes {
            if probe.state.load(Ordering::Relaxed) == ProbeState::Empty as usize {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                probe.id = id;
                probe.symbol_id = symbol_id;
                probe.venue_id = venue_id;
                probe.t_inject.store(LatencyProbe::rdtsc(), Ordering::Relaxed);
                probe.state.store(ProbeState::Injected as usize, Ordering::Release);
                self.active_count.fetch_add(1, Ordering::Relaxed);
                return Some(probe);
            }
        }
        
        None // No available slots
    }
    
    /// Find a probe by ID
    #[inline]
    pub fn find_by_id(&self, id: u64) -> Option<&LatencyProbe> {
        for probe in &self.probes {
            if probe.id == id && probe.state.load(Ordering::Relaxed) != ProbeState::Empty as usize {
                return Some(probe);
            }
        }
        None
    }
    
    /// Complete a probe and update statistics
    #[inline]
    pub fn complete_probe(&self, id: u64) -> Option<u64> {
        if let Some(probe) = self.find_by_id(id) {
            if probe.state.load(Ordering::Relaxed) == ProbeState::Completed as usize {
                let t2t = probe.tick_to_trade();
                
                // Update running average (EMA with alpha = 0.1)
                let current_avg = self.avg_tick_to_trade.load(Ordering::Relaxed);
                let new_avg = current_avg
                    .wrapping_mul(9)
                    .wrapping_add(t2t)
                    .wrapping_div(10);
                self.avg_tick_to_trade.store(new_avg, Ordering::Relaxed);
                
                // Reset probe slot
                probe.state.store(ProbeState::Empty as usize, Ordering::Release);
                self.active_count.fetch_sub(1, Ordering::Relaxed);
                self.completed_count.fetch_add(1, Ordering::Relaxed);
                
                return Some(t2t);
            }
        }
        None
    }
    
    /// Get average tick-to-trade latency
    #[inline]
    pub fn avg_tick_to_trade(&self) -> u64 {
        self.avg_tick_to_trade.load(Ordering::Relaxed)
    }
    
    /// Get active probe count
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }
    
    /// Get completed probe count
    #[inline]
    pub fn completed_count(&self) -> usize {
        self.completed_count.load(Ordering::Relaxed)
    }
}

impl Default for LatencyProbeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_probe_size() {
        assert_eq!(core::mem::size_of::<LatencyProbe>(), 64);
    }
    
    #[test]
    fn test_manager_creation() {
        let mgr = LatencyProbeManager::new();
        assert!(!mgr.is_enabled());
        assert_eq!(mgr.active_count(), 0);
    }
}
