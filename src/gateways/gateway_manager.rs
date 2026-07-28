//! Gateway Manager - Unified trait-based abstraction for multi-venue routing.
//! 
//! This module provides a zero-allocation, lock-free gateway abstraction layer
//! that normalizes exchange-specific protocols into a unified internal format.
//! Designed for microsecond-level routing decisions with circuit breaker support.

#![allow(dead_code)]

use core::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use crate::transport::ring_buffer::RingBuffer;

/// Maximum number of supported venues (power of 2 for fast indexing)
pub const MAX_VENUES: usize = 16;

/// Cache-line padding to prevent false sharing between gateway slots
const CACHE_LINE_PAD: [u8; 56] = [0; 56];

/// Unified market data event that all gateways must produce
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct MarketEvent {
    pub venue_id: u8,
    pub symbol_id: u16,
    pub event_type: u8, // 0=Trade, 1=L2Update, 2=Quote
    pub timestamp_ns: u64, // rdtsc capture
    pub price: i64, // Fixed point: price * 1e8
    pub quantity: i64, // Fixed point: quantity * 1e8
    pub flags: u32, // Exchange-specific flags normalized
    _pad: [u8; 8], // Ensure 64-byte alignment
}

impl Default for MarketEvent {
    fn default() -> Self {
        Self {
            venue_id: 0,
            symbol_id: 0,
            event_type: 0,
            timestamp_ns: 0,
            price: 0,
            quantity: 0,
            flags: 0,
            _pad: [0; 8],
        }
    }
}

/// Gateway state machine for handling exchange lifecycle events
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GatewayState {
    Disconnected = 0,
    Connecting = 1,
    Connected = 2,
    Syncing = 3,
    Ready = 4,
    Halted = 5, // Exchange halt/auction
    Error = 6,
}

/// Per-venue gateway slot with cache-line isolation
#[repr(C, align(64))]
struct GatewaySlot {
    state: AtomicUsize, // GatewayState as usize
    active: AtomicBool,
    sequence: AtomicUsize,
    latency_score: AtomicUsize, // Inverse latency for routing weights
    _pad: [u8; 32], // Pad to 64 bytes
}

impl GatewaySlot {
    const fn new() -> Self {
        Self {
            state: AtomicUsize::new(GatewayState::Disconnected as usize),
            active: AtomicBool::new(false),
            sequence: AtomicUsize::new(0),
            latency_score: AtomicUsize::new(0),
            _pad: [0; 32],
        }
    }
}

/// Gateway trait that all exchange adapters must implement
/// Zero-allocation requirement: all methods must work with borrowed buffers
pub trait Gateway: Send + Sync {
    /// Returns the unique venue ID (0..MAX_VENUES)
    fn venue_id(&self) -> u8;
    
    /// Returns the venue name for logging
    fn venue_name(&self) -> &'static str;
    
    /// Initialize the gateway with output ring buffer
    /// Must be called before start()
    fn init(&mut self, output: &RingBuffer<MarketEvent>);
    
    /// Start the gateway connection (non-blocking)
    /// Returns immediately, actual connection is async
    fn start(&mut self) -> Result<(), GatewayError>;
    
    /// Stop the gateway gracefully
    fn stop(&mut self);
    
    /// Process incoming data buffer (zero-copy)
    /// Returns number of events written to ring buffer
    fn process_buffer(&mut self, data: &[u8]) -> usize;
    
    /// Get current gateway state
    fn state(&self) -> GatewayState;
    
    /// Get current latency score (higher = better)
    fn latency_score(&self) -> u32;
    
    /// Update latency score based on recent measurements
    fn update_latency_score(&self, score: u32);
}

/// Gateway error types for fast pattern matching
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GatewayError {
    AlreadyConnected = 0,
    NotConnected = 1,
    BufferFull = 2,
    InvalidMessage = 3,
    SequenceGap = 4,
    Halted = 5,
    Unknown = 255,
}

/// Gateway Manager - Central coordinator for all exchange gateways
/// Uses array-based storage to avoid heap allocations
pub struct GatewayManager {
    slots: [GatewaySlot; MAX_VENUES],
    gateways: [Option<&'static mut dyn Gateway>; MAX_VENUES],
    active_count: AtomicUsize,
    circuit_breaker: AtomicBool,
    total_events: AtomicUsize,
}

// SAFETY: GatewayManager is safe to share between threads
// as all mutable access is through atomic operations or requires &mut self
unsafe impl Send for GatewayManager {}
unsafe impl Sync for GatewayManager {}

impl GatewayManager {
    /// Create a new gateway manager with all slots initialized
    pub const fn new() -> Self {
        const INIT_SLOT: GatewaySlot = GatewaySlot::new();
        Self {
            slots: [INIT_SLOT; MAX_VENUES],
            gateways: [None; MAX_VENUES],
            active_count: AtomicUsize::new(0),
            circuit_breaker: AtomicBool::new(false),
            total_events: AtomicUsize::new(0),
        }
    }
    
    /// Register a gateway for a specific venue
    /// Panics if venue_id is out of range or already registered
    pub fn register(&mut self, gateway: &'static mut dyn Gateway) {
        let vid = gateway.venue_id() as usize;
        assert!(vid < MAX_VENUES, "Venue ID {} exceeds maximum {}", vid, MAX_VENUES);
        
        let slot = &self.slots[vid];
        assert!(!slot.active.load(Ordering::Relaxed), "Venue {} already registered", vid);
        
        slot.active.store(true, Ordering::Release);
        slot.state.store(GatewayState::Connecting as usize, Ordering::Release);
        
        // SAFETY: We ensure the gateway lives long enough via 'static lifetime
        self.gateways[vid] = Some(gateway);
        self.active_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Start all registered gateways
    pub fn start_all(&mut self) -> Result<(), GatewayError> {
        if self.circuit_breaker.load(Ordering::Acquire) {
            return Err(GatewayError::Halted);
        }
        
        for (i, gw_opt) in self.gateways.iter_mut().enumerate() {
            if let Some(gw) = gw_opt {
                match gw.start() {
                    Ok(()) => {
                        self.slots[i].state.store(GatewayState::Connected as usize, Ordering::Release);
                    }
                    Err(e) => {
                        self.slots[i].state.store(GatewayState::Error as usize, Ordering::Release);
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Stop all gateways gracefully
    pub fn stop_all(&mut self) {
        for gw_opt in self.gateways.iter_mut() {
            if let Some(gw) = gw_opt.as_mut() {
                gw.stop();
            }
        }
        self.circuit_breaker.store(true, Ordering::Release);
    }
    
    /// Process data for a specific venue (called from network layer)
    /// Returns number of events published to ring buffer
    #[inline(always)]
    pub fn process_venue_data(&mut self, venue_id: u8, data: &[u8]) -> usize {
        if self.circuit_breaker.load(Ordering::Acquire) {
            return 0;
        }
        
        let vid = venue_id as usize;
        if vid >= MAX_VENUES || !self.slots[vid].active.load(Ordering::Acquire) {
            return 0;
        }
        
        if let Some(gw) = self.gateways[vid].as_mut() {
            let count = gw.process_buffer(data);
            self.total_events.fetch_add(count, Ordering::Relaxed);
            self.slots[vid].sequence.fetch_add(count, Ordering::Relaxed);
            count
        } else {
            0
        }
    }
    
    /// Get the number of active gateways
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }
    
    /// Get total events processed across all gateways
    #[inline]
    pub fn total_events(&self) -> usize {
        self.total_events.load(Ordering::Relaxed)
    }
    
    /// Check if circuit breaker is tripped
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.circuit_breaker.load(Ordering::Acquire)
    }
    
    /// Trip the circuit breaker to halt all trading
    #[inline]
    pub fn trip_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Release);
    }
    
    /// Reset the circuit breaker (requires manual confirmation)
    #[inline]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Release);
    }
    
    /// Get venue reliability score for routing decisions
    /// Higher score = more reliable (lower latency, fewer gaps)
    #[inline]
    pub fn venue_reliability(&self, venue_id: u8) -> u32 {
        let vid = venue_id as usize;
        if vid >= MAX_VENUES {
            return 0;
        }
        self.slots[vid].latency_score.load(Ordering::Relaxed) as u32
    }
    
    /// Update venue reliability score
    #[inline]
    pub fn update_venue_reliability(&self, venue_id: u8, score: u32) {
        let vid = venue_id as usize;
        if vid < MAX_VENUES {
            self.slots[vid].latency_score.store(score as usize, Ordering::Relaxed);
        }
    }
}

impl Default for GatewayManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_event_size() {
        // Ensure MarketEvent fits in single cache line
        assert_eq!(core::mem::size_of::<MarketEvent>(), 64);
        assert_eq!(core::mem::align_of::<MarketEvent>(), 64);
    }
    
    #[test]
    fn test_gateway_slot_size() {
        assert_eq!(core::mem::size_of::<GatewaySlot>(), 64);
    }
    
    #[test]
    fn test_gateway_manager_creation() {
        let manager = GatewayManager::new();
        assert_eq!(manager.active_count(), 0);
        assert!(!manager.is_halted());
    }
}
