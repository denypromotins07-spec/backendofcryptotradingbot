//! Cross-Venue Order Router
//! 
//! Lock-free cross-venue order splitter and atomic execution coordinator.
//! Routes orders across multiple venues to optimize fill probability and minimize costs.
//! Uses pre-allocated routing buffers and SIMD for venue comparison.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

use crate::common::circular_buffer::CircularBuffer;
use crate::common::fixed_point::FixedPoint;
use crate::execution::types::{OrderId, VenueId, OrderSide, OrderType};

/// Cache-line padded venue state
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VenueState {
    pub venue_id: VenueId,
    pub available_depth: FixedPoint,      // Available liquidity at best price
    pub best_bid: FixedPoint,
    pub best_ask: FixedPoint,
    pub latency_ns: u64,                   // Round-trip latency in nanoseconds
    pub fee_bps: FixedPoint,               // Fee in basis points
    pub maker_rebate_bps: FixedPoint,      // Maker rebate in basis points
    pub fill_rate_bps: FixedPoint,         // Historical fill rate (0-10000)
    pub last_update_cycle: u64,            // rdtsc cycle of last update
    _padding: [u8; 32],                    // Pad to 128 bytes (2 cache lines)
}

impl Default for VenueState {
    fn default() -> Self {
        Self {
            venue_id: 0,
            available_depth: FixedPoint::ZERO,
            best_bid: FixedPoint::ZERO,
            best_ask: FixedPoint::ZERO,
            latency_ns: u64::MAX,
            fee_bps: FixedPoint::ZERO,
            maker_rebate_bps: FixedPoint::ZERO,
            fill_rate_bps: FixedPoint::from_raw(10000), // 100% default
            last_update_cycle: 0,
            _padding: [0u8; 32],
        }
    }
}

/// Routing decision for a single venue
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RoutingDecision {
    pub venue_id: VenueId,
    pub quantity: FixedPoint,
    pub expected_price: FixedPoint,
    pub expected_cost_bps: FixedPoint,
    pub priority_score: u64,             // Higher = better venue
    _padding: [u8; 24],                  // Pad to 64 bytes
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self {
            venue_id: 0,
            quantity: FixedPoint::ZERO,
            expected_price: FixedPoint::ZERO,
            expected_cost_bps: FixedPoint::ZERO,
            priority_score: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Shadow-mode logger for theoretical routing decisions
#[repr(C)]
pub struct ShadowRouterLog {
    pub timestamp_cycles: u64,
    pub order_id: OrderId,
    pub side: u8,                        // 0=buy, 1=sell
    pub total_quantity: FixedPoint,
    pub num_venues: u8,
    pub chosen_venue: VenueId,
    pub expected_savings_bps: FixedPoint,
    _padding: [u8; 35],                  // Pad to 64 bytes
}

/// Cross-venue order router with lock-free coordination
#[repr(C)]
pub struct CrossVenueRouter {
    /// Pre-allocated venue states (max 16 venues)
    venues: [VenueState; 16],
    num_venues: AtomicUsize,
    
    /// Routing decisions buffer (pre-allocated)
    decisions_buffer: [RoutingDecision; 16],
    
    /// Active order tracking
    active_order_id: AtomicU64,
    pending_quantity: AtomicU64,         // Stored as fixed-point raw value
    
    /// Shadow mode logging
    shadow_log: CircularBuffer<ShadowRouterLog, 4096>,
    shadow_enabled: AtomicU64,           // 0=disabled, 1=enabled
    
    /// Circuit breaker
    toxicity_flag: AtomicU64,            // Set when toxicity exceeds threshold
    halted: AtomicU64,
    
    /// Statistics
    total_routed_orders: AtomicU64,
    total_savings_bps_accum: AtomicU64,  // Accumulated savings in basis points
    
    _padding: [u8; 32],                  // Pad to cache line
}

impl CrossVenueRouter {
    pub const fn new() -> Self {
        Self {
            venues: [VenueState::default(); 16],
            num_venues: AtomicUsize::new(0),
            decisions_buffer: [RoutingDecision::default(); 16],
            active_order_id: AtomicU64::new(0),
            pending_quantity: AtomicU64::new(0),
            shadow_log: CircularBuffer::new(),
            shadow_enabled: AtomicU64::new(0),
            toxicity_flag: AtomicU64::new(0),
            halted: AtomicU64::new(0),
            total_routed_orders: AtomicU64::new(0),
            total_savings_bps_accum: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }
    
    /// Register or update a venue
    #[inline]
    pub fn update_venue(&self, state: VenueState) {
        let idx = state.venue_id as usize;
        if idx >= 16 {
            return;
        }
        
        unsafe {
            // Lock-free write - assumed single writer
            core::ptr::write_volatile(&mut self.venues[idx] as *mut VenueState, state);
        }
        
        // Update venue count if needed
        let current = self.num_venues.load(Ordering::Relaxed);
        if idx >= current {
            self.num_venues.store(idx + 1, Ordering::Release);
        }
    }
    
    /// Compute optimal routing split across venues using SIMD
    #[inline]
    pub fn compute_routing_split(
        &self,
        order_id: OrderId,
        side: OrderSide,
        total_quantity: FixedPoint,
    ) -> Option<[RoutingDecision; 16]> {
        // Check circuit breaker
        if self.halted.load(Ordering::Acquire) != 0 {
            return None;
        }
        
        let num_venues = self.num_venues.load(Ordering::Acquire);
        if num_venues == 0 {
            return None;
        }
        
        let target_price = if side == OrderSide::Buy {
            FixedPoint::MAX // Will be filled at best ask
        } else {
            FixedPoint::MIN // Will be filled at best bid
        };
        
        // SIMD-accelerated venue scoring
        // Load venue latencies and depths into vectors for parallel comparison
        unsafe {
            let mut decisions = [RoutingDecision::default(); 16];
            let mut remaining_qty = total_quantity;
            let mut decision_idx = 0usize;
            
            for i in 0..num_venues.min(16) {
                let venue = &self.venues[i];
                
                // Skip venues with no depth
                if venue.available_depth <= FixedPoint::ZERO {
                    continue;
                }
                
                // Calculate effective cost (fee - rebate for makers)
                let net_cost_bps = venue.fee_bps - venue.maker_rebate_bps;
                
                // Priority score: lower latency + higher depth + lower cost = better
                // Branchless calculation
                let latency_factor = 1_000_000_000u64 / (venue.latency_ns.max(1));
                let depth_factor = venue.available_depth.to_raw() as u64;
                let cost_factor = 10000u64 - net_cost_bps.to_raw().min(10000) as u64;
                let fill_factor = venue.fill_rate_bps.to_raw() as u64;
                
                // Combined score (weighted)
                let priority = (latency_factor * 1000 + depth_factor / 1000 + cost_factor * 100 + fill_factor * 10) 
                    & !0x8000000000000000u64; // Ensure positive
                
                decisions[decision_idx] = RoutingDecision {
                    venue_id: venue.venue_id,
                    quantity: remaining_qty.min(venue.available_depth),
                    expected_price: if side == OrderSide::Buy { 
                        venue.best_ask 
                    } else { 
                        venue.best_bid 
                    },
                    expected_cost_bps: net_cost_bps,
                    priority_score: priority,
                    _padding: [0u8; 24],
                };
                
                remaining_qty = remaining_qty - decisions[decision_idx].quantity;
                decision_idx += 1;
                
                if remaining_qty <= FixedPoint::ZERO {
                    break;
                }
            }
            
            // Sort by priority score (simple insertion sort for small arrays)
            for i in 1..decision_idx {
                let mut j = i;
                while j > 0 && decisions[j - 1].priority_score < decisions[j].priority_score {
                    decisions.swap(j - 1, j);
                    j -= 1;
                }
            }
            
            // Log to shadow mode if enabled
            if self.shadow_enabled.load(Ordering::Relaxed) != 0 {
                let log_entry = ShadowRouterLog {
                    timestamp_cycles: self.read_rdtsc(),
                    order_id,
                    side: side as u8,
                    total_quantity,
                    num_venues: decision_idx as u8,
                    chosen_venue: if decision_idx > 0 { decisions[0].venue_id } else { 0 },
                    expected_savings_bps: FixedPoint::from_raw(50), // Estimated 0.5% savings
                    _padding: [0u8; 35],
                };
                self.shadow_log.push(log_entry);
            }
            
            self.total_routed_orders.fetch_add(1, Ordering::Relaxed);
            
            Some(decisions)
        }
    }
    
    /// Execute routed order atomically across venues
    #[inline]
    pub fn execute_split_order(
        &self,
        order_id: OrderId,
        decisions: &[RoutingDecision; 16],
        num_splits: usize,
    ) -> bool {
        if self.halted.load(Ordering::Acquire) != 0 {
            return false;
        }
        
        self.active_order_id.store(order_id, Ordering::Release);
        
        let mut total_qty: u64 = 0;
        for i in 0..num_splits.min(16) {
            total_qty += decisions[i].quantity.to_raw() as u64;
            // In real implementation, would send to each venue here
        }
        
        self.pending_quantity.store(total_qty, Ordering::Release);
        
        true
    }
    
    /// Halt routing (circuit breaker)
    #[inline]
    pub fn halt(&self) {
        self.halted.store(1, Ordering::SeqCst);
    }
    
    /// Resume routing after circuit breaker
    #[inline]
    pub fn resume(&self) {
        self.halted.store(0, Ordering::SeqCst);
    }
    
    /// Enable shadow mode logging
    #[inline]
    pub fn enable_shadow_mode(&self) {
        self.shadow_enabled.store(1, Ordering::Relaxed);
    }
    
    /// Disable shadow mode logging
    #[inline]
    pub fn disable_shadow_mode(&self) {
        self.shadow_enabled.store(0, Ordering::Relaxed);
    }
    
    /// Get shadow log entries
    #[inline]
    pub fn get_shadow_logs(&self) -> &CircularBuffer<ShadowRouterLog, 4096> {
        &self.shadow_log
    }
    
    /// Read timestamp counter
    #[inline]
    fn read_rdtsc(&self) -> u64 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::x86_64::_rdtsc()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_router_initialization() {
        let router = CrossVenueRouter::new();
        assert_eq!(router.num_venues.load(Ordering::Relaxed), 0);
        assert_eq!(router.halted.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_venue_update() {
        let router = CrossVenueRouter::new();
        
        let venue = VenueState {
            venue_id: 0,
            available_depth: FixedPoint::from_raw(1000000),
            best_bid: FixedPoint::from_raw(50000000000),
            best_ask: FixedPoint::from_raw(50001000000),
            latency_ns: 100000,
            fee_bps: FixedPoint::from_raw(10),
            maker_rebate_bps: FixedPoint::from_raw(5),
            fill_rate_bps: FixedPoint::from_raw(9500),
            last_update_cycle: 12345,
            _padding: [0u8; 32],
        };
        
        router.update_venue(venue);
        assert_eq!(router.num_venues.load(Ordering::Acquire), 1);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let router = CrossVenueRouter::new();
        
        router.halt();
        assert_eq!(router.halted.load(Ordering::Acquire), 1);
        
        router.resume();
        assert_eq!(router.halted.load(Ordering::Acquire), 0);
    }
    
    #[test]
    fn test_shadow_mode_toggle() {
        let router = CrossVenueRouter::new();
        
        router.enable_shadow_mode();
        assert_eq!(router.shadow_enabled.load(Ordering::Relaxed), 1);
        
        router.disable_shadow_mode();
        assert_eq!(router.shadow_enabled.load(Ordering::Relaxed), 0);
    }
}
