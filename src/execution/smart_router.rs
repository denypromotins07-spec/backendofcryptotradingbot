//! Smart Order Router (SOR)
//! 
//! Evaluates venue latency and liquidity depth to route orders optimally.
//! Dynamically penalizes venues with high adverse-selection toxicity.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of venues supported
const MAX_VENUES: usize = 16;

/// Venue health status
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VenueHealth {
    Healthy = 0,
    Degraded = 1,
    Unhealthy = 2,
    Offline = 3,
}

/// Venue statistics - cache-line aligned
#[repr(C)]
struct VenueStats {
    /// Average round-trip latency in nanoseconds
    avg_latency_ns: AtomicU64,
    /// Recent fill rate (basis points)
    fill_rate_bp: AtomicU64,
    /// Adverse selection score (higher = worse)
    toxicity_score: AtomicU64,
    /// Available liquidity at best bid (base units)
    bid_liquidity: AtomicU64,
    /// Available liquidity at best ask (base units)
    ask_liquidity: AtomicU64,
    /// Best bid price
    best_bid: AtomicU64,
    /// Best ask price
    best_ask: AtomicU64,
    /// WebSocket message sequence number
    ws_sequence: AtomicU64,
    /// Last heartbeat timestamp (cycles)
    last_heartbeat: AtomicU64,
    /// Health status
    health: AtomicU64,
    /// Total fills count
    total_fills: AtomicU64,
    /// Total rejected count
    total_rejected: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 80],
}

impl VenueStats {
    const fn new() -> Self {
        Self {
            avg_latency_ns: AtomicU64::new(0),
            fill_rate_bp: AtomicU64::new(10000), // 100% default
            toxicity_score: AtomicU64::new(0),
            bid_liquidity: AtomicU64::new(0),
            ask_liquidity: AtomicU64::new(0),
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(0),
            ws_sequence: AtomicU64::new(0),
            last_heartbeat: AtomicU64::new(0),
            health: AtomicU64::new(VenueHealth::Healthy as u64),
            total_fills: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 80],
        }
    }

    #[inline(always)]
    fn update_latency(&self, latency_ns: u64) {
        // Exponential moving average: new = alpha * sample + (1-alpha) * old
        // Using alpha = 1/8 for smoothing
        let old = self.avg_latency_ns.load(Ordering::Relaxed);
        let new = (old * 7 + latency_ns) / 8;
        self.avg_latency_ns.store(new, Ordering::Release);
    }

    #[inline(always)]
    fn record_fill(&self) {
        self.total_fills.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn record_rejection(&self) {
        self.total_rejected.fetch_add(1, Ordering::Relaxed);
        // Update fill rate
        let fills = self.total_fills.load(Ordering::Relaxed);
        let rejects = self.total_rejected.load(Ordering::Relaxed);
        let total = fills + rejects;
        if total > 0 {
            let rate = (fills * 10000) / total;
            self.fill_rate_bp.store(rate, Ordering::Release);
        }
    }
}

/// Routing decision result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    pub venue_id: u8,
    pub confidence_score: u16,
    pub expected_latency_ns: u64,
    pub available_liquidity: u64,
    pub penalty_applied: u64,
    pub _padding: [u8; 22],
}

impl RoutingDecision {
    const fn empty() -> Self {
        Self {
            venue_id: 0xFF,
            confidence_score: 0,
            expected_latency_ns: u64::MAX,
            available_liquidity: 0,
            penalty_applied: 0,
            _padding: [0u8; 22],
        }
    }
}

/// Smart Order Router
pub struct SmartRouter {
    /// Per-venue statistics
    venues: [VenueStats; MAX_VENUES],
    /// Number of active venues
    num_venues: AtomicU64,
    /// Minimum liquidity threshold (base units)
    min_liquidity: AtomicU64,
    /// Maximum acceptable latency (nanoseconds)
    max_latency_ns: AtomicU64,
    /// Toxicity penalty multiplier (basis points)
    toxicity_penalty_bp: AtomicU64,
    /// Global routing enabled flag
    routing_enabled: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for SmartRouter {}
unsafe impl Sync for SmartRouter {}

impl SmartRouter {
    /// Create new smart router
    pub const fn new() -> Self {
        const EMPTY_VENUE: VenueStats = VenueStats::new();
        Self {
            venues: [EMPTY_VENUE; MAX_VENUES],
            num_venues: AtomicU64::new(0),
            min_liquidity: AtomicU64::new(1000),
            max_latency_ns: AtomicU64::new(10_000_000), // 10ms default
            toxicity_penalty_bp: AtomicU64::new(5000), // 50% penalty max
            routing_enabled: AtomicBool::new(true),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Register a venue
    #[inline(always)]
    pub fn register_venue(&self, venue_id: u8) {
        if venue_id < MAX_VENUES as u8 {
            let idx = venue_id as usize;
            self.venues[idx].health.store(VenueHealth::Healthy as u64, Ordering::Release);
            self.venues[idx].last_heartbeat.store(unsafe { core::arch::x86_64::_rdtsc() }, Ordering::Release);
            
            let count = self.num_venues.load(Ordering::Relaxed);
            self.num_venues.store(count + 1, Ordering::Release);
        }
    }

    /// Update venue latency measurement
    #[inline(always)]
    pub fn update_venue_latency(&self, venue_id: u8, latency_ns: u64) {
        if venue_id < MAX_VENUES as u8 {
            self.venues[venue_id as usize].update_latency(latency_ns);
        }
    }

    /// Update venue liquidity
    #[inline(always)]
    pub fn update_venue_liquidity(&self, venue_id: u8, bid_qty: u64, ask_qty: u64, bid_price: u64, ask_price: u64) {
        if venue_id < MAX_VENUES as u8 {
            let venue = &self.venues[venue_id as usize];
            venue.bid_liquidity.store(bid_qty, Ordering::Relaxed);
            venue.ask_liquidity.store(ask_qty, Ordering::Relaxed);
            venue.best_bid.store(bid_price, Ordering::Relaxed);
            venue.best_ask.store(ask_price, Ordering::Relaxed);
        }
    }

    /// Update venue toxicity score (from market analysis)
    #[inline(always)]
    pub fn update_toxicity_score(&self, venue_id: u8, score: u64) {
        if venue_id < MAX_VENUES as u8 {
            // Score is 0-10000, higher = more toxic
            self.venues[venue_id as usize].toxicity_score.store(score.min(10000), Ordering::Release);
        }
    }

    /// Record venue heartbeat
    #[inline(always)]
    pub fn record_heartbeat(&self, venue_id: u8, sequence: u64) {
        if venue_id < MAX_VENUES as u8 {
            let venue = &self.venues[venue_id as usize];
            venue.ws_sequence.store(sequence, Ordering::Release);
            venue.last_heartbeat.store(unsafe { core::arch::x86_64::_rdtsc() }, Ordering::Release);
            
            // Check if venue is healthy based on heartbeat recency
            let current_tsc = unsafe { core::arch::x86_64::_rdtsc() };
            let elapsed = current_tsc.saturating_sub(venue.last_heartbeat.load(Ordering::Relaxed));
            
            // Assume ~3GHz CPU, 100M cycles = ~33ms
            let health = if elapsed < 100_000_000 {
                VenueHealth::Healthy
            } else if elapsed < 500_000_000 {
                VenueHealth::Degraded
            } else {
                VenueHealth::Unhealthy
            };
            
            venue.health.store(health as u64, Ordering::Release);
        }
    }

    /// Record fill for venue
    #[inline(always)]
    pub fn record_fill(&self, venue_id: u8) {
        if venue_id < MAX_VENUES as u8 {
            self.venues[venue_id as usize].record_fill();
        }
    }

    /// Record rejection for venue
    #[inline(always)]
    pub fn record_rejection(&self, venue_id: u8) {
        if venue_id < MAX_VENUES as u8 {
            self.venues[venue_id as usize].record_rejection();
        }
    }

    /// Route a buy order to the best venue
    #[inline(always)]
    pub fn route_buy(&self, quantity: u64) -> RoutingDecision {
        self.route_order(quantity, true)
    }

    /// Route a sell order to the best venue
    #[inline(always)]
    pub fn route_sell(&self, quantity: u64) -> RoutingDecision {
        self.route_order(quantity, false)
    }

    /// Core routing logic
    #[inline(always)]
    fn route_order(&self, quantity: u64, is_buy: bool) -> RoutingDecision {
        if !self.routing_enabled.load(Ordering::Acquire) {
            return RoutingDecision::empty();
        }

        let mut best_venue: u8 = 0xFF;
        let mut best_score: i64 = -1;
        let mut best_liquidity: u64 = 0;
        let mut best_latency: u64 = u64::MAX;
        let mut best_penalty: u64 = 0;

        let num_venues = self.num_venues.load(Ordering::Relaxed) as usize;
        let min_liq = self.min_liquidity.load(Ordering::Relaxed);
        let max_lat = self.max_latency_ns.load(Ordering::Relaxed);
        let tox_penalty = self.toxicity_penalty_bp.load(Ordering::Relaxed);

        for i in 0..num_venues.min(MAX_VENUES) {
            let venue = &self.venues[i];
            
            // Check health (branchless)
            let health = venue.health.load(Ordering::Relaxed);
            let health_ok = (health <= VenueHealth::Degraded as u64) as i64;
            
            // Get liquidity for side
            let liquidity = if is_buy {
                venue.ask_liquidity.load(Ordering::Relaxed)
            } else {
                venue.bid_liquidity.load(Ordering::Relaxed)
            };

            // Branchless liquidity check
            let liq_ok = ((liquidity >= min_liq) | (quantity <= liquidity)) as i64;

            // Get latency
            let latency = venue.avg_latency_ns.load(Ordering::Relaxed);
            let lat_ok = (latency <= max_lat) as i64;

            // Calculate composite score
            // Higher is better: low latency, high liquidity, low toxicity, good health
            let toxicity = venue.toxicity_score.load(Ordering::Relaxed);
            let fill_rate = venue.fill_rate_bp.load(Ordering::Relaxed);

            // Score components (scaled to avoid overflow)
            let latency_score = if latency > 0 { 1_000_000_000 / latency } else { 1_000_000 };
            let liq_score = liquidity / 1000;
            let tox_penalty_val = (toxicity * tox_penalty) / 10000;
            let fill_score = fill_rate;

            // Composite: latency + liquidity + fill_rate - toxicity_penalty
            let score = (latency_score as i64) 
                + (liq_score as i64) 
                + (fill_score as i64)
                - (tox_penalty_val as i64);

            // Apply health multiplier (branchless)
            let adjusted_score = score * health_ok;

            // Check if this venue is better
            let is_better = ((adjusted_score > best_score) 
                & (liq_ok != 0) 
                & (lat_ok != 0)) as u8;

            best_venue = best_venue * (1 - is_better) + (i as u8) * is_better;
            best_score = best_score * (1 - is_better as i64) + adjusted_score * is_better as i64;
            best_liquidity = best_liquidity * (1 - is_better as u64) + liquidity * is_better as u64;
            best_latency = best_latency * (1 - is_better as u64) + latency * is_better as u64;
            best_penalty = tox_penalty_val;
        }

        let confidence = if best_venue == 0xFF {
            0
        } else {
            ((best_score + 1000) / 10) as u16
        };

        RoutingDecision {
            venue_id: best_venue,
            confidence_score: confidence.min(10000),
            expected_latency_ns: best_latency,
            available_liquidity: best_liquidity,
            penalty_applied: best_penalty,
            _padding: [0u8; 22],
        }
    }

    /// Split order across multiple venues
    #[inline(always)]
    pub fn split_order(&self, quantity: u64, max_venues: u8) -> [RoutingDecision; 4] {
        const EMPTY: RoutingDecision = RoutingDecision::empty();
        let mut decisions = [EMPTY; 4];
        
        if max_venues == 0 || quantity == 0 {
            return decisions;
        }

        let mut remaining = quantity;
        let max_splits = max_venues.min(4) as usize;

        for i in 0..max_splits {
            if remaining == 0 {
                break;
            }

            let decision = self.route_buy(remaining);
            if decision.venue_id == 0xFF {
                break;
            }

            // Determine split quantity (proportional to available liquidity)
            let split_qty = if decision.available_liquidity > 0 {
                let qty = (remaining * decision.available_liquidity) 
                    / (decision.available_liquidity + remaining);
                qty.min(remaining)
            } else {
                remaining
            };

            decisions[i] = RoutingDecision {
                venue_id: decision.venue_id,
                confidence_score: decision.confidence_score,
                expected_latency_ns: decision.expected_latency_ns,
                available_liquidity: split_qty,
                penalty_applied: decision.penalty_applied,
                _padding: [0u8; 22],
            };

            remaining = remaining.saturating_sub(split_qty);
        }

        decisions
    }

    /// Disable routing (emergency stop)
    #[inline(always)]
    pub fn disable_routing(&self) {
        self.routing_enabled.store(false, Ordering::SeqCst);
    }

    /// Enable routing
    #[inline(always)]
    pub fn enable_routing(&self) {
        self.routing_enabled.store(true, Ordering::SeqCst);
    }

    /// Get venue health status
    #[inline(always)]
    pub fn get_venue_health(&self, venue_id: u8) -> VenueHealth {
        if venue_id < MAX_VENUES as u8 {
            match self.venues[venue_id as usize].health.load(Ordering::Relaxed) {
                0 => VenueHealth::Healthy,
                1 => VenueHealth::Degraded,
                2 => VenueHealth::Unhealthy,
                _ => VenueHealth::Offline,
            }
        } else {
            VenueHealth::Offline
        }
    }

    /// Set minimum liquidity threshold
    #[inline(always)]
    pub fn set_min_liquidity(&self, min_liq: u64) {
        self.min_liquidity.store(min_liq, Ordering::Release);
    }

    /// Set maximum acceptable latency
    #[inline(always)]
    pub fn set_max_latency(&self, max_lat_ns: u64) {
        self.max_latency_ns.store(max_lat_ns, Ordering::Release);
    }
}

impl Default for SmartRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_decision_size() {
        assert_eq!(core::mem::size_of::<RoutingDecision>(), 48);
    }

    #[test]
    fn test_venue_registration() {
        let router = SmartRouter::new();
        router.register_venue(0);
        router.register_venue(1);

        assert_eq!(router.num_venues.load(Ordering::Relaxed), 2);
        assert_eq!(router.get_venue_health(0), VenueHealth::Healthy);
    }

    #[test]
    fn test_latency_update() {
        let router = SmartRouter::new();
        router.register_venue(0);

        router.update_venue_latency(0, 100_000); // 100us
        router.update_venue_latency(0, 200_000); // 200us
        
        // EMA should be between the two values
        let avg = router.venues[0].avg_latency_ns.load(Ordering::Relaxed);
        assert!(avg > 100_000 && avg < 200_000);
    }

    #[test]
    fn test_routing_decision() {
        let router = SmartRouter::new();
        router.register_venue(0);
        router.register_venue(1);

        // Make venue 0 more attractive
        router.update_venue_latency(0, 50_000);
        router.update_venue_latency(1, 200_000);
        router.update_venue_liquidity(0, 10000, 10000, 49990, 50010);
        router.update_venue_liquidity(1, 1000, 1000, 49990, 50010);

        let decision = router.route_buy(100);
        assert_eq!(decision.venue_id, 0);
        assert!(decision.confidence_score > 0);
    }

    #[test]
    fn test_toxicity_penalty() {
        let router = SmartRouter::new();
        router.register_venue(0);
        router.register_venue(1);

        router.update_venue_latency(0, 100_000);
        router.update_venue_latency(1, 100_000);
        router.update_venue_liquidity(0, 10000, 10000, 49990, 50010);
        router.update_venue_liquidity(1, 10000, 10000, 49990, 50010);

        // Make venue 0 toxic
        router.update_toxicity_score(0, 8000); // High toxicity

        let decision = router.route_buy(100);
        // Venue 1 should be preferred due to lower toxicity
        assert_eq!(decision.venue_id, 1);
    }

    #[test]
    fn test_split_order() {
        let router = SmartRouter::new();
        router.register_venue(0);
        router.register_venue(1);

        router.update_venue_liquidity(0, 5000, 5000, 49990, 50010);
        router.update_venue_liquidity(1, 5000, 5000, 49990, 50010);

        let decisions = router.split_order(10000, 2);
        
        // Should have at least one valid decision
        assert!(decisions[0].venue_id != 0xFF || decisions[1].venue_id != 0xFF);
    }

    #[test]
    fn test_routing_disable() {
        let router = SmartRouter::new();
        router.register_venue(0);

        let decision_before = router.route_buy(100);
        assert_ne!(decision_before.venue_id, 0xFF);

        router.disable_routing();
        
        let decision_after = router.route_buy(100);
        assert_eq!(decision_after.venue_id, 0xFF);
    }
}
