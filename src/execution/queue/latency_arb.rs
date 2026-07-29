//! Latency arbitrage detector to pull passive quotes instantly.
//! 
//! Detects latency arbitrage opportunities by monitoring:
//! - Cross-venue price discrepancies
//! - Order book update latencies
//! - Trade-through events
//! 
//! Uses rdtsc for nanosecond-precision timing and lock-free flags.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Number of venues we track
const MAX_VENUES: usize = 16;

/// Latency threshold in cycles (approximately 1 microsecond at 3GHz)
const LATENCY_THRESHOLD_CYCLES: u64 = 3000;

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Venue state - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VenueState {
    pub venue_id: u32,
    pub best_bid: i64,              // Fixed-point price
    pub best_ask: i64,              // Fixed-point price
    pub last_update_cycles: u64,    // rdtsc timestamp of last update
    pub latency_us: u32,            // Measured latency in microseconds
    pub is_stale: bool,             // Is this quote stale?
    _padding: [u8; 43],             // Pad to 64 bytes
}

impl Default for VenueState {
    fn default() -> Self {
        Self {
            venue_id: 0,
            best_bid: 0,
            best_ask: 0,
            last_update_cycles: 0,
            latency_us: 0,
            is_stale: false,
            _padding: [0; 43],
        }
    }
}

/// Arbitrage opportunity detected
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArbOpportunity {
    pub buy_venue: u32,
    pub sell_venue: u32,
    pub profit_bps: i64,            // Expected profit in basis points
    pub size: i64,                  // Available size
    pub detected_cycles: u64,       // Detection timestamp
    pub expires_cycles: u64,        // When this opportunity expires
    pub is_valid: bool,
    _padding: [u8; 38],             // Pad to 64 bytes
}

impl Default for ArbOpportunity {
    fn default() -> Self {
        Self {
            buy_venue: 0,
            sell_venue: 0,
            profit_bps: 0,
            size: 0,
            detected_cycles: 0,
            expires_cycles: 0,
            is_valid: false,
            _padding: [0; 38],
        }
    }
}

/// Lock-free latency arbitrage detector
#[repr(C)]
pub struct LatencyArbDetector {
    /// State for each venue
    venues: [VenueState; MAX_VENUES],
    
    /// Best bid/ask across all venues
    global_best_bid: AtomicI64,
    global_best_ask: AtomicI64,
    global_bid_venue: AtomicU64,
    global_ask_venue: AtomicU64,
    
    /// Latest detected opportunity
    current_opportunity: ArbOpportunity,
    
    /// Statistics
    opportunities_detected: AtomicU64,
    opportunities_expired: AtomicU64,
    total_profit_bps: AtomicI64,
    
    /// Latency tracking
    avg_latency_cycles: AtomicU64,
    max_latency_cycles: AtomicU64,
    latency_samples: AtomicU64,
    
    /// Kill switches
    arb_enabled: AtomicBool,
    toxicity_flag: AtomicBool,      // Set if toxic flow detected
    
    _padding: [u8; 16],             // Pad to cache line
}

impl LatencyArbDetector {
    /// Create new latency arbitrage detector
    pub const fn new() -> Self {
        Self {
            venues: [VenueState::default(); MAX_VENUES],
            global_best_bid: AtomicI64::new(0),
            global_best_ask: AtomicI64::new(0),
            global_bid_venue: AtomicU64::new(0),
            global_ask_venue: AtomicU64::new(0),
            current_opportunity: ArbOpportunity::default(),
            opportunities_detected: AtomicU64::new(0),
            opportunities_expired: AtomicU64::new(0),
            total_profit_bps: AtomicI64::new(0),
            avg_latency_cycles: AtomicU64::new(0),
            max_latency_cycles: AtomicU64::new(0),
            latency_samples: AtomicU64::new(0),
            arb_enabled: AtomicBool::new(true),
            toxicity_flag: AtomicBool::new(false),
            _padding: [0; 16],
        }
    }
    
    /// Update venue state
    #[inline(always)]
    pub fn update_venue(&mut self, venue_idx: usize, bid: i64, ask: i64, venue_id: u32) {
        if venue_idx >= MAX_VENUES {
            return;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        
        let venue = &mut self.venues[venue_idx];
        venue.venue_id = venue_id;
        venue.best_bid = bid;
        venue.best_ask = ask;
        venue.last_update_cycles = cycles;
        venue.is_stale = false;
        
        // Calculate latency from previous update
        let latency = cycles.wrapping_sub(venue.last_update_cycles);
        venue.latency_us = ((latency as f64 / 3000.0) as u32).min(u32::MAX);
        
        // Update latency statistics (branchless)
        let samples = self.latency_samples.load(Ordering::Relaxed);
        let avg = self.avg_latency_cycles.load(Ordering::Relaxed);
        let max = self.max_latency_cycles.load(Ordering::Relaxed);
        
        let new_avg = ((avg as u128 * samples as u128 + latency as u128) / (samples + 1) as u128) as u64;
        let new_max = max.max(latency);
        
        self.avg_latency_cycles.store(new_avg, Ordering::Relaxed);
        self.max_latency_cycles.store(new_max, Ordering::Relaxed);
        self.latency_samples.store(samples + 1, Ordering::Relaxed);
        
        // Update global bests
        self.update_global_bests();
    }
    
    /// Update global best bid/ask across venues
    #[inline(always)]
    fn update_global_bests(&self) {
        let mut best_bid = i64::MIN;
        let mut best_ask = i64::MAX;
        let mut bid_venue = 0u64;
        let mut ask_venue = 0u64;
        
        // SIMD-like parallel comparison (unrolled loop)
        for i in 0..MAX_VENUES {
            let venue = unsafe { self.venues.get_unchecked(i) };
            
            // Branchless max/min
            let bid_mask = -(venue.best_bid > best_bid) as i64;
            best_bid = (best_bid & !bid_mask) | (venue.best_bid & bid_mask);
            bid_venue = (bid_venue & !(bid_mask as u64)) | ((i as u64) & (bid_mask as u64));
            
            let ask_mask = -(venue.best_ask < best_ask) as i64;
            best_ask = (best_ask & !ask_mask) | (venue.best_ask & ask_mask);
            ask_venue = (ask_venue & !(ask_mask as u64)) | ((i as u64) & (ask_mask as u64));
        }
        
        self.global_best_bid.store(best_bid, Ordering::Release);
        self.global_best_ask.store(best_ask, Ordering::Release);
        self.global_bid_venue.store(bid_venue, Ordering::Release);
        self.global_ask_venue.store(ask_venue, Ordering::Release);
    }
    
    /// Detect arbitrage opportunities
    #[inline(always)]
    pub fn detect_arbitrage(&mut self) -> Option<ArbOpportunity> {
        if !self.arb_enabled.load(Ordering::Acquire) || self.toxicity_flag.load(Ordering::Acquire) {
            return None;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        
        let mut best_opportunity: Option<ArbOpportunity> = None;
        
        // Check all venue pairs for arbitrage
        for i in 0..MAX_VENUES {
            for j in (i + 1)..MAX_VENUES {
                let venue_i = unsafe { self.venues.get_unchecked(i) };
                let venue_j = unsafe { self.venues.get_unchecked(j) };
                
                // Skip stale venues
                if venue_i.is_stale || venue_j.is_stale {
                    continue;
                }
                
                // Check for cross-venue arbitrage
                // Buy at venue_i bid, sell at venue_j ask
                let spread_ij = venue_j.best_ask - venue_i.best_bid;
                let profit_ij = -spread_ij; // Negative spread = profit
                
                // Buy at venue_j bid, sell at venue_i ask
                let spread_ji = venue_i.best_ask - venue_j.best_bid;
                let profit_ji = -spread_ji;
                
                // Find best opportunity (branchless)
                let profit_max = profit_ij.max(profit_ji);
                
                if profit_max > 0 {
                    let opp = ArbOpportunity {
                        buy_venue: if profit_ij >= profit_ji { venue_i.venue_id } else { venue_j.venue_id },
                        sell_venue: if profit_ij >= profit_ji { venue_j.venue_id } else { venue_i.venue_id },
                        profit_bps: ((profit_max as i128 * 10_000) / venue_i.best_bid.abs().max(1) as i128) as i64,
                        size: venue_i.best_bid.min(venue_j.best_ask).abs(),
                        detected_cycles: cycles,
                        expires_cycles: cycles + LATENCY_THRESHOLD_CYCLES * 10, // ~10us lifetime
                        is_valid: true,
                        _padding: [0; 38],
                    };
                    
                    best_opportunity = Some(opp);
                }
            }
        }
        
        if let Some(opp) = best_opportunity {
            self.current_opportunity = opp;
            self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
            self.total_profit_bps.fetch_add(opp.profit_bps, Ordering::Relaxed);
            return Some(opp);
        }
        
        None
    }
    
    /// Check if current opportunity has expired
    #[inline(always)]
    pub fn check_expiry(&self) -> bool {
        if !self.current_opportunity.is_valid {
            return true;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        let expired = cycles > self.current_opportunity.expires_cycles;
        
        if expired {
            self.opportunities_expired.fetch_add(1, Ordering::Relaxed);
        }
        
        expired
    }
    
    /// Mark a venue as stale (quote too old)
    #[inline(always)]
    pub fn mark_stale(&mut self, venue_idx: usize, staleness_threshold_cycles: u64) {
        if venue_idx >= MAX_VENUES {
            return;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        let venue = &mut self.venues[venue_idx];
        
        let age = cycles.wrapping_sub(venue.last_update_cycles);
        venue.is_stale = age > staleness_threshold_cycles;
    }
    
    /// Pull quote (cancel passive order) due to latency arb detection
    #[inline(always)]
    pub fn should_pull_quote(&self, venue_idx: usize) -> bool {
        if venue_idx >= MAX_VENUES {
            return false;
        }
        
        let venue = unsafe { self.venues.get_unchecked(venue_idx) };
        
        // Check if venue is stale
        if venue.is_stale {
            return true;
        }
        
        // Check if we're on the wrong side of an arb opportunity
        if self.current_opportunity.is_valid {
            let opp = self.current_opportunity;
            
            // If we're quoting at the sell venue and there's an arb, pull
            let is_sell_venue = (venue.venue_id == opp.sell_venue) as u8;
            return is_sell_venue != 0;
        }
        
        false
    }
    
    /// Enable/disable arbitrage detection
    #[inline(always)]
    pub fn set_enabled(&self, enabled: bool) {
        self.arb_enabled.store(enabled, Ordering::Release);
    }
    
    /// Mark as toxic (stop arbitrage)
    #[inline(always)]
    pub fn mark_toxic(&self) {
        self.toxicity_flag.store(true, Ordering::Release);
    }
    
    /// Get statistics
    #[inline(always)]
    pub fn stats(&self) -> (u64, u64, i64) {
        (
            self.opportunities_detected.load(Ordering::Acquire),
            self.opportunities_expired.load(Ordering::Acquire),
            self.total_profit_bps.load(Ordering::Acquire),
        )
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<VenueState>() == 64);
        assert!(core::mem::size_of::<ArbOpportunity>() == 64);
        assert!(core::mem::size_of::<LatencyArbDetector>() % 64 == 0);
    }
    
    #[test]
    fn test_venue_update() {
        let mut detector = LatencyArbDetector::new();
        
        // Update two venues with different prices
        detector.update_venue(0, 100_000_000, 100_100_000, 1);
        detector.update_venue(1, 99_900_000, 100_000_000, 2);
        
        // Global bests should reflect the updates
        assert_eq!(detector.global_best_bid.load(Ordering::Acquire), 100_000_000);
        assert_eq!(detector.global_best_ask.load(Ordering::Acquire), 100_000_000);
    }
    
    #[test]
    fn test_arbitrage_detection() {
        let mut detector = LatencyArbDetector::new();
        
        // Set up clear arbitrage opportunity
        detector.update_venue(0, 100_000_000, 100_200_000, 1);
        detector.update_venue(1, 99_800_000, 100_000_000, 2);
        
        // Should detect arb: buy at venue 1 (99.8), sell at venue 0 (100.0)
        let opp = detector.detect_arbitrage();
        assert!(opp.is_some());
        
        let opp = opp.unwrap();
        assert!(opp.profit_bps > 0);
    }
}
