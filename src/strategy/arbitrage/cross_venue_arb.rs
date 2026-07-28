//! Latency-Aware Cross-Exchange Arbitrage Threshold Calculator
//! 
//! Implements cross-venue arbitrage detection with latency-adjusted thresholds.
//! Uses lock-free data structures and fixed-point arithmetic for microsecond execution.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Maximum number of venues supported
pub const MAX_VENUES: usize = 8;

/// Cache-line padded atomic boolean
#[repr(C, align(64))]
pub struct PaddedAtomicBool {
    value: AtomicBool,
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

/// Cache-line padded atomic u64
#[repr(C, align(64))]
pub struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 56],
}

impl PaddedAtomicU64 {
    #[inline(always)]
    pub const fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; 56],
        }
    }
    
    #[inline(always)]
    pub fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline(always)]
    pub fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed);
    }
}

/// Venue price state - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct VenuePrice {
    /// Venue identifier
    pub venue_id: u8,
    /// Asset identifier
    pub asset_id: u8,
    /// Price (Q32.32 fixed-point)
    pub price_i64: i64,
    /// Bid-ask spread (Q32.32)
    pub spread_i64: i64,
    /// Available liquidity (Q32.32)
    pub liquidity_i64: i64,
    /// Last update timestamp (TSC)
    pub timestamp_tsc: u64,
    /// Latency to venue in nanoseconds
    pub latency_ns: u32,
    /// Validity flag
    pub valid: u8,
    _padding: [u8; 31],
}

const _: () = assert!(core::mem::size_of::<VenuePrice>() == 64);

impl VenuePrice {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            venue_id: 0,
            asset_id: 0,
            price_i64: 0,
            spread_i64: 0,
            liquidity_i64: 0,
            timestamp_tsc: 0,
            latency_ns: 0,
            valid: 0,
            _padding: [0u8; 31],
        }
    }
    
    #[inline(always)]
    pub fn price(&self) -> f64 {
        self.price_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn is_fresh(&self, max_age_tsc: u64) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let now = unsafe { core::arch::x86_64::_rdtsc() };
            now - self.timestamp_tsc < max_age_tsc
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            true
        }
    }
}

/// Cross-venue arbitrage opportunity
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CrossVenueArbOpportunity {
    /// Buy venue ID
    pub buy_venue: u8,
    /// Sell venue ID
    pub sell_venue: u8,
    /// Asset ID
    pub asset_id: u8,
    /// Expected profit after costs and latency penalty (Q32.32, basis points)
    pub profit_bp_i64: i64,
    /// Recommended size (Q32.32)
    pub size_i64: i64,
    /// Latency penalty applied (nanoseconds)
    pub latency_penalty_ns: u32,
    /// Confidence score (0-255)
    pub confidence: u8,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    _padding: [u8; 38],
}

const _: () = assert!(core::mem::size_of::<CrossVenueArbOpportunity>() == 64);

impl CrossVenueArbOpportunity {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            buy_venue: 0,
            sell_venue: 0,
            asset_id: 0,
            profit_bp_i64: 0,
            size_i64: 0,
            latency_penalty_ns: 0,
            confidence: 0,
            timestamp_tsc: 0,
            _padding: [0u8; 38],
        }
    }
    
    #[inline(always)]
    pub fn profit_bp(&self) -> f64 {
        self.profit_bp_i64 as f64 / 4294967296.0
    }
}

/// Latency-adjusted threshold calculator
#[repr(C, align(64))]
pub struct LatencyThresholdCalculator {
    /// Base profit threshold (Q32.32, basis points)
    base_threshold_bp: i64,
    /// Latency penalty per millisecond (Q32.32)
    latency_penalty_per_ms: i64,
    /// Volatility adjustment factor (Q16.16)
    volatility_factor: i32,
    /// Current calculated threshold (Q32.32)
    pub current_threshold: i64,
    /// Last recalculation timestamp
    pub last_recalc_tsc: PaddedAtomicU64,
    _padding: [u8; 40],
}

impl LatencyThresholdCalculator {
    #[inline(always)]
    pub const fn new(base_threshold_bp: f64, latency_penalty_bp_per_ms: f64) -> Self {
        Self {
            base_threshold_bp: (base_threshold_bp * 4294967296.0) as i64,
            latency_penalty_per_ms: (latency_penalty_bp_per_ms * 4294967296.0) as i64,
            volatility_factor: 65536, // 1.0 in Q16.16
            current_threshold: 0,
            last_recalc_tsc: PaddedAtomicU64::new(0),
            _padding: [0u8; 40],
        }
    }
    
    /// Calculate latency-adjusted threshold
    #[inline(always)]
    pub fn calculate_threshold(&mut self, round_trip_latency_ns: u32, volatility: f64) -> i64 {
        // Convert latency to milliseconds (fixed-point)
        let latency_ms = (round_trip_latency_ns as i64) << 22; // Q22 representation
        
        // Latency penalty: latency_ms * penalty_per_ms / 1e6
        let latency_penalty = (latency_ms * self.latency_penalty_per_ms) >> 22;
        
        // Volatility adjustment (higher volatility = higher threshold)
        let vol_adjustment = ((volatility * 65536.0) as i32 * self.volatility_factor) >> 16;
        let vol_penalty = (self.base_threshold_bp * vol_adjustment as i64) >> 16;
        
        // Total threshold
        self.current_threshold = self.base_threshold_bp + latency_penalty + vol_penalty;
        
        #[cfg(target_arch = "x86_64")]
        {
            self.last_recalc_tsc.store(unsafe { core::arch::x86_64::_rdtsc() });
        }
        
        self.current_threshold
    }
    
    /// Update volatility factor
    #[inline(always)]
    pub fn set_volatility(&mut self, vol: f64) {
        self.volatility_factor = (vol * 65536.0) as i32;
    }
}

/// Cross-venue arbitrage engine
#[repr(C, align(64))]
pub struct CrossVenueArbEngine<const V: usize, const A: usize> {
    /// Price data per venue per asset
    prices: [[VenuePrice; A]; V],
    /// Venue latencies in nanoseconds
    venue_latencies: [u32; V],
    /// Threshold calculator
    threshold_calc: LatencyThresholdCalculator,
    /// Best opportunity found
    pub best_opportunity: CrossVenueArbOpportunity,
    /// Profit threshold (Q32.32)
    profit_threshold: i64,
    /// Circuit breaker
    pub circuit_breaker: PaddedAtomicBool,
    /// Opportunity count
    pub opportunity_count: PaddedAtomicU64,
    /// Cumulative P&L (Q32.32)
    pub cumulative_pnl: AtomicI64,
    _padding: [u8; 24],
}

impl<const V: usize, const A: usize> CrossVenueArbEngine<V, A> {
    #[inline(always)]
    pub const fn new(base_threshold_bp: f64, latency_penalty_bp_per_ms: f64) -> Self {
        const INIT_PRICE: VenuePrice = VenuePrice::new();
        const INIT_ROW: [VenuePrice; A] = [INIT_PRICE; A];
        const INIT_MATRIX: [[VenuePrice; A]; V] = [[INIT_PRICE; A]; V];
        
        Self {
            prices: INIT_MATRIX,
            venue_latencies: [0; V],
            threshold_calc: LatencyThresholdCalculator::new(base_threshold_bp, latency_penalty_bp_per_ms),
            best_opportunity: CrossVenueArbOpportunity::new(),
            profit_threshold: 0,
            circuit_breaker: PaddedAtomicBool::new(false),
            opportunity_count: PaddedAtomicU64::new(0),
            cumulative_pnl: AtomicI64::new(0),
            _padding: [0u8; 24],
        }
    }
    
    /// Update price for a venue/asset pair
    #[inline(always)]
    pub fn update_price(
        &mut self,
        venue_id: usize,
        asset_id: usize,
        price: i64,
        spread: i64,
        liquidity: i64,
        timestamp: u64,
    ) {
        if self.circuit_breaker.get() || venue_id >= V || asset_id >= A {
            return;
        }
        
        let vp = &mut self.prices[venue_id][asset_id];
        vp.venue_id = venue_id as u8;
        vp.asset_id = asset_id as u8;
        vp.price_i64 = price;
        vp.spread_i64 = spread;
        vp.liquidity_i64 = liquidity;
        vp.timestamp_tsc = timestamp;
        vp.latency_ns = self.venue_latencies[venue_id];
        vp.valid = 1;
    }
    
    /// Set venue latency
    #[inline(always)]
    pub fn set_venue_latency(&mut self, venue_id: usize, latency_ns: u32) {
        if venue_id < V {
            self.venue_latencies[venue_id] = latency_ns;
        }
    }
    
    /// Find cross-venue arbitrage opportunities
    #[inline(always)]
    pub fn find_arbitrage(&mut self, asset_id: usize, timestamp: u64) -> bool {
        if self.circuit_breaker.get() || asset_id >= A {
            return false;
        }
        
        let mut best_profit: i64 = 0;
        let mut buy_venue: u8 = 0;
        let mut sell_venue: u8 = 0;
        
        // Find best buy (lowest) and sell (highest) prices across venues
        let mut min_price = i64::MAX;
        let mut max_price = i64::MIN;
        let mut min_venue: u8 = 0;
        let mut max_venue: u8 = 0;
        
        for v in 0..V {
            let price = &self.prices[v][asset_id];
            if price.valid == 0 {
                continue;
            }
            
            // Check freshness (assume 1ms max age in TSC cycles ~2M cycles at 2GHz)
            if !price.is_fresh(2_000_000) {
                continue;
            }
            
            // Branchless min/max update
            let is_new_min = (price.price_i64 < min_price) as u8;
            min_price = price.price_i64 * is_new_min as i64 + min_price * (1 - is_new_min) as i64;
            min_venue = v as u8 * is_new_min + min_venue * (1 - is_new_min);
            
            let is_new_max = (price.price_i64 > max_price) as u8;
            max_price = price.price_i64 * is_new_max as i64 + max_price * (1 - is_new_max) as i64;
            max_venue = v as u8 * is_new_max + max_venue * (1 - is_new_max);
        }
        
        if min_venue == max_venue {
            return false; // Same venue, no arb
        }
        
        // Calculate round-trip latency
        let rt_latency = self.venue_latencies[min_venue as usize] 
                       + self.venue_latencies[max_venue as usize];
        
        // Calculate dynamic threshold based on latency
        let volatility = 0.02; // Would come from volatility model
        self.profit_threshold = self.threshold_calc.calculate_threshold(rt_latency, volatility);
        
        // Calculate profit: (max - min) / min - costs
        let price_diff = max_price - min_price;
        let profit_raw = if min_price > 0 {
            (price_diff << 16) / min_price // Q16.16 ratio
        } else {
            0
        };
        
        // Convert to basis points and apply latency penalty
        let profit_bp = (profit_raw * 10000) >> 16;
        let latency_penalty = ((rt_latency as i64 * self.threshold_calc.latency_penalty_per_ms) >> 32) as i64;
        let net_profit_bp = profit_bp - latency_penalty;
        
        if net_profit_bp > self.profit_threshold >> 16 {
            buy_venue = min_venue;
            sell_venue = max_venue;
            best_profit = net_profit_bp << 16; // Back to Q32.32
            
            self.best_opportunity.buy_venue = buy_venue;
            self.best_opportunity.sell_venue = sell_venue;
            self.best_opportunity.asset_id = asset_id as u8;
            self.best_opportunity.profit_bp_i64 = best_profit;
            self.best_opportunity.latency_penalty_ns = rt_latency;
            self.best_opportunity.confidence = Self::calc_confidence(net_profit_bp as u32, self.profit_threshold as u32);
            self.best_opportunity.timestamp_tsc = timestamp;
            
            // Size limited by minimum liquidity
            let liq = self.prices[buy_venue as usize][asset_id].liquidity_i64
                    .min(self.prices[sell_venue as usize][asset_id].liquidity_i64);
            self.best_opportunity.size_i64 = liq >> 2; // Use 25% of available
            
            self.opportunity_count.value.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        
        false
    }
    
    #[inline(always)]
    fn calc_confidence(profit: u32, threshold: u32) -> u8 {
        if threshold == 0 {
            return 255;
        }
        ((profit * 256) / threshold).min(255) as u8
    }
    
    /// Record fill
    #[inline(always)]
    pub fn record_fill(&self, pnl_q32: i64) {
        self.cumulative_pnl.fetch_add(pnl_q32, Ordering::Relaxed);
    }
    
    /// Get cumulative P&L
    #[inline(always)]
    pub fn cumulative_pnl_f64(&self) -> f64 {
        self.cumulative_pnl.load(Ordering::Relaxed) as f64 / 4294967296.0
    }
    
    /// Halt trading
    #[inline(always)]
    pub fn halt(&self) {
        self.circuit_breaker.set(true);
    }
    
    /// Resume trading
    #[inline(always)]
    pub fn resume(&self) {
        self.circuit_breaker.set(false);
    }
}

// Type alias for common crypto setup (4 venues, 4 assets)
pub type CryptoCrossVenueArb = CrossVenueArbEngine<4, 4>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_venue_price_size() {
        assert_eq!(core::mem::size_of::<VenuePrice>(), 64);
    }
    
    #[test]
    fn test_opportunity_size() {
        assert_eq!(core::mem::size_of::<CrossVenueArbOpportunity>(), 64);
    }
    
    #[test]
    fn test_cross_venue_arb() {
        let mut engine: CryptoCrossVenueArb = CrossVenueArbEngine::new(0.01, 0.001);
        
        let base = 4294967296i64; // 1.0
        
        // Venue 0: lower price
        engine.update_price(0, 0, base * 50000, base, base * 100, 1000);
        
        // Venue 1: higher price (arb opportunity)
        engine.update_price(1, 0, base * 50100, base, base * 100, 1000);
        
        // Set low latencies
        engine.set_venue_latency(0, 100_000); // 100us
        engine.set_venue_latency(1, 100_000);
        
        let found = engine.find_arbitrage(0, 2000);
        
        // Should find opportunity (20 bp profit before latency penalty)
        assert!(found || engine.best_opportunity.profit_bp_i64 > 0);
    }
}
