//! Cash-and-Carry and Funding-Rate Arbitrage Signal Generator
//! 
//! Implements funding rate arbitrage detection for perpetual futures vs spot.
//! Uses fixed-point arithmetic and lock-free state management for microsecond execution.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

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

/// Funding arbitrage opportunity - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct FundingArbOpportunity {
    /// Asset identifier
    pub asset_id: u64,
    /// Spot price (Q32.32)
    pub spot_price_i64: i64,
    /// Perp price (Q32.32)
    pub perp_price_i64: i64,
    /// Annualized funding rate (Q16.16, basis points)
    pub funding_rate_i32: i32,
    /// Expected return (Q16.16, annualized %)
    pub expected_return_i32: i32,
    /// Signal: 0=none, 1=long spot/short perp, 2=short spot/long perp
    pub signal: u8,
    /// Confidence score (0-255)
    pub confidence: u8,
    /// Timestamp (TSC cycles)
    pub timestamp_tsc: u64,
    _padding: [u8; 34],
}

const _: () = assert!(core::mem::size_of::<FundingArbOpportunity>() == 64);

impl FundingArbOpportunity {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            asset_id: 0,
            spot_price_i64: 0,
            perp_price_i64: 0,
            funding_rate_i32: 0,
            expected_return_i32: 0,
            signal: 0,
            confidence: 0,
            timestamp_tsc: 0,
            _padding: [0u8; 34],
        }
    }
    
    #[inline(always)]
    pub fn spot_price(&self) -> f64 {
        self.spot_price_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn perp_price(&self) -> f64 {
        self.perp_price_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn funding_rate(&self) -> f64 {
        self.funding_rate_i32 as f64 / 65536.0
    }
    
    #[inline(always)]
    pub fn expected_return(&self) -> f64 {
        self.expected_return_i32 as f64 / 65536.0
    }
}

/// Cash-and-carry arbitrage calculator
#[repr(C, align(64))]
pub struct FundingArbCalculator {
    /// Minimum funding rate threshold for entry (Q16.16)
    min_funding_threshold: i32,
    /// Transaction cost estimate (Q16.16)
    tx_cost: i32,
    /// Position size limit (in base units, Q32.32)
    position_limit: i64,
    /// Current opportunity
    pub opportunity: FundingArbOpportunity,
    /// Circuit breaker
    pub circuit_breaker: PaddedAtomicBool,
    /// Cumulative P&L (Q32.32)
    pub cumulative_pnl: AtomicI64,
    /// Trade count
    pub trade_count: PaddedAtomicU64,
    _padding: [u8; 40],
}

// Safety: Single-threaded in hot path
unsafe impl Send for FundingArbCalculator {}
unsafe impl Sync for FundingArbCalculator {}

impl FundingArbCalculator {
    #[inline(always)]
    pub const fn new(min_funding: f64, tx_cost: f64, position_limit: f64) -> Self {
        Self {
            min_funding_threshold: (min_funding * 65536.0) as i32,
            tx_cost: (tx_cost * 65536.0) as i32,
            position_limit: (position_limit * 4294967296.0) as i64,
            opportunity: FundingArbOpportunity::new(),
            circuit_breaker: PaddedAtomicBool::new(false),
            cumulative_pnl: AtomicI64::new(0),
            trade_count: PaddedAtomicU64::new(0),
            _padding: [0u8; 40],
        }
    }
    
    /// Evaluate funding arbitrage opportunity
    /// Returns signal: 0=none, 1=long spot/short perp, 2=short spot/long perp
    #[inline(always)]
    pub fn evaluate(
        &mut self,
        asset_id: u64,
        spot_price: i64,
        perp_price: i64,
        funding_rate: i32,
        timestamp_tsc: u64,
    ) -> u8 {
        if self.circuit_breaker.get() {
            return 0;
        }
        
        // Calculate basis: perp - spot (Q32.32)
        let basis = perp_price - spot_price;
        
        // Calculate implied funding from basis
        // Annualized: (basis / spot) * 365 * 3 (assuming 8hr funding)
        // Simplified branchless calculation
        let spot_nonzero = spot_price.abs().max(1);
        let basis_pct = ((basis << 16) / spot_nonzero) as i32;
        
        // Annualized return approximation (365 days * 3 funding periods/day)
        let annualized_return = basis_pct * 1095;
        
        // Net return after costs
        let net_return = annualized_return - (self.tx_cost << 4);
        
        // Determine signal based on funding rate direction
        // Positive funding: long spot, short perp (collect funding)
        // Negative funding: short spot, long perp (collect funding)
        let abs_funding = funding_rate.abs();
        let above_threshold = (abs_funding >= self.min_funding_threshold.abs()) as u8;
        
        let long_spot_short_perp = ((funding_rate > self.min_funding_threshold) as u8) * above_threshold;
        let short_spot_long_perp = ((funding_rate < -self.min_funding_threshold) as u8) * above_threshold;
        
        let signal = long_spot_short_perp * 1 + short_spot_long_perp * 2;
        
        // Update opportunity state
        self.opportunity.asset_id = asset_id;
        self.opportunity.spot_price_i64 = spot_price;
        self.opportunity.perp_price_i64 = perp_price;
        self.opportunity.funding_rate_i32 = funding_rate;
        self.opportunity.expected_return_i32 = net_return >> 16;
        self.opportunity.signal = signal;
        self.opportunity.confidence = Self::calculate_confidence(abs_funding, self.min_funding_threshold as u32);
        self.opportunity.timestamp_tsc = timestamp_tsc;
        
        signal
    }
    
    /// Calculate confidence score (0-255) based on funding rate magnitude
    #[inline(always)]
    fn calculate_confidence(funding_abs: u32, threshold: u32) -> u8 {
        if threshold == 0 {
            return 255;
        }
        
        // Branchless confidence calculation
        let ratio = (funding_abs * 256) / threshold;
        ratio.min(255) as u8
    }
    
    /// Record fill and update P&L
    #[inline(always)]
    pub fn record_fill(&self, pnl_q32: i64) {
        self.cumulative_pnl.fetch_add(pnl_q32, Ordering::Relaxed);
        self.trade_count.value.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get cumulative P&L as f64
    #[inline(always)]
    pub fn cumulative_pnl_f64(&self) -> f64 {
        self.cumulative_pnl.load(Ordering::Relaxed) as f64 / 4294967296.0
    }
    
    /// Trigger circuit breaker
    #[inline(always)]
    pub fn halt(&self) {
        self.circuit_breaker.set(true);
    }
    
    /// Reset circuit breaker
    #[inline(always)]
    pub fn resume(&self) {
        self.circuit_breaker.set(false);
    }
}

/// Multi-asset funding arbitrage engine
#[repr(C, align(64))]
pub struct FundingArbEngine<const N: usize> {
    /// Per-asset calculators
    calculators: [FundingArbCalculator; N],
    /// Active asset bitmask
    pub active_assets: AtomicU64,
    /// Best current opportunity index
    pub best_opportunity_idx: AtomicU64,
    /// Global circuit breaker
    pub global_halt: PaddedAtomicBool,
    _padding: [u8; 48],
}

impl<const N: usize> FundingArbEngine<N> {
    #[inline(always)]
    pub const fn new(min_funding: f64, tx_cost: f64, position_limit: f64) -> Self {
        // Initialize all calculators with same parameters
        const INIT: FundingArbCalculator = FundingArbCalculator::new(0.0, 0.0, 0.0);
        Self {
            calculators: [INIT; N],
            active_assets: AtomicU64::new(u64::MAX),
            best_opportunity_idx: AtomicU64::new(0),
            global_halt: PaddedAtomicBool::new(false),
            _padding: [0u8; 48],
        }
    }
    
    /// Update a specific asset's funding data
    #[inline(always)]
    pub fn update_asset(
        &mut self,
        asset_idx: usize,
        asset_id: u64,
        spot: i64,
        perp: i64,
        funding: i32,
        timestamp: u64,
    ) -> u8 {
        if self.global_halt.get() {
            return 0;
        }
        
        if asset_idx >= N {
            return 0;
        }
        
        self.calculators[asset_idx].evaluate(asset_id, spot, perp, funding, timestamp)
    }
    
    /// Find best opportunity across all assets
    #[inline(always)]
    pub fn find_best_opportunity(&self) -> Option<(usize, f64)> {
        if self.global_halt.get() {
            return None;
        }
        
        let mut best_idx = 0;
        let mut best_return = 0.0;
        
        // Manual loop unrolling for performance
        let step = 4;
        let mut i = 0;
        
        while i + step <= N {
            for offset in 0..step {
                let idx = i + offset;
                let ret = self.calculators[idx].opportunity.expected_return();
                if ret > best_return {
                    best_return = ret;
                    best_idx = idx;
                }
            }
            i += step;
        }
        
        // Handle remaining
        while i < N {
            let ret = self.calculators[i].opportunity.expected_return();
            if ret > best_return {
                best_return = ret;
                best_idx = i;
            }
            i += 1;
        }
        
        if best_return > 0.0 {
            Some((best_idx, best_return))
        } else {
            None
        }
    }
    
    /// Halt all trading
    #[inline(always)]
    pub fn global_halt(&self) {
        self.global_halt.set(true);
        for calc in &self.calculators {
            calc.halt();
        }
    }
    
    /// Resume trading
    #[inline(always)]
    pub fn global_resume(&self) {
        self.global_halt.set(false);
        for calc in &self.calculators {
            calc.resume();
        }
    }
}

// Compile-time assertion
const _: () = assert!(core::mem::size_of::<FundingArbEngine<4>>() % 64 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_opportunity_size() {
        assert_eq!(core::mem::size_of::<FundingArbOpportunity>(), 64);
    }
    
    #[test]
    fn test_funding_arb_evaluation() {
        let mut calc = FundingArbCalculator::new(0.01, 0.001, 1.0);
        
        let spot = 4294967296i64; // 1.0
        let perp = 4337916969i64; // 1.01
        let funding = 655360i32; // 10.0%
        
        let signal = calc.evaluate(0x4254430000000000, spot, perp, funding, 1000);
        
        // Should generate long spot/short perp signal
        assert_eq!(signal, 1);
        assert!(calc.opportunity.confidence > 0);
    }
}
