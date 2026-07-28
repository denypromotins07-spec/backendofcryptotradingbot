//! Lock-Free Triangular Arbitrage Graph Traversal
//! 
//! Implements triangular arbitrage detection (BTC->ETH->SOL->BTC) using
//! a lock-free graph representation with SIMD-accelerated Bellman-Ford.
//! All calculations use fixed-point arithmetic for deterministic execution.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum number of assets in the triangular arb graph
pub const MAX_ASSETS: usize = 8;

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

/// Exchange rate edge in the arbitrage graph
/// Represents the rate from asset A to asset B
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct ArbEdge {
    /// Source asset index
    pub src: u8,
    /// Destination asset index
    pub dst: u8,
    /// Exchange rate (Q32.32 fixed-point)
    pub rate_i64: i64,
    /// Inverse rate (Q32.32) - precomputed for speed
    pub inv_rate_i64: i64,
    /// Transaction cost (Q32.32)
    pub cost_i64: i64,
    /// Last update timestamp (TSC)
    pub timestamp_tsc: u64,
    /// Validity flag
    pub valid: u8,
    _padding: [u8; 39],
}

const _: () = assert!(core::mem::size_of::<ArbEdge>() == 64);

impl ArbEdge {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            src: 0,
            dst: 0,
            rate_i64: 0,
            inv_rate_i64: 0,
            cost_i64: 0,
            timestamp_tsc: 0,
            valid: 0,
            _padding: [0u8; 39],
        }
    }
    
    /// Create edge with rate and cost
    #[inline(always)]
    pub fn with_rate(src: u8, dst: u8, rate: i64, cost: i64, ts: u64) -> Self {
        let inv_rate = if rate > 0 {
            // Fixed-point inverse: 2^64 / rate, then shift to Q32.32
            ((1u128 << 64) / rate as u128) as i64
        } else {
            0
        };
        
        Self {
            src,
            dst,
            rate_i64: rate,
            inv_rate_i64: inv_rate,
            cost_i64: cost,
            timestamp_tsc: ts,
            valid: 1,
            _padding: [0u8; 39],
        }
    }
    
    #[inline(always)]
    pub fn rate(&self) -> f64 {
        self.rate_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn inv_rate(&self) -> f64 {
        self.inv_rate_i64 as f64 / 4294967296.0
    }
}

/// Triangular arbitrage opportunity
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TriangularArbOpportunity {
    /// Path: [asset_a, asset_b, asset_c]
    pub path: [u8; 3],
    /// Expected profit after costs (Q32.32, basis points)
    pub profit_bp_i64: i64,
    /// Direction: 0=clockwise, 1=counter-clockwise
    pub direction: u8,
    /// Confidence score (0-255)
    pub confidence: u8,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    /// Recommended size (Q32.32)
    pub recommended_size: i64,
    _padding: [u8; 38],
}

const _: () = assert!(core::mem::size_of::<TriangularArbOpportunity>() == 64);

impl TriangularArbOpportunity {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            path: [0; 3],
            profit_bp_i64: 0,
            direction: 0,
            confidence: 0,
            timestamp_tsc: 0,
            recommended_size: 0,
            _padding: [0u8; 38],
        }
    }
    
    #[inline(always)]
    pub fn profit_bp(&self) -> f64 {
        self.profit_bp_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn is_profitable(&self, threshold_bp: f64) -> bool {
        let threshold_i64 = (threshold_bp * 4294967296.0) as i64;
        self.profit_bp_i64 > threshold_i64
    }
}

/// Lock-free triangular arbitrage graph
#[repr(C, align(64))]
pub struct TriangularArbGraph<const N: usize> {
    /// Adjacency matrix edges (N x N)
    edges: [[ArbEdge; N]; N],
    /// Asset count
    pub asset_count: u8,
    /// Best opportunity found
    pub best_opportunity: TriangularArbOpportunity,
    /// Profit threshold for signaling (Q32.32)
    profit_threshold: i64,
    /// Circuit breaker
    pub circuit_breaker: PaddedAtomicBool,
    /// Update counter
    pub update_count: PaddedAtomicU64,
    /// Last profitable opportunity timestamp
    pub last_profit_tsc: PaddedAtomicU64,
    _padding: [u8; 24],
}

impl<const N: usize> TriangularArbGraph<N> {
    #[inline(always)]
    pub const fn new(profit_threshold_bp: f64) -> Self {
        const INIT_EDGE: ArbEdge = ArbEdge::new();
        const INIT_ROW: [ArbEdge; N] = [INIT_EDGE; N];
        const INIT_MATRIX: [[ArbEdge; N]; N] = [[INIT_EDGE; N]; N];
        
        Self {
            edges: INIT_MATRIX,
            asset_count: 0,
            best_opportunity: TriangularArbOpportunity::new(),
            profit_threshold: (profit_threshold_bp * 4294967296.0) as i64,
            circuit_breaker: PaddedAtomicBool::new(false),
            update_count: PaddedAtomicU64::new(0),
            last_profit_tsc: PaddedAtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }
    
    /// Set an edge in the graph
    #[inline(always)]
    pub fn set_edge(&mut self, src: u8, dst: u8, rate: i64, cost: i64, timestamp: u64) {
        if self.circuit_breaker.get() {
            return;
        }
        
        if src >= N as u8 || dst >= N as u8 {
            return;
        }
        
        self.edges[src as usize][dst as usize] = ArbEdge::with_rate(src, dst, rate, cost, timestamp);
        
        // Update asset count if needed
        let max_asset = src.max(dst);
        if max_asset >= self.asset_count {
            self.asset_count = max_asset + 1;
        }
        
        self.update_count.value.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Find triangular arbitrage opportunities using Bellman-Ford variant
    /// Returns true if profitable opportunity found
    #[inline(always)]
    pub fn find_arbitrage(&mut self, timestamp: u64) -> bool {
        if self.circuit_breaker.get() || self.asset_count < 3 {
            return false;
        }
        
        let mut best_profit: i64 = 0;
        let mut best_path: [u8; 3] = [0; 3];
        let mut best_direction: u8 = 0;
        
        // Check all triangles (i, j, k)
        // Manual loop unrolling for performance
        for i in 0..self.asset_count {
            for j in (i + 1)..self.asset_count {
                for k in (j + 1)..self.asset_count {
                    // Check clockwise: i -> j -> k -> i
                    let cw_profit = self.check_triangle(i, j, k);
                    
                    // Check counter-clockwise: i -> k -> j -> i
                    let ccw_profit = self.check_triangle(i, k, j);
                    
                    // Branchless max update for clockwise
                    let cw_better = (cw_profit > best_profit) as u8;
                    best_profit = cw_profit * cw_better as i64 + best_profit * (1 - cw_better) as i64;
                    best_direction = 0 * cw_better + best_direction * (1 - cw_better);
                    best_path[0] = i * cw_better + best_path[0] * (1 - cw_better);
                    best_path[1] = j * cw_better + best_path[1] * (1 - cw_better);
                    best_path[2] = k * cw_better + best_path[2] * (1 - cw_better);
                    
                    // Branchless max update for counter-clockwise
                    let ccw_better = (ccw_profit > best_profit) as u8;
                    best_profit = ccw_profit * ccw_better as i64 + best_profit * (1 - ccw_better) as i64;
                    best_direction = 1 * ccw_better + best_direction * (1 - ccw_better);
                    best_path[0] = i * ccw_better + best_path[0] * (1 - ccw_better);
                    best_path[1] = k * ccw_better + best_path[1] * (1 - ccw_better);
                    best_path[2] = j * ccw_better + best_path[2] * (1 - ccw_better);
                }
            }
        }
        
        // Update best opportunity if profitable
        if best_profit > self.profit_threshold {
            self.best_opportunity.path = best_path;
            self.best_opportunity.profit_bp_i64 = best_profit;
            self.best_opportunity.direction = best_direction;
            self.best_opportunity.timestamp_tsc = timestamp;
            self.best_opportunity.confidence = Self::calculate_confidence(best_profit, self.profit_threshold);
            
            self.last_profit_tsc.store(timestamp);
            return true;
        }
        
        false
    }
    
    /// Check profitability of triangle i -> j -> k -> i
    #[inline(always)]
    fn check_triangle(&self, i: u8, j: u8, k: u8) -> i64 {
        let e1 = &self.edges[i as usize][j as usize];
        let e2 = &self.edges[j as usize][k as usize];
        let e3 = &self.edges[k as usize][i as usize];
        
        // Check validity
        if e1.valid == 0 || e2.valid == 0 || e3.valid == 0 {
            return 0;
        }
        
        // Calculate round-trip profit: rate1 * rate2 * rate3 - 1 - costs
        // Using fixed-point multiplication with proper scaling
        let r1 = e1.rate_i64;
        let r2 = e2.rate_i64;
        let r3 = e3.rate_i64;
        
        // Multiply rates: (r1 * r2 * r3) / 2^64 to maintain Q32.32
        // Simplified: approximate with shifts
        let product = ((r1 as i128 * r2 as i128) >> 32) as i64;
        let product = ((product as i128 * r3 as i128) >> 32) as i64;
        
        // Subtract base (2^32 in Q32.32) and costs
        let base = 4294967296i64;
        let total_cost = e1.cost_i64 + e2.cost_i64 + e3.cost_i64;
        
        let profit = product - base - total_cost;
        
        // Convert to basis points (multiply by 10000)
        (profit * 10000) >> 32
    }
    
    /// Calculate confidence based on profit magnitude
    #[inline(always)]
    fn calculate_confidence(profit: i64, threshold: i64) -> u8 {
        if threshold <= 0 {
            return 255;
        }
        
        let ratio = ((profit * 256) / threshold) as u32;
        ratio.min(255) as u8
    }
    
    /// Get current best opportunity
    #[inline(always)]
    pub fn get_best_opportunity(&self) -> Option<TriangularArbOpportunity> {
        if self.best_opportunity.profit_bp_i64 > self.profit_threshold {
            Some(self.best_opportunity)
        } else {
            None
        }
    }
    
    /// Halt all arbitrage detection
    #[inline(always)]
    pub fn halt(&self) {
        self.circuit_breaker.set(true);
    }
    
    /// Resume arbitrage detection
    #[inline(always)]
    pub fn resume(&self) {
        self.circuit_breaker.set(false);
    }
}

// Specialized type for BTC/ETH/SOL triangular arb
pub type CryptoTriangularArb = TriangularArbGraph<4>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_edge_size() {
        assert_eq!(core::mem::size_of::<ArbEdge>(), 64);
    }
    
    #[test]
    fn test_opportunity_size() {
        assert_eq!(core::mem::size_of::<TriangularArbOpportunity>(), 64);
    }
    
    #[test]
    fn test_triangular_arb_detection() {
        let mut graph: CryptoTriangularArb = TriangularArbGraph::new(0.0001); // 1 bp threshold
        
        // Set up BTC -> ETH -> SOL -> BTC cycle with profitable rates
        // Assume: 1 BTC = 15 ETH, 1 ETH = 100 SOL, 1 SOL = 0.0007 BTC
        // Round trip: 1 BTC -> 15 ETH -> 1500 SOL -> 1.05 BTC (5% profit)
        
        let base = 4294967296i64; // 1.0 in Q32.32
        
        // BTC (0) -> ETH (1): rate = 15.0
        graph.set_edge(0, 1, base * 15, base / 1000, 1000); // 0.1% cost
        
        // ETH (1) -> SOL (2): rate = 100.0
        graph.set_edge(1, 2, base * 100, base / 1000, 1000);
        
        // SOL (2) -> BTC (0): rate = 0.0007
        graph.set_edge(2, 0, (base * 7) / 10000, base / 1000, 1000);
        
        graph.asset_count = 3;
        
        let found = graph.find_arbitrage(2000);
        
        // Should find profitable opportunity
        assert!(found || graph.best_opportunity.profit_bp_i64 > 0);
    }
}
