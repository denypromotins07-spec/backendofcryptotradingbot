//! Inventory-Skewed Avellaneda-Stoikov Market Making Core Logic
//! 
//! Implements the classic AS market making model with inventory risk management.
//! Uses fixed-point arithmetic and lock-free state for microsecond quote updates.

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

/// Market maker quote - single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct MMQuote {
    /// Bid price (Q32.32)
    pub bid_i64: i64,
    /// Ask price (Q32.32)
    pub ask_i64: i64,
    /// Bid size (Q32.32)
    pub bid_size_i64: i64,
    /// Ask size (Q32.32)
    pub ask_size_i64: i64,
    /// Mid price reference (Q32.32)
    pub mid_i64: i64,
    /// Spread (Q32.32)
    pub spread_i64: i64,
    /// Timestamp (TSC)
    pub timestamp_tsc: u64,
    /// Quote flags (bitmask)
    pub flags: u8,
    _padding: [u8; 39],
}

const _: () = assert!(core::mem::size_of::<MMQuote>() == 64);

impl MMQuote {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            bid_i64: 0,
            ask_i64: 0,
            bid_size_i64: 0,
            ask_size_i64: 0,
            mid_i64: 0,
            spread_i64: 0,
            timestamp_tsc: 0,
            flags: 0,
            _padding: [0u8; 39],
        }
    }
    
    #[inline(always)]
    pub fn bid(&self) -> f64 {
        self.bid_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn ask(&self) -> f64 {
        self.ask_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn mid(&self) -> f64 {
        self.mid_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn spread(&self) -> f64 {
        self.spread_i64 as f64 / 4294967296.0
    }
}

/// Avellaneda-Stoikov market maker parameters
#[repr(C, align(64))]
pub struct ASParameters {
    /// Risk aversion coefficient (gamma) in Q16.16
    pub gamma: i32,
    /// Volatility estimate (sigma) in Q32.32 (annualized)
    pub sigma: i64,
    /// Order arrival rate (kappa) in Q16.16
    pub kappa: i32,
    /// Time horizon T in Q16.16 (fraction of day)
    pub time_horizon: i32,
    /// Max inventory position (units)
    pub max_inventory: i32,
    /// Min spread (bps) in Q16.16
    pub min_spread_bp: i32,
    /// Max spread (bps) in Q16.16
    pub max_spread_bp: i32,
    /// Inventory skew factor in Q16.16
    pub inventory_skew: i32,
    _padding: [u8; 32],
}

impl ASParameters {
    #[inline(always)]
    pub const fn new(
        gamma: f64,
        sigma: f64,
        kappa: f64,
        time_horizon: f64,
        max_inv: i32,
        min_spread: f64,
        max_spread: f64,
        inv_skew: f64,
    ) -> Self {
        Self {
            gamma: (gamma * 65536.0) as i32,
            sigma: (sigma * 4294967296.0) as i64,
            kappa: (kappa * 65536.0) as i32,
            time_horizon: (time_horizon * 65536.0) as i32,
            max_inventory: max_inv,
            min_spread_bp: (min_spread * 65536.0) as i32,
            max_spread_bp: (max_spread * 65536.0) as i32,
            inventory_skew: (inv_skew * 65536.0) as i32,
            _padding: [0u8; 32],
        }
    }
}

/// Avellaneda-Stoikov market maker core
#[repr(C, align(64))]
pub struct AvellanedaStoikovMM {
    /// Model parameters
    pub params: ASParameters,
    /// Current inventory (signed, in base units)
    pub inventory: AtomicI64,
    /// Current mid price (Q32.32)
    pub mid_price: AtomicI64,
    /// Current quote
    pub quote: MMQuote,
    /// Cumulative P&L (Q32.32)
    pub pnl: AtomicI64,
    /// Fill count
    pub fill_count: PaddedAtomicU64,
    /// Circuit breaker
    pub circuit_breaker: PaddedAtomicBool,
    /// Adverse selection toxicity score (Q16.16)
    pub toxicity: i32,
    _padding: [u8; 44],
}

// Safety: Single-threaded in hot path
unsafe impl Send for AvellanedaStoikovMM {}
unsafe impl Sync for AvellanedaStoikovMM {}

impl AvellanedaStoikovMM {
    #[inline(always)]
    pub const fn new(params: ASParameters) -> Self {
        Self {
            params,
            inventory: AtomicI64::new(0),
            mid_price: AtomicI64::new(0),
            quote: MMQuote::new(),
            pnl: AtomicI64::new(0),
            fill_count: PaddedAtomicU64::new(0),
            circuit_breaker: PaddedAtomicBool::new(false),
            toxicity: 0,
            _padding: [0u8; 44],
        }
    }
    
    /// Update mid price and recalculate quotes
    #[inline(always)]
    pub fn update_mid(&mut self, mid: i64, timestamp: u64) {
        if self.circuit_breaker.get() {
            return;
        }
        
        self.mid_price.store(mid, Ordering::Relaxed);
        self.quote.mid_i64 = mid;
        
        // Calculate optimal spread using AS formula
        // spread = 2/gamma * ln(1 + gamma/kappa)
        let gamma = self.params.gamma as i64;
        let kappa = self.params.kappa as i64;
        
        // Simplified spread calculation (avoiding log for speed)
        // spread_bp ≈ 2 * sigma * sqrt(T) + inventory_skew * inventory
        let sigma = self.params.sigma;
        let t = self.params.time_horizon as i64;
        
        // Base spread: 2 * sigma * sqrt(T) (simplified)
        let base_spread = (sigma * t) >> 24; // Approximate sqrt and scale
        
        // Inventory adjustment
        let inv = self.inventory.load(Ordering::Relaxed);
        let inv_skew = (inv * self.params.inventory_skew as i64) >> 16;
        
        // Total spread in Q32.32
        let mut spread = base_spread + inv_skew.abs();
        
        // Apply min/max bounds (branchless)
        let min_spread = (self.params.min_spread_bp as i64 * mid) >> 16;
        let max_spread = (self.params.max_spread_bp as i64 * mid) >> 16;
        
        spread = spread.max(min_spread).min(max_spread);
        
        self.quote.spread_i64 = spread;
        
        // Calculate bid/ask with inventory skew
        // If long inventory: skew quotes down to encourage sells
        // If short inventory: skew quotes up to encourage buys
        let half_spread = spread >> 1;
        let skew = (inv * self.params.inventory_skew as i64 * mid) >> 32;
        
        self.quote.bid_i64 = mid - half_spread - skew;
        self.quote.ask_i64 = mid + half_spread - skew;
        self.quote.timestamp_tsc = timestamp;
        
        // Set quote valid flag
        self.quote.flags |= 0x01;
    }
    
    /// Record a fill and update inventory/P&L
    #[inline(always)]
    pub fn record_fill(&mut self, is_buy: bool, price: i64, size: i64) {
        let current_inv = self.inventory.fetch_add(
            if is_buy { -size } else { size },
            Ordering::Relaxed,
        );
        
        // Update P&L
        let pnl_impact = if is_buy {
            // Buying at price, mark to mid
            (self.mid_price.load(Ordering::Relaxed) - price) * size
        } else {
            // Selling at price, mark to mid  
            (price - self.mid_price.load(Ordering::Relaxed)) * size
        };
        
        self.pnl.fetch_add(pnl_impact, Ordering::Relaxed);
        self.fill_count.value.fetch_add(1, Ordering::Relaxed);
        
        // Update toxicity based on fill direction vs price movement
        self.update_toxicity(is_buy, price);
    }
    
    /// Update adverse selection toxicity score
    #[inline(always)]
    fn update_toxicity(&mut self, was_buy: bool, fill_price: i64) {
        let mid = self.mid_price.load(Ordering::Relaxed);
        
        // If we bought and price went down, or sold and price went up = adverse
        // Simplified: compare fill price to current mid
        let adverse = if was_buy {
            fill_price > mid
        } else {
            fill_price < mid
        };
        
        // Exponential moving average update (branchless)
        let alpha = 4096; // 1/16 in Q12.4
        let penalty = if adverse { 65536 } else { 0 };
        
        self.toxicity = ((alpha * penalty) + ((16384 - alpha) * self.toxicity as i32)) >> 14;
        
        // Increase spread if toxicity is high
        if self.toxicity > 32768 {
            self.params.min_spread_bp = (self.params.min_spread_bp as i32 * 1.5) as i32;
        }
    }
    
    /// Get current quote
    #[inline(always)]
    pub fn get_quote(&self) -> MMQuote {
        self.quote
    }
    
    /// Get current inventory
    #[inline(always)]
    pub fn inventory(&self) -> i64 {
        self.inventory.load(Ordering::Relaxed)
    }
    
    /// Get P&L as f64
    #[inline(always)]
    pub fn pnl_f64(&self) -> f64 {
        self.pnl.load(Ordering::Relaxed) as f64 / 4294967296.0
    }
    
    /// Flatten inventory (emergency close)
    #[inline(always)]
    pub fn flatten(&self) -> i64 {
        let inv = self.inventory.swap(0, Ordering::Relaxed);
        inv
    }
    
    /// Halt market making
    #[inline(always)]
    pub fn halt(&self) {
        self.circuit_breaker.set(true);
        self.quote.flags &= !0x01; // Invalidate quote
    }
    
    /// Resume market making
    #[inline(always)]
    pub fn resume(&self) {
        self.circuit_breaker.set(false);
    }
    
    /// Update volatility estimate
    #[inline(always)]
    pub fn update_volatility(&mut self, vol: f64) {
        self.params.sigma = (vol * 4294967296.0) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quote_size() {
        assert_eq!(core::mem::size_of::<MMQuote>(), 64);
    }
    
    #[test]
    fn test_as_market_maker() {
        let params = ASParameters::new(
            0.1,   // gamma
            0.02,  // sigma (2% daily vol)
            1.0,   // kappa
            0.01,  // time horizon (1% of day ~15min)
            100,   // max inventory
            0.05,  // min spread 5bp
            0.50,  // max spread 50bp
            0.01,  // inventory skew
        );
        
        let mut mm = AvellanedaStoikovMM::new(params);
        
        let mid = 4294967296i64 * 50000; // $50,000 in Q32.32
        mm.update_mid(mid, 1000);
        
        let quote = mm.get_quote();
        
        // Verify bid < mid < ask
        assert!(quote.bid_i64 < quote.mid_i64);
        assert!(quote.ask_i64 > quote.mid_i64);
        assert!(quote.spread_i64 > 0);
    }
}
