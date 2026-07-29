//! Real-time stablecoin premium/discount monitor with automated collateral haircuts.
//! 
//! Uses fixed-point arithmetic, circuit breakers for extreme depeg events,
//! and lock-free atomic state management for instant response.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Fixed-point scale (6 decimal precision)
const FIXED_SCALE: u64 = 1_000_000;

/// Peg threshold (1% deviation triggers alert)
const PEG_THRESHOLD_BPS: u64 = 100; // 100 basis points = 1%

/// Circuit breaker threshold (5% depeg halts trading)
const CIRCUIT_BREAKER_THRESHOLD_BPS: u64 = 500; // 500 bps = 5%

/// Padded atomic u64 for cache-line alignment
#[repr(C)]
struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 8],
}

impl PaddedAtomicU64 {
    const fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; CACHE_LINE_SIZE - 8],
        }
    }
    
    #[inline]
    fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline]
    fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed);
    }
}

/// Price feed data - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct PriceFeed {
    /// Feed ID hash
    feed_id: u64,
    /// Token type (0=USDT, 1=USDC, 2=DAI, etc.)
    token_type: u8,
    /// Exchange/venue ID
    venue_id: u8,
    /// Padding
    _pad1: [u8; 6],
    /// Current price (fixed-point, scaled by 1e6)
    price_fixed: u64,
    /// Bid price
    bid_fixed: u64,
    /// Ask price
    ask_fixed: u64,
    /// 24h volume (fixed-point)
    volume_24h_fixed: u64,
    /// Last update timestamp (cycles)
    last_update_cycles: u64,
    /// Is stale
    is_stale: bool,
    /// Padding
    _padding: [u8; 23],
}

const _: () = assert!(core::mem::size_of::<PriceFeed>() == 64);

/// Collateral haircut configuration - cache-line aligned
#[repr(C)]
struct HaircutConfig {
    /// Token type
    token_type: u8,
    /// Padding
    _pad1: [u8; 7],
    /// Base haircut (bps, scaled by 100)
    base_haircut_bps: u64,
    /// Max haircut (bps)
    max_haircut_bps: u64,
    /// Haircut increment per bps deviation
    increment_per_bps: u64,
    /// Depeg trigger threshold (bps)
    depeg_trigger_bps: u64,
}

impl HaircutConfig {
    const fn new(token_type: u8) -> Self {
        Self {
            token_type,
            _pad1: [0u8; 7],
            base_haircut_bps: 100, // 1% base
            max_haircut_bps: 5000, // 50% max
            increment_per_bps: 100, // 1:1 ratio
            depeg_trigger_bps: 200, // 2% triggers increased haircut
        }
    }
}

/// Main depeg safeguard monitor
#[repr(C)]
pub struct StablecoinDepegMonitor {
    /// Price feeds (pre-allocated)
    feeds: [PriceFeed; 64],
    /// Feed count
    feed_count: AtomicU64,
    /// Aggregate prices by token type
    aggregate_prices: [PaddedAtomicU64; 8],
    /// Premium/discount in bps (signed, stored as i64)
    peg_deviation_bps: [AtomicI64; 8],
    /// Haircut configs
    haircut_configs: [HaircutConfig; 8],
    /// Current haircuts applied (bps)
    current_haircuts_bps: [AtomicU64; 8],
    /// Circuit breaker triggered
    circuit_breaker: AtomicBool,
    /// Trading halted flag
    trading_halted: AtomicBool,
    /// Alert level (0=normal, 1=warning, 2=critical)
    alert_level: AtomicU64,
}

impl StablecoinDepegMonitor {
    /// Create a new depeg monitor
    pub const fn new() -> Self {
        Self {
            feeds: [PriceFeed {
                feed_id: 0,
                token_type: 0,
                venue_id: 0,
                _pad1: [0u8; 6],
                price_fixed: 0,
                bid_fixed: 0,
                ask_fixed: 0,
                volume_24h_fixed: 0,
                last_update_cycles: 0,
                is_stale: false,
                _padding: [0u8; 23],
            }; 64],
            feed_count: AtomicU64::new(0),
            aggregate_prices: [PaddedAtomicU64::new(0); 8],
            peg_deviation_bps: [AtomicI64::new(0); 8],
            haircut_configs: [HaircutConfig::new(0); 8],
            current_haircuts_bps: [AtomicU64::new(0); 8],
            circuit_breaker: AtomicBool::new(false),
            trading_halted: AtomicBool::new(false),
            alert_level: AtomicU64::new(0),
        }
    }
    
    /// Initialize haircut configs
    #[inline]
    pub fn init_haircut_configs(&self) {
        // USDT config
        self.haircut_configs[0] = HaircutConfig::new(0);
        // USDC config
        self.haircut_configs[1] = HaircutConfig::new(1);
        // DAI config
        self.haircut_configs[2] = HaircutConfig::new(2);
    }
    
    /// Register a price feed
    #[inline]
    pub fn register_feed(&self, feed: PriceFeed) -> bool {
        let idx = self.feed_count.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= 64 {
            self.feed_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            *self.feeds.get_unchecked_mut(idx) = feed;
        }
        
        true
    }
    
    /// Update price for a feed
    #[inline]
    pub fn update_price(&self, feed_id: u64, price_fixed: u64, bid_fixed: u64, ask_fixed: u64) {
        for i in 0..self.feed_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let feed = &mut *self.feeds.get_unchecked_mut(i);
                if feed.feed_id == feed_id {
                    feed.price_fixed = price_fixed;
                    feed.bid_fixed = bid_fixed;
                    feed.ask_fixed = ask_fixed;
                    
                    use core::arch::x86_64::_rdtsc;
                    feed.last_update_cycles = unsafe { _rdtsc() };
                    feed.is_stale = false;
                    
                    break;
                }
            }
        }
        
        // Recalculate aggregate and deviation
        self.recalculate_aggregate(feed_id % 8);
    }
    
    /// Recalculate aggregate price for a token type
    #[inline]
    fn recalculate_aggregate(&self, token_type: usize) {
        let mut sum = 0u64;
        let mut count = 0u64;
        
        for i in 0..self.feed_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let feed = *self.feeds.get_unchecked(i);
                if feed.token_type as usize == token_type && !feed.is_stale {
                    sum += feed.price_fixed;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            let avg = sum / count;
            self.aggregate_prices[token_type].store(avg);
            
            // Calculate deviation from peg (1.0 = 1_000_000 in fixed-point)
            let peg_fixed = FIXED_SCALE;
            let deviation = if avg >= peg_fixed {
                ((avg - peg_fixed) * 10000 / peg_fixed) as i64
            } else {
                -((peg_fixed - avg) * 10000 / peg_fixed) as i64
            };
            
            self.peg_deviation_bps[token_type].store(deviation, Ordering::Relaxed);
            
            // Update haircut based on deviation
            self.update_haircut(token_type, deviation.unsigned_abs());
            
            // Check circuit breaker (branchless)
            self.check_circuit_breaker(token_type, deviation.unsigned_abs());
        }
    }
    
    /// Update haircut based on deviation
    #[inline]
    fn update_haircut(&self, token_type: usize, deviation_bps: u64) {
        let config = unsafe { *self.haircut_configs.get_unchecked(token_type) };
        
        // Branchless haircut calculation
        let exceeds_trigger = (deviation_bps >= config.depeg_trigger_bps) as u64;
        let excess_bps = deviation_bps.saturating_sub(config.depeg_trigger_bps);
        let additional_haircut = excess_bps * config.increment_per_bps;
        
        let calculated_haircut = config.base_haircut_bps + (additional_haircut * exceeds_trigger);
        let final_haircut = calculated_haircut.min(config.max_haircut_bps);
        
        self.current_haircuts_bps[token_type].store(final_haircut, Ordering::Relaxed);
    }
    
    /// Check circuit breaker conditions
    #[inline]
    fn check_circuit_breaker(&self, token_type: usize, deviation_bps: u64) {
        // Circuit breaker triggers at 5% depeg
        if deviation_bps >= CIRCUIT_BREAKER_THRESHOLD_BPS {
            self.circuit_breaker.store(true, Ordering::Relaxed);
            self.trading_halted.store(true, Ordering::Relaxed);
            self.alert_level.store(2, Ordering::Relaxed); // Critical
        } else if deviation_bps >= PEG_THRESHOLD_BPS * 2 {
            self.alert_level.store(1, Ordering::Relaxed); // Warning
        } else {
            self.alert_level.store(0, Ordering::Relaxed); // Normal
        }
    }
    
    /// Get current aggregate price for a token
    #[inline]
    pub fn get_aggregate_price(&self, token_type: usize) -> u64 {
        if token_type >= 8 { return 0; }
        self.aggregate_prices[token_type].load()
    }
    
    /// Get peg deviation in bps
    #[inline]
    pub fn get_peg_deviation_bps(&self, token_type: usize) -> i64 {
        if token_type >= 8 { return 0; }
        self.peg_deviation_bps[token_type].load(Ordering::Relaxed)
    }
    
    /// Get current haircut for a token
    #[inline]
    pub fn get_current_haircut_bps(&self, token_type: usize) -> u64 {
        if token_type >= 8 { return 0; }
        self.current_haircuts_bps[token_type].load(Ordering::Relaxed)
    }
    
    /// Check if circuit breaker is triggered
    #[inline]
    pub fn is_circuit_breaker_triggered(&self) -> bool {
        self.circuit_breaker.load(Ordering::Relaxed)
    }
    
    /// Check if trading is halted
    #[inline]
    pub fn is_trading_halted(&self) -> bool {
        self.trading_halted.load(Ordering::Relaxed)
    }
    
    /// Get alert level
    #[inline]
    pub fn get_alert_level(&self) -> u64 {
        self.alert_level.load(Ordering::Relaxed)
    }
    
    /// Reset circuit breaker (manual override)
    #[inline]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
        self.trading_halted.store(false, Ordering::Relaxed);
    }
    
    /// Apply haircut to amount
    #[inline]
    pub fn apply_haircut(&self, token_type: usize, amount_fixed: u64) -> u64 {
        let haircut_bps = self.get_current_haircut_bps(token_type);
        // amount * (10000 - haircut_bps) / 10000
        amount_fixed * (10000 - haircut_bps) / 10000
    }
}

impl Default for StablecoinDepegMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normal_peg() {
        let monitor = StablecoinDepegMonitor::new();
        monitor.init_haircut_configs();
        
        // Normal price (at peg)
        let feed = PriceFeed {
            feed_id: 1,
            token_type: 0,
            venue_id: 1,
            _pad1: [0u8; 6],
            price_fixed: FIXED_SCALE, // Exactly 1.0
            bid_fixed: 999900,
            ask_fixed: 1000100,
            volume_24h_fixed: 1_000_000_000_000,
            last_update_cycles: 0,
            is_stale: false,
            _padding: [0u8; 23],
        };
        
        monitor.register_feed(feed);
        monitor.recalculate_aggregate(0);
        
        assert_eq!(monitor.get_peg_deviation_bps(0), 0);
        assert!(!monitor.is_circuit_breaker_triggered());
        assert_eq!(monitor.get_alert_level(), 0);
    }
    
    #[test]
    fn test_depeg_detection() {
        let monitor = StablecoinDepegMonitor::new();
        monitor.init_haircut_configs();
        
        // Depegged price (5% below peg)
        let feed = PriceFeed {
            feed_id: 1,
            token_type: 0,
            venue_id: 1,
            _pad1: [0u8; 6],
            price_fixed: 950_000, // 0.95
            bid_fixed: 949900,
            ask_fixed: 950100,
            volume_24h_fixed: 1_000_000_000_000,
            last_update_cycles: 0,
            is_stale: false,
            _padding: [0u8; 23],
        };
        
        monitor.register_feed(feed);
        monitor.recalculate_aggregate(0);
        
        let deviation = monitor.get_peg_deviation_bps(0);
        assert!(deviation < -400); // Should be around -500 bps
        
        // Circuit breaker should trigger at 5%
        assert!(monitor.is_circuit_breaker_triggered());
        assert!(monitor.is_trading_halted());
    }
    
    #[test]
    fn test_haircut_application() {
        let monitor = StablecoinDepegMonitor::new();
        monitor.init_haircut_configs();
        
        // Set a manual haircut
        monitor.current_haircuts_bps[0].store(200, Ordering::Relaxed); // 2%
        
        let amount = 1_000_000_000; // 1000 units
        let after_haircut = monitor.apply_haircut(0, amount);
        
        // Should be reduced by 2%
        assert_eq!(after_haircut, 980_000_000);
    }
}
