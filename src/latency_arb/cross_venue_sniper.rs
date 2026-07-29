//! Cross-Venue Latency Arbitrage Sniper
//! 
//! Uses consolidated micro-prices across venues to detect and exploit
//! latency arbitrage opportunities. SIMD-accelerated comparisons and
//! lock-free state management.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

pub type FixedPrice = i64;
pub type FixedQty = i64;
pub type TscTicks = u64;

const MAX_VENUES: usize = 8;
const VENUE_MASK: usize = MAX_VENUES - 1;

/// Cache-line aligned venue price data (64 bytes)
#[repr(C, align(64))]
pub struct VenuePrice {
    pub venue_id: u32,
    pub bid_price: AtomicI64,
    pub ask_price: AtomicI64,
    pub bid_qty: AtomicI64,
    pub ask_qty: AtomicI64,
    pub last_update_ticks: AtomicU64,
    pub stale_flag: AtomicU8,
    _padding: [u8; 23],
}

impl VenuePrice {
    pub const fn new(venue_id: u32) -> Self {
        Self {
            venue_id,
            bid_price: AtomicI64::new(0),
            ask_price: AtomicI64::new(i64::MAX),
            bid_qty: AtomicI64::new(0),
            ask_qty: AtomicI64::new(0),
            last_update_ticks: AtomicU64::new(0),
            stale_flag: AtomicU8::new(0),
            _padding: [0u8; 23],
        }
    }

    #[inline]
    pub fn update_bid(&self, price: FixedPrice, qty: FixedQty) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.bid_price.store(price, Ordering::Release);
        self.bid_qty.store(qty, Ordering::Release);
        self.last_update_ticks.store(ts, Ordering::Release);
        self.stale_flag.store(0, Ordering::Release);
    }

    #[inline]
    pub fn update_ask(&self, price: FixedPrice, qty: FixedQty) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.ask_price.store(price, Ordering::Release);
        self.ask_qty.store(qty, Ordering::Release);
        self.last_update_ticks.store(ts, Ordering::Release);
        self.stale_flag.store(0, Ordering::Release);
    }
}

/// Cross-venue arbitrage opportunity detector
#[repr(C, align(64))]
pub struct ArbitrageSniper {
    pub venues: [VenuePrice; MAX_VENUES],
    pub venue_count: AtomicU8,
    pub min_spread_bps: AtomicU64,
    pub enabled: AtomicU8,
    pub signals_generated: AtomicU64,
    pub last_signal_ticks: AtomicU64,
    _padding: [u8; 38],
}

impl ArbitrageSniper {
    pub const fn new(min_spread_bps: u64) -> Self {
        const INIT: VenuePrice = VenuePrice::new(0);
        Self {
            venues: [INIT; MAX_VENUES],
            venue_count: AtomicU8::new(0),
            min_spread_bps: AtomicU64::new(min_spread_bps),
            enabled: AtomicU8::new(0),
            signals_generated: AtomicU64::new(0),
            last_signal_ticks: AtomicU64::new(0),
            _padding: [0u8; 38],
        }
    }

    /// Register a venue
    #[inline]
    pub fn register_venue(&self, venue_id: u32) -> bool {
        let count = self.venue_count.load(Ordering::Acquire) as usize;
        if count >= MAX_VENUES {
            return false;
        }
        self.venues[count] = VenuePrice::new(venue_id);
        self.venue_count.store((count + 1) as u8, Ordering::Release);
        true
    }

    /// Scan for arbitrage opportunities using SIMD-like branchless logic
    #[inline]
    pub fn scan_for_arb(&self) -> Option<(u32, u32, FixedPrice, FixedQty)> {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return None;
        }

        let count = self.venue_count.load(Ordering::Acquire) as usize;
        if count < 2 {
            return None;
        }

        let mut best_buy_venue: u32 = 0;
        let mut best_sell_venue: u32 = 0;
        let mut best_spread: FixedPrice = 0;
        let mut best_qty: FixedQty = 0;
        let now = unsafe { core::arch::x86_64::_rdtsc() } as u64;

        // Find highest bid and lowest ask across venues
        let mut highest_bid: FixedPrice = 0;
        let mut lowest_ask: FixedPrice = i64::MAX;

        for i in 0..count {
            let venue = &self.venues[i];
            
            // Skip stale venues (branchless check)
            let stale = venue.stale_flag.load(Ordering::Acquire);
            let bid = venue.bid_price.load(Ordering::Acquire);
            let ask = venue.ask_price.load(Ordering::Acquire);
            
            // Only consider non-stale, recent updates
            let age = now.wrapping_sub(venue.last_update_ticks.load(Ordering::Acquire));
            let is_fresh = (age < 100000) as u64; // 100k ticks freshness threshold
            
            let effective_bid = bid * is_fresh as i64 * (1 - stale as i64);
            let effective_ask = ask * is_fresh as i64 + i64::MAX * (1 - is_fresh as i64);
            
            if effective_bid > highest_bid && effective_bid > 0 {
                highest_bid = effective_bid;
                best_buy_venue = venue.venue_id;
                best_qty = venue.bid_qty.load(Ordering::Acquire);
            }
            
            if effective_ask < lowest_ask && effective_ask < i64::MAX {
                lowest_ask = effective_ask;
                best_sell_venue = venue.venue_id;
                best_qty = core::cmp::min(best_qty, venue.ask_qty.load(Ordering::Acquire));
            }
        }

        if highest_bid == 0 || lowest_ask == i64::MAX {
            return None;
        }

        // Check for cross: bid > ask means arb opportunity
        if highest_bid > lowest_ask {
            let spread = highest_bid - lowest_ask;
            let mid = (highest_bid + lowest_ask) / 2;
            let spread_bps = (spread * 10000) / mid;
            let min_bps = self.min_spread_bps.load(Ordering::Acquire);

            if spread_bps >= min_bps {
                self.signals_generated.fetch_add(1, Ordering::AcqRel);
                self.last_signal_ticks.store(now, Ordering::Release);
                
                // Return: buy at low ask venue, sell at high bid venue
                return Some((best_sell_venue, best_buy_venue, spread, best_qty));
            }
        }

        None
    }

    #[inline]
    pub fn enable(&self) {
        self.enabled.store(1, Ordering::Release);
    }

    #[inline]
    pub fn disable(&self) {
        self.enabled.store(0, Ordering::Release);
    }
}

/// Circuit breaker for arb engine
#[repr(C, align(64))]
pub struct ArbCircuitBreaker {
    pub max_signals_per_second: AtomicU64,
    pub signal_count: AtomicU64,
    pub window_start_ticks: AtomicU64,
    pub tripped: AtomicU8,
    pub trip_count: AtomicU64,
    _padding: [u8; 32],
}

impl ArbCircuitBreaker {
    pub const fn new(max_sps: u64) -> Self {
        Self {
            max_signals_per_second: AtomicU64::new(max_sps),
            signal_count: AtomicU64::new(0),
            window_start_ticks: AtomicU64::new(0),
            tripped: AtomicU8::new(0),
            trip_count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    #[inline]
    pub fn check_and_record(&self) -> bool {
        if self.tripped.load(Ordering::Acquire) != 0 {
            return false;
        }

        let now = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        let window_start = self.window_start_ticks.load(Ordering::Acquire);
        
        // Reset window if needed (simplified - assumes ~3GHz CPU)
        if now.wrapping_sub(window_start) > 3_000_000_000 {
            self.window_start_ticks.store(now, Ordering::Release);
            self.signal_count.store(1, Ordering::Release);
            return true;
        }

        let count = self.signal_count.fetch_add(1, Ordering::AcqRel) + 1;
        let max = self.max_signals_per_second.load(Ordering::Acquire);

        if count > max {
            self.tripped.store(1, Ordering::Release);
            self.trip_count.fetch_add(1, Ordering::AcqRel);
            return false;
        }

        true
    }

    #[inline]
    pub fn reset(&self) {
        self.tripped.store(0, Ordering::Release);
        self.signal_count.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_registration() {
        let sniper = ArbitrageSniper::new(50); // 5 bps minimum
        
        assert!(sniper.register_venue(1));
        assert!(sniper.register_venue(2));
        assert_eq!(sniper.venue_count.load(Ordering::Acquire), 2);
    }

    #[test]
    fn test_arbitrage_detection() {
        let sniper = ArbitrageSniper::new(10);
        sniper.register_venue(1);
        sniper.register_venue(2);
        sniper.enable();

        // Venue 1: bid 50.00, ask 50.10
        sniper.venues[0].update_bid(50_000_000_000i64, 1_000_000_000i64);
        sniper.venues[0].update_ask(50_100_000_000i64, 1_000_000_000i64);

        // Venue 2: bid 50.15, ask 50.20 (cross with venue 1!)
        sniper.venues[1].update_bid(50_150_000_000i64, 500_000_000i64);
        sniper.venues[1].update_ask(50_200_000_000i64, 500_000_000i64);

        let arb = sniper.scan_for_arb();
        assert!(arb.is_some());
        
        let (buy_venue, sell_venue, spread, qty) = arb.unwrap();
        assert_eq!(buy_venue, 1); // Buy from venue 1 (lower ask)
        assert_eq!(sell_venue, 2); // Sell to venue 2 (higher bid)
        assert!(spread > 0);
    }

    #[test]
    fn test_circuit_breaker() {
        let breaker = ArbCircuitBreaker::new(5);
        
        for _ in 0..5 {
            assert!(breaker.check_and_record());
        }
        
        // Should trip after exceeding limit
        assert!(!breaker.check_and_record());
        assert_eq!(breaker.tripped.load(Ordering::Acquire), 1);
    }
}
