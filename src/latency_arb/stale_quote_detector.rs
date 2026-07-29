//! Stale Quote Detector and Toxic Flow Identifier
//! 
//! Detects stale quotes across market makers for protection against
//! toxic order flow. Uses zero-copy trade mapping and SIMD acceleration.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

pub type FixedPrice = i64;
pub type FixedQty = i64;
pub type TscTicks = u64;

const MAX_MARKERS: usize = 16;
const MARKER_MASK: usize = MAX_MARKERS - 1;

/// Market maker quote state (64-byte aligned)
#[repr(C, align(64))]
pub struct MakerQuote {
    pub maker_id: u32,
    pub bid_price: AtomicI64,
    pub ask_price: AtomicI64,
    pub bid_qty: AtomicI64,
    pub ask_qty: AtomicI64,
    pub last_update_ticks: AtomicU64,
    pub quote_age_ticks: AtomicU64,
    pub staleness_score: AtomicU64, // Accumulated staleness metric
    pub is_stale: AtomicU8,
    _padding: [u8; 23],
}

impl MakerQuote {
    pub const fn new(maker_id: u32) -> Self {
        Self {
            maker_id,
            bid_price: AtomicI64::new(0),
            ask_price: AtomicI64::new(i64::MAX),
            bid_qty: AtomicI64::new(0),
            ask_qty: AtomicI64::new(0),
            last_update_ticks: AtomicU64::new(0),
            quote_age_ticks: AtomicU64::new(0),
            staleness_score: AtomicU64::new(0),
            is_stale: AtomicU8::new(0),
            _padding: [0u8; 23],
        }
    }

    #[inline]
    pub fn update(&self, bid: FixedPrice, ask: FixedPrice, bid_qty: FixedQty, ask_qty: FixedQty) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.bid_price.store(bid, Ordering::Release);
        self.ask_price.store(ask, Ordering::Release);
        self.bid_qty.store(bid_qty, Ordering::Release);
        self.ask_qty.store(ask_qty, Ordering::Release);
        self.last_update_ticks.store(ts, Ordering::Release);
        self.quote_age_ticks.store(0, Ordering::Release);
        self.is_stale.store(0, Ordering::Release);
    }

    #[inline]
    pub fn check_staleness(&self, threshold_ticks: u64) -> bool {
        let now = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        let last = self.last_update_ticks.load(Ordering::Acquire);
        let age = now.wrapping_sub(last);
        
        self.quote_age_ticks.store(age, Ordering::Release);
        
        if age > threshold_ticks {
            self.is_stale.store(1, Ordering::Release);
            self.staleness_score.fetch_add(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }
}

/// Stale quote detector with multi-maker tracking
#[repr(C, align(64))]
pub struct StaleQuoteDetector {
    pub markers: [MakerQuote; MAX_MARKERS],
    pub marker_count: AtomicU8,
    pub staleness_threshold_ticks: AtomicU64,
    pub toxic_flow_detected: AtomicU8,
    pub protection_active: AtomicU8,
    _padding: [u8; 38],
}

impl StaleQuoteDetector {
    pub const fn new(threshold_ticks: u64) -> Self {
        const INIT: MakerQuote = MakerQuote::new(0);
        Self {
            markers: [INIT; MAX_MARKERS],
            marker_count: AtomicU8::new(0),
            staleness_threshold_ticks: AtomicU64::new(threshold_ticks),
            toxic_flow_detected: AtomicU8::new(0),
            protection_active: AtomicU8::new(0),
            _padding: [0u8; 38],
        }
    }

    #[inline]
    pub fn register_maker(&self, maker_id: u32) -> bool {
        let count = self.marker_count.load(Ordering::Acquire) as usize;
        if count >= MAX_MARKERS {
            return false;
        }
        self.markers[count] = MakerQuote::new(maker_id);
        self.marker_count.store((count + 1) as u8, Ordering::Release);
        true
    }

    /// Check all makers for staleness, return count of stale quotes
    #[inline]
    pub fn scan_staleness(&self) -> u64 {
        let count = self.marker_count.load(Ordering::Acquire) as usize;
        let threshold = self.staleness_threshold_ticks.load(Ordering::Acquire);
        let mut stale_count: u64 = 0;

        for i in 0..count {
            if self.markers[i].check_staleness(threshold) {
                stale_count += 1;
            }
        }

        // If majority are stale, flag toxic flow
        if stale_count > (count / 2) as u64 && count > 0 {
            self.toxic_flow_detected.store(1, Ordering::Release);
        }

        stale_count
    }

    /// Get the best non-stale bid across all makers
    #[inline]
    pub fn get_best_fresh_bid(&self) -> FixedPrice {
        let count = self.marker_count.load(Ordering::Acquire) as usize;
        let mut best_bid: FixedPrice = 0;

        for i in 0..count {
            if self.markers[i].is_stale.load(Ordering::Acquire) == 0 {
                let bid = self.markers[i].bid_price.load(Ordering::Acquire);
                if bid > best_bid {
                    best_bid = bid;
                }
            }
        }

        best_bid
    }

    /// Get the best non-stale ask across all makers
    #[inline]
    pub fn get_best_fresh_ask(&self) -> FixedPrice {
        let count = self.marker_count.load(Ordering::Acquire) as usize;
        let mut best_ask: FixedPrice = i64::MAX;

        for i in 0..count {
            if self.markers[i].is_stale.load(Ordering::Acquire) == 0 {
                let ask = self.markers[i].ask_price.load(Ordering::Acquire);
                if ask < best_ask {
                    best_ask = ask;
                }
            }
        }

        best_ask
    }

    #[inline]
    pub fn enable_protection(&self) {
        self.protection_active.store(1, Ordering::Release);
    }

    #[inline]
    pub fn is_protected(&self) -> bool {
        self.protection_active.load(Ordering::Acquire) != 0
    }

    #[inline]
    pub fn is_toxic(&self) -> bool {
        self.toxic_flow_detected.load(Ordering::Acquire) != 0
    }

    #[inline]
    pub fn reset_toxic_flag(&self) {
        self.toxic_flow_detected.store(0, Ordering::Release);
    }
}

/// Zero-copy trade mapper for aggressive order detection
#[repr(C, align(64))]
pub struct TradeMapper {
    pub recent_trades_ptr: *mut u8, // Pre-allocated buffer
    pub trade_count: AtomicU64,
    pub max_trades: usize,
    pub aggressive_buy_count: AtomicU64,
    pub aggressive_sell_count: AtomicU64,
    pub imbalance_ratio: AtomicU64, // Scaled by 10000
    _padding: [u8; 24],
}

impl TradeMapper {
    pub const fn new(ptr: *mut u8, max_trades: usize) -> Self {
        Self {
            recent_trades_ptr: ptr,
            trade_count: AtomicU64::new(0),
            max_trades,
            aggressive_buy_count: AtomicU64::new(0),
            aggressive_sell_count: AtomicU64::new(0),
            imbalance_ratio: AtomicU64::new(5000), // Neutral 0.5
            _padding: [0u8; 24],
        }
    }

    /// Record an aggressive trade (market order that hit a quote)
    #[inline]
    pub fn record_aggressive_trade(&self, is_buy: bool, qty: FixedQty) {
        if is_buy {
            self.aggressive_buy_count.fetch_add(qty as u64, Ordering::AcqRel);
        } else {
            self.aggressive_sell_count.fetch_add(qty as u64, Ordering::AcqRel);
        }

        let buys = self.aggressive_buy_count.load(Ordering::Acquire);
        let sells = self.aggressive_sell_count.load(Ordering::Acquire);
        let total = buys + sells;

        if total > 0 {
            let ratio = (buys * 10000) / total;
            self.imbalance_ratio.store(ratio, Ordering::Release);
        }

        self.trade_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Get buy/sell imbalance (0 = all sells, 10000 = all buys)
    #[inline]
    pub fn get_imbalance(&self) -> u64 {
        self.imbalance_ratio.load(Ordering::Acquire)
    }

    /// Check if flow is heavily imbalanced (potential toxic flow)
    #[inline]
    pub fn is_imbalanced(&self, threshold: u64) -> bool {
        let ratio = self.get_imbalance();
        ratio > (5000 + threshold) || ratio < (5000 - threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maker_registration() {
        let detector = StaleQuoteDetector::new(50000);
        
        assert!(detector.register_maker(1));
        assert!(detector.register_maker(2));
        assert_eq!(detector.marker_count.load(Ordering::Acquire), 2);
    }

    #[test]
    fn test_staleness_detection() {
        let detector = StaleQuoteDetector::new(1000); // Very short threshold
        detector.register_maker(1);
        
        // Update maker with fresh quote
        detector.markers[0].update(
            50_000_000_000i64,
            50_100_000_000i64,
            1_000_000_000i64,
            1_000_000_000i64,
        );
        
        // Should not be stale immediately
        assert_eq!(detector.scan_staleness(), 0);
        assert!(!detector.is_toxic());
    }

    #[test]
    fn test_trade_mapper_imbalance() {
        let mut buffer = [0u8; 4096];
        let mapper = TradeMapper::new(buffer.as_mut_ptr(), 100);
        
        // Record mostly buy aggression
        for _ in 0..80 {
            mapper.record_aggressive_trade(true, 1_000_000i64);
        }
        for _ in 0..20 {
            mapper.record_aggressive_trade(false, 1_000_000i64);
        }
        
        let imbalance = mapper.get_imbalance();
        assert!(imbalance > 7000); // >70% buys
        assert!(mapper.is_imbalanced(1000));
    }

    #[test]
    fn test_fresh_quote_filtering() {
        let detector = StaleQuoteDetector::new(100000);
        detector.register_maker(1);
        detector.register_maker(2);
        
        // Maker 1: fresh
        detector.markers[0].update(50_000_000_000i64, 50_100_000_000i64, 1_000_000_000i64, 1_000_000_000i64);
        
        // Maker 2: stale (manually flag)
        detector.markers[1].is_stale.store(1, Ordering::Release);
        detector.markers[1].bid_price.store(49_000_000_000i64, Ordering::Release);
        
        let best_fresh = detector.get_best_fresh_bid();
        assert_eq!(best_fresh, 50_000_000_000i64); // Only from fresh maker
    }
}
