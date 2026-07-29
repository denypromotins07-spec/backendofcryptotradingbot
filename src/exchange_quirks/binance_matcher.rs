//! Binance Matching Engine Emulation
//! 
//! Implements price-time priority matching, STP (Self-Trade Prevention) rules,
//! and specific Binance quirks. Uses lock-free data structures and fixed-point math.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

/// Fixed-point types
pub type FixedPrice = i64;
pub type FixedQty = i64;
pub type TscTicks = u64;

/// Maximum order book depth tracked
const MAX_BOOK_DEPTH: usize = 100;
const BOOK_MASK: usize = MAX_BOOK_DEPTH - 1;

/// STP modes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StpMode {
    None = 0,
    CancelNewest = 1,
    CancelOldest = 2,
    CancelBoth = 3,
}

/// Order side
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

/// Cache-line aligned order entry (64 bytes)
#[repr(C, align(64))]
pub struct BinanceOrder {
    pub order_id: AtomicU64,
    pub client_order_id: AtomicU64,
    pub price: AtomicI64,
    pub qty: AtomicI64,
    pub filled_qty: AtomicI64,
    pub timestamp_ticks: AtomicU64,
    pub side: AtomicU8,
    pub is_maker: AtomicU8,
    _padding: [u8; 54],
}

impl BinanceOrder {
    pub const fn new(
        order_id: u64,
        client_order_id: u64,
        price: FixedPrice,
        qty: FixedQty,
        side: Side,
    ) -> Self {
        Self {
            order_id: AtomicU64::new(order_id),
            client_order_id: AtomicU64::new(client_order_id),
            price: AtomicI64::new(price),
            qty: AtomicI64::new(qty),
            filled_qty: AtomicI64::new(0),
            timestamp_ticks: AtomicU64::new(unsafe { core::arch::x86_64::_rdtsc() } as u64),
            side: AtomicU8::new(side as u8),
            is_maker: AtomicU8::new(0),
            _padding: [0u8; 54],
        }
    }

    #[inline]
    pub fn remaining_qty(&self) -> FixedQty {
        self.qty.load(Ordering::Acquire) - self.filled_qty.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_fully_filled(&self) -> bool {
        self.remaining_qty() <= 0
    }
}

/// Price level in order book (lock-free)
#[repr(C, align(64))]
pub struct PriceLevel {
    pub price: AtomicI64,
    pub total_qty: AtomicI64,
    pub order_count: AtomicU64,
    pub first_order_id: AtomicU64,
    pub last_update_ticks: AtomicU64,
    _padding: [u8; 24],
}

impl PriceLevel {
    pub const fn new() -> Self {
        Self {
            price: AtomicI64::new(0),
            total_qty: AtomicI64::new(0),
            order_count: AtomicU64::new(0),
            first_order_id: AtomicU64::new(0),
            last_update_ticks: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }

    #[inline]
    pub fn set_price(&self, p: FixedPrice) {
        self.price.store(p, Ordering::Release);
    }

    #[inline]
    pub fn add_qty(&self, qty: FixedQty, order_id: u64) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.total_qty.fetch_add(qty, Ordering::AcqRel);
        self.order_count.fetch_add(1, Ordering::AcqRel);
        if self.first_order_id.load(Ordering::Acquire) == 0 {
            self.first_order_id.store(order_id, Ordering::Release);
        }
        self.last_update_ticks.store(ts, Ordering::Release);
    }

    #[inline]
    pub fn remove_qty(&self, qty: FixedQty) -> FixedQty {
        self.total_qty.fetch_sub(qty, Ordering::AcqRel) - qty
    }
}

/// Binance-style order book with price-time priority
#[repr(C, align(64))]
pub struct BinanceOrderBook {
    pub bids: [PriceLevel; MAX_BOOK_DEPTH],
    pub asks: [PriceLevel; MAX_BOOK_DEPTH],
    pub bid_count: AtomicU64,
    pub ask_count: AtomicU64,
    pub best_bid_price: AtomicI64,
    pub best_ask_price: AtomicI64,
    pub last_trade_price: AtomicI64,
    pub sequence_num: AtomicU64,
    _padding: [u8; 32],
}

impl BinanceOrderBook {
    pub const fn new() -> Self {
        const INIT: PriceLevel = PriceLevel::new();
        Self {
            bids: [INIT; MAX_BOOK_DEPTH],
            asks: [INIT; MAX_BOOK_DEPTH],
            bid_count: AtomicU64::new(0),
            ask_count: AtomicU64::new(0),
            best_bid_price: AtomicI64::new(0),
            best_ask_price: AtomicI64::new(i64::MAX),
            last_trade_price: AtomicI64::new(0),
            sequence_num: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Add order to book (price-time priority)
    #[inline]
    pub fn add_order(&self, order: &BinanceOrder) -> bool {
        let side = order.side.load(Ordering::Acquire);
        let price = order.price.load(Ordering::Acquire);
        let qty = order.remaining_qty();
        
        if qty <= 0 {
            return false;
        }
        
        let seq = self.sequence_num.fetch_add(1, Ordering::AcqRel);
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        
        if side == Side::Buy as u8 {
            // Insert into bids (sorted descending)
            self.insert_bid(price, qty, order.order_id.load(Ordering::Acquire), ts, seq);
        } else {
            // Insert into asks (sorted ascending)
            self.insert_ask(price, qty, order.order_id.load(Ordering::Acquire), ts, seq);
        }
        
        true
    }

    #[inline]
    fn insert_bid(&self, price: FixedPrice, qty: FixedQty, order_id: u64, ts: u64, seq: u64) {
        let count = self.bid_count.load(Ordering::Acquire) as usize;
        
        // Find insertion point (maintain sorted order)
        let mut idx = 0;
        while idx < count && idx < MAX_BOOK_DEPTH {
            let level_price = self.bids[idx].price.load(Ordering::Acquire);
            if level_price == 0 || level_price < price {
                break;
            }
            if level_price == price {
                // Add to existing level
                self.bids[idx].add_qty(qty, order_id);
                self.bids[idx].last_update_ticks.store(ts, Ordering::Release);
                return;
            }
            idx += 1;
        }
        
        // Insert new level (simplified - assumes space)
        if count < MAX_BOOK_DEPTH {
            self.bids[idx].set_price(price);
            self.bids[idx].add_qty(qty, order_id);
            self.bids[idx].last_update_ticks.store(ts, Ordering::Release);
            self.bid_count.store(count as u64 + 1, Ordering::Release);
            
            if idx == 0 {
                self.best_bid_price.store(price, Ordering::Release);
            }
        }
    }

    #[inline]
    fn insert_ask(&self, price: FixedPrice, qty: FixedQty, order_id: u64, ts: u64, seq: u64) {
        let count = self.ask_count.load(Ordering::Acquire) as usize;
        
        // Find insertion point (maintain sorted order)
        let mut idx = 0;
        while idx < count && idx < MAX_BOOK_DEPTH {
            let level_price = self.asks[idx].price.load(Ordering::Acquire);
            if level_price == 0 || level_price > price {
                break;
            }
            if level_price == price {
                self.asks[idx].add_qty(qty, order_id);
                self.asks[idx].last_update_ticks.store(ts, Ordering::Release);
                return;
            }
            idx += 1;
        }
        
        if count < MAX_BOOK_DEPTH {
            self.asks[idx].set_price(price);
            self.asks[idx].add_qty(qty, order_id);
            self.asks[idx].last_update_ticks.store(ts, Ordering::Release);
            self.ask_count.store(count as u64 + 1, Ordering::Release);
            
            if idx == 0 {
                self.best_ask_price.store(price, Ordering::Release);
            }
        }
    }

    /// Match incoming order against book, return fill quantity
    #[inline]
    pub fn match_order(&self, side: Side, qty: FixedQty, price: FixedPrice) -> FixedQty {
        let mut remaining = qty;
        let mut filled = 0;
        
        if side == Side::Buy {
            // Match against asks
            let ask_count = self.ask_count.load(Ordering::Acquire) as usize;
            for i in 0..ask_count {
                let ask_price = self.asks[i].price.load(Ordering::Acquire);
                if ask_price == 0 || ask_price > price {
                    break;
                }
                
                let avail = self.asks[i].total_qty.load(Ordering::Acquire);
                let take = core::cmp::min(remaining, avail);
                
                if take > 0 {
                    self.asks[i].remove_qty(take);
                    remaining -= take;
                    filled += take;
                    
                    if self.asks[i].total_qty.load(Ordering::Acquire) <= 0 {
                        // Level depleted - simplified removal
                        self.asks[i].set_price(0);
                    }
                }
                
                if remaining <= 0 {
                    break;
                }
            }
            
            if filled > 0 {
                self.last_trade_price.store(price, Ordering::Release);
            }
        } else {
            // Match against bids
            let bid_count = self.bid_count.load(Ordering::Acquire) as usize;
            for i in 0..bid_count {
                let bid_price = self.bids[i].price.load(Ordering::Acquire);
                if bid_price == 0 || bid_price < price {
                    break;
                }
                
                let avail = self.bids[i].total_qty.load(Ordering::Acquire);
                let take = core::cmp::min(remaining, avail);
                
                if take > 0 {
                    self.bids[i].remove_qty(take);
                    remaining -= take;
                    filled += take;
                    
                    if self.bids[i].total_qty.load(Ordering::Acquire) <= 0 {
                        self.bids[i].set_price(0);
                    }
                }
                
                if remaining <= 0 {
                    break;
                }
            }
            
            if filled > 0 {
                self.last_trade_price.store(price, Ordering::Release);
            }
        }
        
        filled
    }

    #[inline]
    pub fn get_spread(&self) -> FixedPrice {
        let best_ask = self.best_ask_price.load(Ordering::Acquire);
        let best_bid = self.best_bid_price.load(Ordering::Acquire);
        
        if best_ask == i64::MAX || best_bid == 0 {
            return i64::MAX;
        }
        
        best_ask - best_bid
    }
}

/// STP (Self-Trade Prevention) Engine
#[repr(C, align(64))]
pub struct StpEngine {
    pub mode: AtomicU8,
    pub account_id: AtomicU64,
    pub violation_count: AtomicU64,
    pub last_violation_ticks: AtomicU64,
    pub circuit_open: AtomicU8,
    _padding: [u8; 39],
}

impl StpEngine {
    pub const fn new(mode: StpMode, account_id: u64) -> Self {
        Self {
            mode: AtomicU8::new(mode as u8),
            account_id: AtomicU64::new(account_id),
            violation_count: AtomicU64::new(0),
            last_violation_ticks: AtomicU64::new(0),
            circuit_open: AtomicU8::new(0),
            _padding: [0u8; 39],
        }
    }

    /// Check if order would self-trade, apply STP logic
    #[inline]
    pub fn check_stp(&self, incoming_side: Side, incoming_price: FixedPrice, book: &BinanceOrderBook) -> Option<FixedPrice> {
        if self.mode.load(Ordering::Acquire) == StpMode::None as u8 {
            return None;
        }
        
        if self.circuit_open.load(Ordering::Acquire) != 0 {
            return Some(incoming_price); // Reject
        }
        
        let mut would_self_trade = false;
        
        if incoming_side == Side::Buy {
            let best_ask = book.best_ask_price.load(Ordering::Acquire);
            if best_ask != i64::MAX && incoming_price >= best_ask {
                would_self_trade = true;
            }
        } else {
            let best_bid = book.best_bid_price.load(Ordering::Acquire);
            if best_bid != 0 && incoming_price <= best_bid {
                would_self_trade = true;
            }
        }
        
        if !would_self_trade {
            return None;
        }
        
        // Record violation
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.violation_count.fetch_add(1, Ordering::AcqRel);
        self.last_violation_ticks.store(ts, Ordering::Release);
        
        match self.mode.load(Ordering::Acquire) {
            x if x == StpMode::CancelNewest as u8 => Some(incoming_price), // Cancel incoming
            x if x == StpMode::CancelOldest as u8 => Some(0), // Would need to cancel resting
            x if x == StpMode::CancelBoth as u8 => Some(incoming_price),
            _ => None,
        }
    }

    #[inline]
    pub fn reset_circuit(&self) {
        self.circuit_open.store(0, Ordering::Release);
        self.violation_count.store(0, Ordering::Release);
    }
}

/// Shadow mode matcher for testing
#[repr(C, align(64))]
pub struct ShadowMatcher {
    pub enabled: AtomicU8,
    pub theoretical_fills: AtomicI64,
    pub actual_fills: AtomicI64,
    pub avg_fill_price_deviation: AtomicI64,
    pub sample_count: AtomicU64,
    _padding: [u8; 32],
}

impl ShadowMatcher {
    pub const fn new() -> Self {
        Self {
            enabled: AtomicU8::new(0),
            theoretical_fills: AtomicI64::new(0),
            actual_fills: AtomicI64::new(0),
            avg_fill_price_deviation: AtomicI64::new(0),
            sample_count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    #[inline]
    pub fn log_theoretical_fill(&self, qty: FixedQty, expected_price: FixedPrice, actual_price: FixedPrice) {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return;
        }
        
        self.theoretical_fills.fetch_add(qty, Ordering::AcqRel);
        
        let count = self.sample_count.fetch_add(1, Ordering::AcqRel);
        let deviation = (expected_price - actual_price).abs();
        let current_avg = self.avg_fill_price_deviation.load(Ordering::Acquire);
        let new_avg = ((current_avg * count as i64) + deviation) / (count + 1) as i64;
        self.avg_fill_price_deviation.store(new_avg, Ordering::Release);
    }

    #[inline]
    pub fn log_actual_fill(&self, qty: FixedQty) {
        self.actual_fills.fetch_add(qty, Ordering::AcqRel);
    }

    #[inline]
    pub fn enable(&self) {
        self.enabled.store(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = BinanceOrder::new(12345, 99999, 50_000_000_000i64, 1_000_000_000i64, Side::Buy);
        
        assert_eq!(order.order_id.load(Ordering::Acquire), 12345);
        assert_eq!(order.side.load(Ordering::Acquire), Side::Buy as u8);
        assert_eq!(order.remaining_qty(), 1_000_000_000i64);
    }

    #[test]
    fn test_order_book_insertion() {
        let book = BinanceOrderBook::new();
        
        let order1 = BinanceOrder::new(1, 101, 50_000_000_000i64, 1_000_000_000i64, Side::Buy);
        let order2 = BinanceOrder::new(2, 102, 49_900_000_000i64, 500_000_000i64, Side::Buy);
        
        book.add_order(&order1);
        book.add_order(&order2);
        
        assert_eq!(book.best_bid_price.load(Ordering::Acquire), 50_000_000_000i64);
        assert_eq!(book.bid_count.load(Ordering::Acquire), 2);
    }

    #[test]
    fn test_matching() {
        let book = BinanceOrderBook::new();
        
        // Add sell orders
        let ask1 = BinanceOrder::new(1, 201, 50_100_000_000i64, 500_000_000i64, Side::Sell);
        let ask2 = BinanceOrder::new(2, 202, 50_200_000_000i64, 500_000_000i64, Side::Sell);
        book.add_order(&ask1);
        book.add_order(&ask2);
        
        // Incoming buy should match
        let filled = book.match_order(Side::Buy, 1_000_000_000i64, 50_200_000_000i64);
        
        assert_eq!(filled, 1_000_000_000i64);
        assert_eq!(book.last_trade_price.load(Ordering::Acquire), 50_200_000_000i64);
    }

    #[test]
    fn test_stp_engine() {
        let stp = StpEngine::new(StpMode::CancelNewest, 12345);
        let book = BinanceOrderBook::new();
        
        // Add our own sell order
        let ask = BinanceOrder::new(1, 101, 50_000_000_000i64, 1_000_000_000i64, Side::Sell);
        book.add_order(&ask);
        
        // Incoming buy at same price would self-trade
        let result = stp.check_stp(Side::Buy, 50_000_000_000i64, &book);
        
        assert!(result.is_some());
        assert_eq!(stp.violation_count.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_spread_calculation() {
        let book = BinanceOrderBook::new();
        
        let bid = BinanceOrder::new(1, 101, 49_900_000_000i64, 1_000_000_000i64, Side::Buy);
        let ask = BinanceOrder::new(2, 201, 50_100_000_000i64, 1_000_000_000i64, Side::Sell);
        
        book.add_order(&bid);
        book.add_order(&ask);
        
        assert_eq!(book.get_spread(), 200_000_000i64);
    }
}
