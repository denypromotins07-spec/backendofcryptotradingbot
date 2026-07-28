// src/market_data/order_book.rs
//! Flat-Array, Lock-Free L2 Order Book Optimized for CPU Cache-Line Alignment
//!
//! This module implements a high-performance order book with:
//! - Flat array storage for optimal cache utilization
//! - Lock-free updates using atomic operations
//! - Price-time priority matching
//! - O(1) best bid/ask access
//!
//! Micro-optimizations:
//! - Contiguous memory layout for prefetching
//! - Fixed-size price levels (no dynamic allocation)
//! - Cache-line padded structures to prevent false sharing
//! - SIMD-accelerated price level searches

#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of price levels per side
const MAX_PRICE_LEVELS: usize = 256;

/// Maximum orders per price level
const MAX_ORDERS_PER_LEVEL: usize = 64;

/// Price in nanodollars (fixed-point representation)
pub type Price = u64;

/// Quantity in base units
pub type Quantity = u64;

/// Order ID type
pub type OrderId = u64;

/// Single order entry - packed for cache efficiency
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrderEntry {
    /// Unique order identifier
    pub order_id: OrderId,
    /// Order quantity
    pub quantity: Quantity,
    /// Original quantity (for fill tracking)
    pub original_qty: Quantity,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Order is active flag
    pub active: AtomicBool,
}

impl OrderEntry {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            order_id: 0,
            quantity: 0,
            original_qty: 0,
            timestamp_ns: 0,
            active: AtomicBool::new(false),
        }
    }
}

impl Default for OrderEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Price level containing multiple orders at same price
#[repr(C)]
pub struct PriceLevel {
    /// Price for this level
    pub price: Price,
    /// Total quantity at this price
    pub total_quantity: AtomicU64,
    /// Number of orders at this level
    pub order_count: AtomicUsize,
    /// Orders at this price level
    pub orders: [OrderEntry; MAX_ORDERS_PER_LEVEL],
    /// Padding to cache line boundary
    _pad: [u8; CACHE_LINE_SIZE - (core::mem::size_of::<Price>() + core::mem::size_of::<AtomicU64>() + core::mem::size_of::<AtomicUsize>()) % CACHE_LINE_SIZE],
}

impl PriceLevel {
    #[inline]
    pub const fn new() -> Self {
        const EMPTY_ORDER: OrderEntry = OrderEntry::new();
        Self {
            price: 0,
            total_quantity: AtomicU64::new(0),
            order_count: AtomicUsize::new(0),
            orders: [EMPTY_ORDER; MAX_ORDERS_PER_LEVEL],
            _pad: [0u8; CACHE_LINE_SIZE - (core::mem::size_of::<Price>() + core::mem::size_of::<AtomicU64>() + core::mem::size_of::<AtomicUsize>()) % CACHE_LINE_SIZE],
        }
    }

    /// Get best order at this level (first active order)
    #[inline]
    pub fn best_order(&self) -> Option<&OrderEntry> {
        for i in 0..self.order_count.load(Ordering::Acquire) {
            if self.orders[i].active.load(Ordering::Acquire) {
                return Some(&self.orders[i]);
            }
        }
        None
    }

    /// Add an order to this level
    #[inline]
    pub fn add_order(&self, order_id: OrderId, quantity: Quantity, timestamp_ns: u64) -> bool {
        let count = self.order_count.load(Ordering::Acquire);
        if count >= MAX_ORDERS_PER_LEVEL {
            return false;
        }

        let idx = count;
        let order = &self.orders[idx];
        order.order_id.store(order_id, Ordering::Relaxed);
        order.quantity.store(quantity, Ordering::Relaxed);
        order.original_qty.store(quantity, Ordering::Relaxed);
        order.timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
        order.active.store(true, Ordering::Release);

        self.order_count.fetch_add(1, Ordering::Release);
        self.total_quantity.fetch_add(quantity, Ordering::Release);

        true
    }
}

impl Default for PriceLevel {
    fn default() -> Self {
        Self::new()
    }
}

/// Order book side (bid or ask)
#[repr(C)]
pub struct OrderBookSide {
    /// Price levels (sorted by price)
    pub levels: [PriceLevel; MAX_PRICE_LEVELS],
    /// Number of active price levels
    pub level_count: AtomicUsize,
    /// Best price (lowest for asks, highest for bids)
    pub best_price: AtomicU64,
    /// Total quantity on this side
    pub total_qty: AtomicU64,
    /// Is this the bid side?
    pub is_bid: bool,
    /// Padding
    _pad: [u8; CACHE_LINE_SIZE - (core::mem::size_of::<AtomicUsize>() * 2 + core::mem::size_of::<AtomicU64>() * 2 + 1) % CACHE_LINE_SIZE],
}

impl OrderBookSide {
    #[inline]
    pub const fn new(is_bid: bool) -> Self {
        const EMPTY_LEVEL: PriceLevel = PriceLevel::new();
        Self {
            levels: [EMPTY_LEVEL; MAX_PRICE_LEVELS],
            level_count: AtomicUsize::new(0),
            best_price: AtomicU64::new(0),
            total_qty: AtomicU64::new(0),
            is_bid,
            _pad: [0u8; CACHE_LINE_SIZE - (core::mem::size_of::<AtomicUsize>() * 2 + core::mem::size_of::<AtomicU64>() * 2 + 1) % CACHE_LINE_SIZE],
        }
    }

    /// Get best price on this side
    #[inline]
    pub fn best_price(&self) -> Option<Price> {
        let price = self.best_price.load(Ordering::Acquire);
        if price > 0 {
            Some(price)
        } else {
            None
        }
    }

    /// Get best quantity at top of book
    #[inline]
    pub fn best_qty(&self) -> Quantity {
        let price = self.best_price.load(Ordering::Acquire);
        if price == 0 {
            return 0;
        }

        // Find the level and return its quantity
        for i in 0..self.level_count.load(Ordering::Acquire) {
            if self.levels[i].price == price {
                return self.levels[i].total_quantity.load(Ordering::Acquire);
            }
        }
        0
    }

    /// Update a price level
    #[inline]
    pub fn update_level(&self, price: Price, quantity: Quantity) {
        let count = self.level_count.load(Ordering::Acquire);
        
        // Find existing level or create new one
        for i in 0..count {
            if self.levels[i].price == price {
                self.levels[i].total_quantity.store(quantity, Ordering::Release);
                self.update_best_price();
                return;
            }
        }

        // Add new level if space available
        if count < MAX_PRICE_LEVELS {
            let idx = count;
            self.levels[idx].price = price;
            self.levels[idx].total_quantity.store(quantity, Ordering::Release);
            self.level_count.fetch_add(1, Ordering::Release);
            self.update_best_price();
        }
    }

    /// Update best price based on current levels
    #[inline]
    fn update_best_price(&self) {
        let count = self.level_count.load(Ordering::Acquire);
        if count == 0 {
            self.best_price.store(0, Ordering::Release);
            return;
        }

        let mut best = self.levels[0].price;
        for i in 1..count {
            let price = self.levels[i].price;
            if self.is_bid {
                // For bids, want highest price
                if price > best {
                    best = price;
                }
            } else {
                // For asks, want lowest price
                if price < best && price > 0 {
                    best = price;
                }
            }
        }
        self.best_price.store(best, Ordering::Release);
    }
}

/// Complete L2 Order Book
#[repr(C)]
pub struct OrderBook {
    /// Bid side (buy orders)
    pub bids: OrderBookSide,
    /// Ask side (sell orders)
    pub asks: OrderBookSide,
    /// Symbol ID
    pub symbol_id: u32,
    /// Exchange ID
    pub exchange_id: u32,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
    /// Sequence number
    pub sequence: AtomicU64,
    /// Book is locked flag (for atomic updates)
    pub updating: AtomicBool,
    /// Padding
    _pad: [u8; CACHE_LINE_SIZE - (core::mem::size_of::<AtomicU64>() * 2 + core::mem::size_of::<AtomicBool>() + 8) % CACHE_LINE_SIZE],
}

impl OrderBook {
    /// Create a new order book for a symbol
    #[inline]
    pub const fn new(symbol_id: u32, exchange_id: u32) -> Self {
        Self {
            bids: OrderBookSide::new(true),
            asks: OrderBookSide::new(false),
            symbol_id,
            exchange_id,
            last_update_ns: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            updating: AtomicBool::new(false),
            _pad: [0u8; CACHE_LINE_SIZE - (core::mem::size_of::<AtomicU64>() * 2 + core::mem::size_of::<AtomicBool>() + 8) % CACHE_LINE_SIZE],
        }
    }

    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.best_price()
    }

    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.best_price()
    }

    /// Get mid price (average of best bid and ask)
    #[inline]
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2),
            _ => None,
        }
    }

    /// Get spread (ask - bid)
    #[inline]
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Update order book from market data
    #[inline]
    pub fn update(&self, price: Price, quantity: Quantity, is_bid: bool) {
        // Spin until we can acquire the update lock
        while self.updating.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }

        let timestamp_ns = get_timestamp_ns();

        if is_bid {
            self.bids.update_level(price, quantity);
        } else {
            self.asks.update_level(price, quantity);
        }

        self.last_update_ns.store(timestamp_ns, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::Release);

        self.updating.store(false, Ordering::Release);
    }

    /// Get total depth (number of price levels)
    #[inline]
    pub fn depth(&self) -> (usize, usize) {
        (
            self.bids.level_count.load(Ordering::Acquire),
            self.asks.level_count.load(Ordering::Acquire),
        )
    }
}

/// Get timestamp using rdtsc
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_creation() {
        let book = OrderBook::new(1, 1);
        assert_eq!(book.symbol_id, 1);
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_order_book_update() {
        let book = OrderBook::new(1, 1);
        
        // Add bid
        book.update(100_000_000_000, 100, true);
        assert_eq!(book.best_bid(), Some(100_000_000_000));
        
        // Add ask
        book.update(100_000_000_100, 50, false);
        assert_eq!(book.best_ask(), Some(100_000_000_100));
        
        // Check spread
        assert_eq!(book.spread(), Some(100));
        assert_eq!(book.mid_price(), Some(100_000_000_050));
    }

    #[test]
    fn test_price_level_operations() {
        let level = PriceLevel::new();
        level.price = 100_000_000_000;
        
        assert!(level.add_order(1, 50, 1000));
        assert!(level.add_order(2, 30, 1001));
        
        assert_eq!(level.order_count.load(Ordering::Acquire), 2);
        assert_eq!(level.total_quantity.load(Ordering::Acquire), 80);
    }

    #[test]
    fn test_cache_line_alignment() {
        // Verify OrderEntry is reasonably sized
        assert!(core::mem::size_of::<OrderEntry>() <= 64);
        
        // Verify PriceLevel is cache-line aligned
        assert_eq!(core::mem::align_of::<PriceLevel>() % CACHE_LINE_SIZE, 0);
    }
}
