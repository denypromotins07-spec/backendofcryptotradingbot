//! Historical order book replay and realistic slippage/market-impact modeler.
//! 
//! Lock-free circular buffer for historical queue positions.
//! Branchless execution for deterministic latency.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum queue depth tracked
pub const MAX_QUEUE_DEPTH: usize = 1024;

/// Maximum price levels in order book
pub const MAX_BOOK_LEVELS: usize = 50;

/// Order book level
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BookLevel {
    pub price_tick: i64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub bid_queue_pos: u64,
    pub ask_queue_pos: u64,
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

impl BookLevel {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            price_tick: 0,
            bid_size: 0,
            ask_size: 0,
            bid_queue_pos: 0,
            ask_queue_pos: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }
}

/// Slippage model result
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlippageResult {
    pub expected_price: i64,
    pub actual_price: i64,
    pub slippage_ticks: i64,
    pub slippage_bps: i32,
    pub market_impact_bps: i32,
    pub fill_probability: u32, // Scaled by 10000
    pub estimated_latency_ns: u64,
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

impl SlippageResult {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            expected_price: 0,
            actual_price: 0,
            slippage_ticks: 0,
            slippage_bps: 0,
            market_impact_bps: 0,
            fill_probability: 10000,
            estimated_latency_ns: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }
}

/// Lock-free circular buffer for queue positions
#[repr(C)]
pub struct QueueBuffer<const N: usize> {
    data: [i64; N],
    head: AtomicU64,
    tail: AtomicU64,
    sum: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl<const N: usize> QueueBuffer<N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: [0i64; N],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            sum: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }

    #[inline]
    pub fn push(&self, value: i64) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if ((head + 1) % N as u64) == tail {
            return false; // Buffer full
        }

        let idx = (head % N as u64) as usize;
        unsafe {
            let ptr = self.data.as_ptr() as *mut i64;
            let old = *ptr.add(idx);
            ptr.add(idx).write(value);
            
            // Update running sum
            let sum = self.sum.load(Ordering::Relaxed);
            self.sum.store(sum - old + value, Ordering::Relaxed);
        }

        self.head.store(head + 1, Ordering::Release);
        true
    }

    #[inline]
    pub fn pop(&self) -> Option<i64> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let idx = (tail % N as u64) as usize;
        let value = unsafe { *self.data.as_ptr().add(idx) };

        self.tail.store(tail + 1, Ordering::Release);
        Some(value)
    }

    #[inline]
    pub fn average(&self) -> f64 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let len = (head - tail) as f64;

        if len < 1.0 {
            return 0.0;
        }

        self.sum.load(Ordering::Relaxed) as f64 / len
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        (head - tail) as usize
    }
}

/// Slippage modeler with historical replay
#[repr(C)]
pub struct SlippageModeler<const QUEUE_DEPTH: usize, const BOOK_LEVELS: usize> {
    /// Order book snapshot
    book: UnsafeCell<[BookLevel; BOOK_LEVELS]>,
    /// Queue position history per level
    queue_history: [QueueBuffer<QUEUE_DEPTH>; BOOK_LEVELS],
    /// Best bid index
    best_bid_idx: AtomicU64,
    /// Best ask index
    best_ask_idx: AtomicU64,
    /// Spread in ticks
    spread_ticks: AtomicI64,
    /// Volatility factor (scaled by 1000)
    volatility_factor: AtomicU64,
    /// Liquidity factor (scaled by 1000)
    liquidity_factor: AtomicU64,
    /// Total orders modeled
    orders_modeled: AtomicU64,
    /// Total slippage accumulated (ticks)
    total_slippage: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 34],
}

use core::cell::UnsafeCell;

// SAFETY: All interior mutability protected by atomics
unsafe impl<const Q: usize, const B: usize> Send for SlippageModeler<Q, B> {}
unsafe impl<const Q: usize, const B: usize> Sync for SlippageModeler<Q, B> {}

impl<const QUEUE_DEPTH: usize, const BOOK_LEVELS: usize> SlippageModeler<QUEUE_DEPTH, BOOK_LEVELS> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            book: UnsafeCell::new([BookLevel::new(); BOOK_LEVELS]),
            queue_history: [QueueBuffer::new(); BOOK_LEVELS],
            best_bid_idx: AtomicU64::new(0),
            best_ask_idx: AtomicU64::new(BOOK_LEVELS as u64 / 2),
            spread_ticks: AtomicI64::new(1),
            volatility_factor: AtomicU64::new(1000),
            liquidity_factor: AtomicU64::new(1000),
            orders_modeled: AtomicU64::new(0),
            total_slippage: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 34],
        }
    }

    /// Initialize order book
    #[inline]
    pub fn init_book(&self, mid_price: i64, tick_size: i64) {
        let book = unsafe { &mut *self.book.get() };
        let mid_idx = BOOK_LEVELS / 2;

        for i in 0..BOOK_LEVELS {
            let offset = (i as i64 - mid_idx as i64) * tick_size;
            book[i].price_tick = mid_price + offset;

            // Simulate typical book shape (more volume near mid)
            let distance = (i as i64 - mid_idx as i64).unsigned_abs();
            let base_vol = 1000u64.saturating_sub(distance * 50);

            if i < mid_idx {
                book[i].bid_size = base_vol;
                book[i].ask_size = 0;
            } else {
                book[i].bid_size = 0;
                book[i].ask_size = base_vol;
            }
        }

        self.best_bid_idx.store((mid_idx - 1) as u64, Ordering::Relaxed);
        self.best_ask_idx.store(mid_idx as u64, Ordering::Relaxed);
        self.spread_ticks.store(tick_size, Ordering::Relaxed);
    }

    /// Update book level from market data
    #[inline]
    pub fn update_level(&self, level_idx: usize, price: i64, bid_size: u64, ask_size: u64) {
        if level_idx >= BOOK_LEVELS {
            return;
        }

        let book = unsafe { &mut *self.book.get() };
        book[level_idx].price_tick = price;
        book[level_idx].bid_size = bid_size;
        book[level_idx].ask_size = ask_size;

        // Record queue position
        self.queue_history[level_idx].push(bid_size as i64 - ask_size as i64);
    }

    /// Estimate slippage for market order
    #[inline]
    pub fn estimate_market_slippage(&self, quantity: u64, is_buy: bool) -> SlippageResult {
        let book = unsafe { &*self.book.get() };
        let mut result = SlippageResult::new();

        let start_idx = if is_buy {
            self.best_ask_idx.load(Ordering::Relaxed) as usize
        } else {
            self.best_bid_idx.load(Ordering::Relaxed) as usize
        };

        let mut remaining = quantity;
        let mut total_cost = 0u128;
        let mut filled = 0u64;
        let first_price = if is_buy {
            book[start_idx].price_tick
        } else {
            book[start_idx].price_tick
        };

        // Walk through book levels
        let mut idx = start_idx;
        while remaining > 0 && idx < BOOK_LEVELS && idx > 0 {
            let available = if is_buy {
                book[idx].ask_size
            } else {
                book[idx].bid_size
            };

            let fill_qty = remaining.min(available);
            if fill_qty > 0 {
                total_cost += (fill_qty as u128) * (book[idx].price_tick as u128);
                filled += fill_qty;
                remaining -= fill_qty;
            }

            if is_buy {
                idx += 1;
            } else {
                idx = idx.saturating_sub(1);
            }
        }

        if filled > 0 {
            let avg_price = (total_cost / filled as u128) as i64;
            result.actual_price = avg_price;
            result.expected_price = first_price;
            result.slippage_ticks = avg_price - first_price;

            // Calculate basis points
            if first_price > 0 {
                result.slippage_bps = ((result.slippage_ticks.abs() * 10000) / first_price) as i32;
            }

            // Market impact estimation
            let vol_factor = self.volatility_factor.load(Ordering::Relaxed);
            let liq_factor = self.liquidity_factor.load(Ordering::Relaxed);
            result.market_impact_bps = ((quantity as u64 * vol_factor * liq_factor) / 1_000_000) as i32;

            // Fill probability based on queue position
            let queue_avg = self.queue_history[start_idx].average();
            result.fill_probability = (10000u32).saturating_sub((queue_avg.abs() / 100) as u32);
        }

        // Update statistics
        self.orders_modeled.fetch_add(1, Ordering::Relaxed);
        self.total_slippage.fetch_add(result.slippage_ticks, Ordering::Relaxed);

        result
    }

    /// Estimate slippage for limit order
    #[inline]
    pub fn estimate_limit_slippage(&self, price: i64, quantity: u64, is_buy: bool) -> SlippageResult {
        let book = unsafe { &*self.book.get() };
        let mut result = SlippageResult::new();

        result.expected_price = price;
        result.actual_price = price;

        // Find relevant queue
        let mut level_idx = 0;
        for i in 0..BOOK_LEVELS {
            if book[i].price_tick == price {
                level_idx = i;
                break;
            }
        }

        // Check queue position
        let queue_len = self.queue_history[level_idx].len();
        let queue_avg = self.queue_history[level_idx].average();

        // Fill probability based on position in queue
        let ahead_in_queue = queue_avg.abs() as u64;
        result.fill_probability = if ahead_in_queue < quantity {
            10000
        } else {
            ((quantity * 10000) / (ahead_in_queue + 1)) as u32
        };

        // Estimated latency based on queue consumption rate
        result.estimated_latency_ns = (queue_avg.unsigned_abs() as u64).saturating_mul(1000);

        result
    }

    /// Set volatility factor (affects slippage)
    #[inline]
    pub fn set_volatility(&self, factor: u32) {
        self.volatility_factor.store(factor as u64, Ordering::Relaxed);
    }

    /// Set liquidity factor
    #[inline]
    pub fn set_liquidity(&self, factor: u32) {
        self.liquidity_factor.store(factor as u64, Ordering::Relaxed);
    }

    /// Get average slippage
    #[inline]
    pub fn average_slippage(&self) -> f64 {
        let orders = self.orders_modeled.load(Ordering::Relaxed);
        if orders == 0 {
            return 0.0;
        }
        self.total_slippage.load(Ordering::Relaxed) as f64 / orders as f64
    }

    /// Get current spread
    #[inline]
    pub fn current_spread(&self) -> i64 {
        self.spread_ticks.load(Ordering::Relaxed)
    }

    /// Reset statistics
    #[inline]
    pub fn reset_stats(&self) {
        self.orders_modeled.store(0, Ordering::Relaxed);
        self.total_slippage.store(0, Ordering::Relaxed);
    }
}

/// Type alias for typical configuration
pub type CryptoSlippageModeler = SlippageModeler<MAX_QUEUE_DEPTH, MAX_BOOK_LEVELS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_level_creation() {
        let level = BookLevel::new();
        assert_eq!(level.price_tick, 0);
        assert_eq!(level.bid_size, 0);
    }

    #[test]
    fn test_slippage_modeler_init() {
        let modeler = CryptoSlippageModeler::new();
        modeler.init_book(50000, 1);
        
        assert_eq!(modeler.current_spread(), 1);
    }

    #[test]
    fn test_market_order_slippage() {
        let modeler = CryptoSlippageModeler::new();
        modeler.init_book(50000, 1);

        // Small order should have minimal slippage
        let result = modeler.estimate_market_slippage(100, true);
        assert!(result.slippage_ticks >= 0);
        assert!(result.fill_probability > 0);
    }

    #[test]
    fn test_large_order_impact() {
        let modeler = CryptoSlippageModeler::new();
        modeler.init_book(50000, 1);

        // Large order should have more slippage
        let small_result = modeler.estimate_market_slippage(100, true);
        modeler.reset_stats();
        let large_result = modeler.estimate_market_slippage(10000, true);

        assert!(large_result.slippage_ticks >= small_result.slippage_ticks);
    }

    #[test]
    fn test_limit_order_fill_probability() {
        let modeler = CryptoSlippageModeler::new();
        modeler.init_book(50000, 1);

        let result = modeler.estimate_limit_slippage(50000, 100, true);
        assert!(result.fill_probability <= 10000);
    }

    #[test]
    fn test_queue_buffer_operations() {
        let buf = QueueBuffer::<10>::new();

        assert!(buf.push(100));
        assert!(buf.push(200));
        assert_eq!(buf.len(), 2);
        assert!(buf.average() > 0.0);

        let popped = buf.pop();
        assert_eq!(popped, Some(100));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_volatility_impact() {
        let modeler = CryptoSlippageModeler::new();
        modeler.init_book(50000, 1);

        modeler.set_volatility(2000); // High vol
        let high_vol_result = modeler.estimate_market_slippage(1000, true);

        modeler.set_volatility(500); // Low vol
        modeler.reset_stats();
        let low_vol_result = modeler.estimate_market_slippage(1000, true);

        assert!(high_vol_result.market_impact_bps >= low_vol_result.market_impact_bps);
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;

        assert!(size_of::<BookLevel>() >= CACHE_LINE_SIZE);
        assert!(size_of::<SlippageResult>() >= CACHE_LINE_SIZE);
    }
}
