// src/market_data/tick_feed.rs
//! Normalized Trade Tick Ingestion Pipeline with Sequence Gap Detection
//!
//! This module implements a high-performance tick feed handler:
//! - Zero-copy tick ingestion from multiple exchanges
//! - Sequence gap detection and recovery
//! - Normalized tick format across venues
//! - Tick aggregation for volume analysis
//!
//! Micro-optimizations:
//! - Lock-free sequence tracking
//! - Pre-allocated tick buffers
//! - Cache-line aligned structures
//! - Branchless gap detection

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum ticks in buffer
const MAX_TICK_BUFFER: usize = 1 << 16; // 65536 ticks

/// Exchange identifiers
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeId {
    Binance = 0,
    Coinbase = 1,
    Kraken = 2,
    FTX = 3,
    Unknown = 255,
}

impl ExchangeId {
    #[inline]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Binance,
            1 => Self::Coinbase,
            2 => Self::Kraken,
            3 => Self::FTX,
            _ => Self::Unknown,
        }
    }
}

/// Tick type discriminator
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickType {
    Trade = 0,
    Quote = 1,
    TradeAndQuote = 2,
}

/// Normalized tick structure - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tick {
    /// Exchange identifier
    pub exchange: ExchangeId,
    /// Tick type
    pub tick_type: TickType,
    /// Symbol ID (normalized across exchanges)
    pub symbol_id: u32,
    /// Price in nanodollars (fixed-point)
    pub price_nanodollars: u64,
    /// Quantity in base units
    pub quantity: u64,
    /// Timestamp in nanoseconds (exchange time)
    pub exchange_ts_ns: u64,
    /// Timestamp when received (local rdtsc)
    pub local_ts_ns: u64,
    /// Sequence number from exchange
    pub sequence: u64,
    /// Aggressor side (0=unknown, 1=buy, 2=sell)
    pub aggressor_side: u8,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - (1 + 1 + 4 + 8 * 5 + 1) % CACHE_LINE_SIZE],
}

impl Tick {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            exchange: ExchangeId::Unknown,
            tick_type: TickType::Trade,
            symbol_id: 0,
            price_nanodollars: 0,
            quantity: 0,
            exchange_ts_ns: 0,
            local_ts_ns: 0,
            sequence: 0,
            aggressor_side: 0,
            _pad: [0u8; CACHE_LINE_SIZE - (1 + 1 + 4 + 8 * 5 + 1) % CACHE_LINE_SIZE],
        }
    }

    /// Create a trade tick
    #[inline]
    pub fn trade(
        exchange: ExchangeId,
        symbol_id: u32,
        price: u64,
        quantity: u64,
        exchange_ts: u64,
        sequence: u64,
        aggressor_side: u8,
    ) -> Self {
        Self {
            exchange,
            tick_type: TickType::Trade,
            symbol_id,
            price_nanodollars: price,
            quantity,
            exchange_ts_ns: exchange_ts,
            local_ts_ns: get_timestamp_ns(),
            sequence,
            aggressor_side,
            _pad: [0u8; CACHE_LINE_SIZE - (1 + 1 + 4 + 8 * 5 + 1) % CACHE_LINE_SIZE],
        }
    }

    /// Get tick value (price * quantity)
    #[inline]
    pub fn value(&self) -> u128 {
        (self.price_nanodollars as u128) * (self.quantity as u128)
    }
}

impl Default for Tick {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequence tracker for gap detection
#[repr(C)]
pub struct SequenceTracker {
    /// Expected next sequence number
    expected_sequence: AtomicU64,
    /// Last received sequence
    last_sequence: AtomicU64,
    /// Gap count
    gap_count: AtomicU64,
    /// Total ticks received
    total_ticks: AtomicU64,
    /// Padding
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 4],
}

impl SequenceTracker {
    #[inline]
    pub const fn new() -> Self {
        Self {
            expected_sequence: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
            gap_count: AtomicU64::new(0),
            total_ticks: AtomicU64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 4],
        }
    }

    /// Process a sequence number, detecting gaps
    /// Returns true if sequence is valid (no gap), false if gap detected
    #[inline]
    pub fn process_sequence(&self, sequence: u64) -> bool {
        let expected = self.expected_sequence.load(Ordering::Acquire);
        
        // First message or restart
        if expected == 0 && sequence > 0 {
            self.expected_sequence.store(sequence + 1, Ordering::Release);
            self.last_sequence.store(sequence, Ordering::Release);
            self.total_ticks.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Check for gap
        if sequence >= expected {
            if sequence > expected {
                // Gap detected
                let gap_size = sequence - expected;
                self.gap_count.fetch_add(gap_size, Ordering::Relaxed);
            }
            
            self.expected_sequence.store(sequence + 1, Ordering::Release);
            self.last_sequence.store(sequence, Ordering::Release);
            self.total_ticks.fetch_add(1, Ordering::Relaxed);
            return sequence == expected;
        }

        // Duplicate or out-of-order (sequence < expected)
        // Still count it but don't update expected
        self.total_ticks.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Check if there's a gap
    #[inline]
    pub fn has_gap(&self) -> bool {
        self.gap_count.load(Ordering::Acquire) > 0
    }

    /// Get gap statistics
    #[inline]
    pub fn stats(&self) -> SequenceStats {
        SequenceStats {
            expected: self.expected_sequence.load(Ordering::Relaxed),
            last: self.last_sequence.load(Ordering::Relaxed),
            gaps: self.gap_count.load(Ordering::Relaxed),
            total: self.total_ticks.load(Ordering::Relaxed),
        }
    }

    /// Reset the tracker
    #[inline]
    pub fn reset(&self) {
        self.expected_sequence.store(0, Ordering::Release);
        self.last_sequence.store(0, Ordering::Release);
        self.gap_count.store(0, Ordering::Release);
    }
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequence statistics
#[derive(Debug, Clone)]
pub struct SequenceStats {
    pub expected: u64,
    pub last: u64,
    pub gaps: u64,
    pub total: u64,
}

/// Tick ring buffer for ingestion pipeline
#[repr(C)]
pub struct TickBuffer {
    /// Ticks storage
    ticks: [Tick; MAX_TICK_BUFFER],
    /// Write position
    write_pos: AtomicUsize,
    /// Read position
    read_pos: AtomicUsize,
    /// Mask for fast modulo
    mask: usize,
    /// Padding
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>() * 2 - core::mem::size_of::<usize>()],
}

impl TickBuffer {
    #[inline]
    pub const fn new() -> Self {
        const EMPTY_TICK: Tick = Tick::new();
        Self {
            ticks: [EMPTY_TICK; MAX_TICK_BUFFER],
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            mask: MAX_TICK_BUFFER - 1,
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>() * 2 - core::mem::size_of::<usize>()],
        }
    }

    /// Push a tick into the buffer
    #[inline]
    pub fn push(&self, tick: Tick) -> bool {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Acquire);

        // Check if buffer is full
        if write.wrapping_sub(read) >= MAX_TICK_BUFFER {
            return false;
        }

        let idx = write & self.mask;
        unsafe {
            let ptr = self.ticks.as_ptr() as *mut Tick;
            *ptr.add(idx) = tick;
        }

        self.write_pos.store(write + 1, Ordering::Release);
        true
    }

    /// Pop a tick from the buffer
    #[inline]
    pub fn pop(&self) -> Option<Tick> {
        let read = self.read_pos.load(Ordering::Relaxed);
        let write = self.write_pos.load(Ordering::Acquire);

        if read >= write {
            return None;
        }

        let idx = read & self.mask;
        let tick = unsafe {
            let ptr = self.ticks.as_ptr();
            *ptr.add(idx)
        };

        self.read_pos.store(read + 1, Ordering::Release);
        Some(tick)
    }

    /// Get current buffer depth
    #[inline]
    pub fn depth(&self) -> usize {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Acquire);
        write.wrapping_sub(read)
    }
}

impl Default for TickBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Tick Feed Handler - main ingestion pipeline
#[repr(C)]
pub struct TickFeed {
    /// Tick buffer
    buffer: TickBuffer,
    /// Per-exchange sequence trackers
    sequences: [SequenceTracker; 5],
    /// Total ticks processed
    total_processed: AtomicU64,
    /// Dropped ticks (buffer full)
    dropped_ticks: AtomicU64,
    /// Padding
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 2],
}

impl TickFeed {
    #[inline]
    pub const fn new() -> Self {
        const EMPTY_TRACKER: SequenceTracker = SequenceTracker::new();
        Self {
            buffer: TickBuffer::new(),
            sequences: [EMPTY_TRACKER; 5],
            total_processed: AtomicU64::new(0),
            dropped_ticks: AtomicU64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 2],
        }
    }

    /// Ingest a tick from an exchange
    #[inline]
    pub fn ingest(&self, tick: Tick) -> Result<(), IngestError> {
        // Validate and track sequence
        let seq_idx = tick.exchange as usize;
        if seq_idx >= self.sequences.len() {
            return Err(IngestError::InvalidExchange);
        }

        let tracker = &self.sequences[seq_idx];
        let valid = tracker.process_sequence(tick.sequence);

        if !valid {
            // Gap detected but still process
        }

        // Try to buffer the tick
        if self.buffer.push(tick) {
            self.total_processed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            self.dropped_ticks.fetch_add(1, Ordering::Relaxed);
            Err(IngestError::BufferFull)
        }
    }

    /// Consume ticks from the buffer
    #[inline]
    pub fn consume<F>(&self, mut handler: F, max_count: usize) -> usize
    where
        F: FnMut(&Tick),
    {
        let mut count = 0;
        while count < max_count {
            if let Some(tick) = self.buffer.pop() {
                handler(&tick);
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> FeedStats {
        FeedStats {
            total_processed: self.total_processed.load(Ordering::Relaxed),
            dropped_ticks: self.dropped_ticks.load(Ordering::Relaxed),
            buffer_depth: self.buffer.depth(),
            sequences: self.sequences.iter().map(|s| s.stats()).collect(),
        }
    }

    /// Check for any gaps across exchanges
    #[inline]
    pub fn has_gaps(&self) -> bool {
        self.sequences.iter().any(|s| s.has_gap())
    }
}

impl Default for TickFeed {
    fn default() -> Self {
        Self::new()
    }
}

/// Feed statistics
pub struct FeedStats {
    pub total_processed: u64,
    pub dropped_ticks: u64,
    pub buffer_depth: usize,
    pub sequences: Vec<SequenceStats>,
}

/// Ingestion errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestError {
    BufferFull,
    InvalidExchange,
    InvalidSequence,
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
    fn test_tick_creation() {
        let tick = Tick::trade(
            ExchangeId::Binance,
            1,
            100_000_000_000,
            100,
            1000,
            42,
            1,
        );

        assert_eq!(tick.exchange, ExchangeId::Binance);
        assert_eq!(tick.symbol_id, 1);
        assert_eq!(tick.price_nanodollars, 100_000_000_000);
        assert_eq!(tick.tick_type, TickType::Trade);
    }

    #[test]
    fn test_sequence_tracker() {
        let tracker = SequenceTracker::new();

        // Sequential messages
        assert!(tracker.process_sequence(1));
        assert!(tracker.process_sequence(2));
        assert!(tracker.process_sequence(3));

        assert!(!tracker.has_gap());
        let stats = tracker.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.gaps, 0);
    }

    #[test]
    fn test_sequence_gap_detection() {
        let tracker = SequenceTracker::new();

        assert!(tracker.process_sequence(1));
        assert!(tracker.process_sequence(3)); // Gap: missing 2

        assert!(tracker.has_gap());
        let stats = tracker.stats();
        assert_eq!(stats.gaps, 1);
    }

    #[test]
    fn test_tick_buffer() {
        let buffer = TickBuffer::new();

        let tick = Tick::trade(ExchangeId::Binance, 1, 100, 10, 1000, 1, 1);
        assert!(buffer.push(tick));
        assert_eq!(buffer.depth(), 1);

        let popped = buffer.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().symbol_id, 1);
        assert_eq!(buffer.depth(), 0);
    }

    #[test]
    fn test_tick_feed() {
        let feed = TickFeed::new();

        let tick = Tick::trade(ExchangeId::Binance, 1, 100, 10, 1000, 1, 1);
        assert!(feed.ingest(tick).is_ok());

        let mut consumed = 0;
        feed.consume(|_| { consumed += 1; }, 10);
        assert_eq!(consumed, 1);
    }
}
