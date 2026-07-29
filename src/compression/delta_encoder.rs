//! Delta Encoder for L2 Order Book Compression
//! 
//! Zero-copy L2 order book delta compression using run-length
//! and dictionary encoding. All operations are lock-free and
//! use pre-allocated buffers.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicU32, AtomicU8, Ordering};

pub type FixedPrice = i64;
pub type FixedQty = i64;

/// Maximum encoded message size (pre-allocated)
const MAX_ENCODED_SIZE: usize = 4096;
const CACHE_LINE: usize = 64;

/// Encoded delta operation types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaOp {
    Insert = 0,
    Update = 1,
    Delete = 2,
    Snapshot = 3,
}

/// Cache-line aligned delta entry (64 bytes)
#[repr(C, align(64))]
pub struct DeltaEntry {
    pub price: AtomicI64,
    pub qty: AtomicI64,
    pub side: AtomicU8, // 0 = bid, 1 = ask
    pub op_type: AtomicU8,
    pub sequence: AtomicU64,
    pub timestamp_ticks: AtomicU64,
    _padding: [u8; 30],
}

impl DeltaEntry {
    pub const fn new() -> Self {
        Self {
            price: AtomicI64::new(0),
            qty: AtomicI64::new(0),
            side: AtomicU8::new(0),
            op_type: AtomicU8::new(0),
            sequence: AtomicU64::new(0),
            timestamp_ticks: AtomicU64::new(0),
            _padding: [0u8; 30],
        }
    }

    #[inline]
    pub fn set(&self, price: FixedPrice, qty: FixedQty, side: u8, op: DeltaOp, seq: u64) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.price.store(price, Ordering::Release);
        self.qty.store(qty, Ordering::Release);
        self.side.store(side, Ordering::Release);
        self.op_type.store(op as u8, Ordering::Release);
        self.sequence.store(seq, Ordering::Release);
        self.timestamp_ticks.store(ts, Ordering::Release);
    }
}

/// Circular buffer for delta entries (lock-free)
#[repr(C, align(64))]
pub struct DeltaBuffer {
    pub entries: [DeltaEntry; 256],
    pub head: AtomicU64,
    pub tail: AtomicU64,
    pub count: AtomicU64,
    _padding: [u8; 32],
}

impl DeltaBuffer {
    pub const fn new() -> Self {
        const INIT: DeltaEntry = DeltaEntry::new();
        Self {
            entries: [INIT; 256],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    #[inline]
    pub fn push(&self, price: FixedPrice, qty: FixedQty, side: u8, op: DeltaOp, seq: u64) {
        let head = self.head.load(Ordering::Acquire);
        let idx = (head as usize) & 255;
        
        self.entries[idx].set(price, qty, side, op, seq);
        
        let tail = self.tail.load(Ordering::Acquire);
        if head - tail >= 256 {
            self.tail.fetch_add(1, Ordering::AcqRel);
        }
        
        self.head.fetch_add(1, Ordering::Release);
        self.count.fetch_add(1, Ordering::AcqRel);
    }

    #[inline]
    pub fn pop(&self) -> Option<(FixedPrice, FixedQty, u8, DeltaOp, u64)> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        
        if tail >= head {
            return None;
        }
        
        let idx = (tail as usize) & 255;
        let entry = &self.entries[idx];
        
        let result = (
            entry.price.load(Ordering::Acquire),
            entry.qty.load(Ordering::Acquire),
            entry.side.load(Ordering::Acquire),
            unsafe { core::mem::transmute::<u8, DeltaOp>(entry.op_type.load(Ordering::Acquire)) },
            entry.sequence.load(Ordering::Acquire),
        );
        
        self.tail.fetch_add(1, Ordering::AcqRel);
        self.count.fetch_sub(1, Ordering::AcqRel);
        
        Some(result)
    }
}

/// Run-length encoder for consecutive identical deltas
#[repr(C, align(64))]
pub struct RunLengthEncoder {
    pub output_buffer: [u8; MAX_ENCODED_SIZE],
    pub output_len: AtomicU64,
    pub input_count: AtomicU64,
    pub compressed_count: AtomicU64,
    _padding: [u8; 32],
}

impl RunLengthEncoder {
    pub const fn new() -> Self {
        Self {
            output_buffer: [0u8; MAX_ENCODED_SIZE],
            output_len: AtomicU64::new(0),
            input_count: AtomicU64::new(0),
            compressed_count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Encode a run of identical operations (simplified)
    #[inline]
    pub fn encode_run(&self, price: FixedPrice, qty: FixedQty, side: u8, op: DeltaOp, run_length: u8) -> usize {
        let mut pos = self.output_len.load(Ordering::Acquire) as usize;
        
        // Header byte: op_type (2 bits) + side (1 bit) + run_length (5 bits)
        let header = ((op as u8) << 6) | ((side & 1) << 5) | (run_length & 0x1F);
        
        if pos + 17 > MAX_ENCODED_SIZE {
            return 0; // Buffer full
        }
        
        unsafe {
            *self.output_buffer.get_unchecked_mut(pos) = header;
            pos += 1;
            
            // Write price (8 bytes)
            core::ptr::write_unaligned(
                self.output_buffer.get_unchecked_mut(pos) as *mut u8 as *mut i64,
                price,
            );
            pos += 8;
            
            // Write qty (8 bytes)
            core::ptr::write_unaligned(
                self.output_buffer.get_unchecked_mut(pos) as *mut u8 as *mut i64,
                qty,
            );
        }
        
        self.output_len.store((pos + 8) as u64, Ordering::Release);
        self.input_count.fetch_add(run_length as u64, Ordering::AcqRel);
        self.compressed_count.fetch_add(1, Ordering::AcqRel);
        
        pos + 8
    }

    #[inline]
    pub fn reset(&self) {
        self.output_len.store(0, Ordering::Release);
    }

    #[inline]
    pub fn get_compression_ratio(&self) -> u64 {
        let input = self.input_count.load(Ordering::Acquire);
        let compressed = self.compressed_count.load(Ordering::Acquire);
        
        if compressed == 0 {
            return 0;
        }
        
        (input * 100) / compressed
    }
}

/// Dictionary encoder for repeated price levels
#[repr(C, align(64))]
pub struct DictionaryEncoder {
    pub price_dict: [AtomicI64; 64],
    pub dict_count: AtomicU32,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    _padding: [u8; 32],
}

impl DictionaryEncoder {
    pub const fn new() -> Self {
        const INIT: AtomicI64 = AtomicI64::new(0);
        Self {
            price_dict: [INIT; 64],
            dict_count: AtomicU32::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Lookup or insert price, return index
    #[inline]
    pub fn lookup_or_insert(&self, price: FixedPrice) -> u8 {
        let count = self.dict_count.load(Ordering::Acquire) as usize;
        
        // Search existing entries
        for i in 0..count.min(64) {
            if self.price_dict[i].load(Ordering::Acquire) == price {
                self.hits.fetch_add(1, Ordering::AcqRel);
                return i as u8;
            }
        }
        
        // Insert new entry if space
        if count < 64 {
            let idx = self.dict_count.fetch_add(1, Ordering::AcqRel) as usize;
            if idx < 64 {
                self.price_dict[idx].store(price, Ordering::Release);
                self.misses.fetch_add(1, Ordering::AcqRel);
                return idx as u8;
            }
        }
        
        self.misses.fetch_add(1, Ordering::AcqRel);
        0xFF // Not found indicator
    }

    #[inline]
    pub fn get_hit_rate(&self) -> u64 {
        let hits = self.hits.load(Ordering::Acquire);
        let misses = self.misses.load(Ordering::Acquire);
        let total = hits + misses;
        
        if total == 0 {
            return 0;
        }
        
        (hits * 100) / total
    }

    #[inline]
    pub fn reset(&self) {
        self.dict_count.store(0, Ordering::Release);
        self.hits.store(0, Ordering::Release);
        self.misses.store(0, Ordering::Release);
        
        const INIT: AtomicI64 = AtomicI64::new(0);
        for i in 0..64 {
            self.price_dict[i].store(0, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_buffer() {
        let buffer = DeltaBuffer::new();
        
        buffer.push(50_000_000_000i64, 1_000_000_000i64, 0, DeltaOp::Insert, 1);
        buffer.push(50_100_000_000i64, 500_000_000i64, 1, DeltaOp::Update, 2);
        
        assert_eq!(buffer.count.load(Ordering::Acquire), 2);
        
        let popped = buffer.pop();
        assert!(popped.is_some());
        let (price, qty, side, op, seq) = popped.unwrap();
        assert_eq!(price, 50_000_000_000i64);
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_run_length_encoder() {
        let encoder = RunLengthEncoder::new();
        
        encoder.encode_run(50_000_000_000i64, 1_000_000_000i64, 0, DeltaOp::Insert, 5);
        
        assert_eq!(encoder.input_count.load(Ordering::Acquire), 5);
        assert_eq!(encoder.compressed_count.load(Ordering::Acquire), 1);
        assert_eq!(encoder.get_compression_ratio(), 500); // 5:1 = 500%
    }

    #[test]
    fn test_dictionary_encoder() {
        let encoder = DictionaryEncoder::new();
        
        // First insertion - miss
        let idx1 = encoder.lookup_or_insert(50_000_000_000i64);
        assert_eq!(idx1, 0);
        
        // Same price - hit
        let idx2 = encoder.lookup_or_insert(50_000_000_000i64);
        assert_eq!(idx2, 0);
        
        // Different price - miss
        let idx3 = encoder.lookup_or_insert(50_100_000_000i64);
        assert_eq!(idx3, 1);
        
        assert_eq!(encoder.get_hit_rate(), 33); // 1 hit / 3 total
    }
}
