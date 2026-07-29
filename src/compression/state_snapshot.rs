//! Ultra-Fast State Snapshot and Recovery Serialization
//! 
//! Provides zero-downtime hot-restart capability with lock-free
//! circular buffers for historical state hashes.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

const SNAPSHOT_BUFFER_SIZE: usize = 1024 * 1024; // 1MB pre-allocated
const HASH_HISTORY_SIZE: usize = 64;
const HASH_MASK: usize = HASH_HISTORY_SIZE - 1;

/// Cache-line aligned snapshot header (64 bytes)
#[repr(C, align(64))]
pub struct SnapshotHeader {
    pub magic: AtomicU64,
    pub version: AtomicU64,
    pub timestamp_ticks: AtomicU64,
    pub sequence_num: AtomicU64,
    pub payload_size: AtomicU64,
    pub checksum: AtomicU64,
    pub state_flags: AtomicU8,
    _padding: [u8; 39],
}

impl SnapshotHeader {
    pub const fn new() -> Self {
        Self {
            magic: AtomicU64::new(0x534E415053484F54), // "SNAPSHOT"
            version: AtomicU64::new(1),
            timestamp_ticks: AtomicU64::new(0),
            sequence_num: AtomicU64::new(0),
            payload_size: AtomicU64::new(0),
            checksum: AtomicU64::new(0),
            state_flags: AtomicU8::new(0),
            _padding: [0u8; 39],
        }
    }

    #[inline]
    pub fn set_timestamp(&self) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        self.timestamp_ticks.store(ts, Ordering::Release);
    }

    #[inline]
    pub fn get_timestamp(&self) -> u64 {
        self.timestamp_ticks.load(Ordering::Acquire)
    }
}

/// State snapshot manager with circular buffer
#[repr(C, align(64))]
pub struct StateSnapshot {
    pub header: SnapshotHeader,
    pub data_buffer: [u8; SNAPSHOT_BUFFER_SIZE],
    pub write_pos: AtomicU64,
    pub read_pos: AtomicU64,
    pub snapshot_count: AtomicU64,
    pub is_valid: AtomicU8,
    _padding: [u8; 31],
}

impl StateSnapshot {
    pub const fn new() -> Self {
        Self {
            header: SnapshotHeader::new(),
            data_buffer: [0u8; SNAPSHOT_BUFFER_SIZE],
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            snapshot_count: AtomicU64::new(0),
            is_valid: AtomicU8::new(0),
            _padding: [0u8; 31],
        }
    }

    /// Write data to snapshot buffer (lock-free, overwrites if full)
    #[inline]
    pub fn write(&self, data: &[u8]) -> bool {
        if data.len() > SNAPSHOT_BUFFER_SIZE {
            return false;
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data_buffer.as_mut_ptr(),
                data.len(),
            );
        }

        self.header.payload_size.store(data.len() as u64, Ordering::Release);
        self.header.set_timestamp();
        
        let seq = self.header.sequence_num.fetch_add(1, Ordering::AcqRel);
        self.write_pos.store(seq + 1, Ordering::Release);
        self.snapshot_count.fetch_add(1, Ordering::AcqRel);
        self.is_valid.store(1, Ordering::Release);

        true
    }

    /// Read current snapshot data
    #[inline]
    pub fn read(&self) -> Option<&[u8]> {
        if self.is_valid.load(Ordering::Acquire) == 0 {
            return None;
        }

        let size = self.header.payload_size.load(Ordering::Acquire) as usize;
        if size == 0 || size > SNAPSHOT_BUFFER_SIZE {
            return None;
        }

        unsafe {
            Some(core::slice::from_raw_parts(self.data_buffer.as_ptr(), size))
        }
    }

    /// Calculate simple checksum (XOR-based, fast)
    #[inline]
    pub fn calculate_checksum(&self) -> u64 {
        let size = self.header.payload_size.load(Ordering::Acquire) as usize;
        let mut checksum: u64 = 0;

        for i in 0..size.min(SNAPSHOT_BUFFER_SIZE) {
            let byte = unsafe { *self.data_buffer.get_unchecked(i) };
            checksum ^= (byte as u64) << ((i % 8) * 8);
        }

        self.header.checksum.store(checksum, Ordering::Release);
        checksum
    }

    /// Verify snapshot integrity
    #[inline]
    pub fn verify(&self) -> bool {
        if self.is_valid.load(Ordering::Acquire) == 0 {
            return false;
        }

        let stored_checksum = self.header.checksum.load(Ordering::Acquire);
        let calculated = self.calculate_checksum();

        stored_checksum == calculated
    }

    /// Get snapshot statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.snapshot_count.load(Ordering::Acquire),
            self.header.sequence_num.load(Ordering::Acquire),
            self.header.payload_size.load(Ordering::Acquire),
        )
    }
}

/// Historical state hash tracker (circular buffer)
#[repr(C, align(64))]
pub struct HashHistory {
    pub hashes: [AtomicU64; HASH_HISTORY_SIZE],
    pub timestamps: [AtomicU64; HASH_HISTORY_SIZE],
    pub head: AtomicU64,
    pub count: AtomicU64,
    _padding: [u8; 32],
}

impl HashHistory {
    pub const fn new() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            hashes: [INIT; HASH_HISTORY_SIZE],
            timestamps: [INIT; HASH_HISTORY_SIZE],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Record a state hash (lock-free circular buffer)
    #[inline]
    pub fn record(&self, hash: u64) {
        let ts = unsafe { core::arch::x86_64::_rdtsc() } as u64;
        let head = self.head.fetch_add(1, Ordering::AcqRel);
        let idx = (head as usize) & HASH_MASK;

        self.hashes[idx].store(hash, Ordering::Release);
        self.timestamps[idx].store(ts, Ordering::Release);

        // Update count (cap at buffer size)
        let count = self.count.load(Ordering::Acquire);
        if count < HASH_HISTORY_SIZE as u64 {
            self.count.store(count + 1, Ordering::Release);
        }
    }

    /// Get the Nth most recent hash
    #[inline]
    pub fn get_hash(&self, n: usize) -> Option<u64> {
        if n == 0 || n > HASH_HISTORY_SIZE {
            return None;
        }

        let head = self.head.load(Ordering::Acquire);
        let count = self.count.load(Ordering::Acquire);

        if n > count as usize {
            return None;
        }

        let idx = ((head - n as u64) as usize) & HASH_MASK;
        Some(self.hashes[idx].load(Ordering::Acquire))
    }

    /// Check if a hash exists in history (for duplicate detection)
    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        let count = self.count.load(Ordering::Acquire) as usize;
        let head = self.head.load(Ordering::Acquire) as usize;

        for i in 0..count {
            let idx = ((head - 1 - i) & HASH_MASK) as usize;
            if self.hashes[idx].load(Ordering::Acquire) == hash {
                return true;
            }
        }

        false
    }

    /// Get time delta between last two hashes (in ticks)
    #[inline]
    pub fn get_last_delta(&self) -> u64 {
        let head = self.head.load(Ordering::Acquire);
        let count = self.count.load(Ordering::Acquire);

        if count < 2 {
            return 0;
        }

        let idx1 = ((head - 1) as usize) & HASH_MASK;
        let idx2 = ((head - 2) as usize) & HASH_MASK;

        let ts1 = self.timestamps[idx1].load(Ordering::Acquire);
        let ts2 = self.timestamps[idx2].load(Ordering::Acquire);

        ts1.wrapping_sub(ts2)
    }
}

/// Simple FNV-1a hash for state fingerprinting
#[inline]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_write_read() {
        let snapshot = StateSnapshot::new();
        let data = b"Test snapshot data for state recovery";
        
        assert!(snapshot.write(data));
        
        let read_data = snapshot.read();
        assert!(read_data.is_some());
        assert_eq!(read_data.unwrap(), data);
    }

    #[test]
    fn test_snapshot_checksum() {
        let snapshot = StateSnapshot::new();
        let data = b"Data to checksum";
        
        snapshot.write(data);
        let checksum = snapshot.calculate_checksum();
        
        assert!(checksum != 0);
        assert!(snapshot.verify());
    }

    #[test]
    fn test_hash_history() {
        let history = HashHistory::new();
        
        history.record(0x12345678);
        history.record(0xABCDEF00);
        history.record(0xDEADBEEF);
        
        assert_eq!(history.get_hash(1), Some(0xDEADBEEF));
        assert_eq!(history.get_hash(2), Some(0xABCDEF00));
        assert_eq!(history.get_hash(3), Some(0x12345678));
        
        assert!(history.contains(0xABCDEF00));
        assert!(!history.contains(0xFFFFFFFF));
    }

    #[test]
    fn test_fnv1a_hash() {
        let data1 = b"Hello";
        let data2 = b"Hello";
        let data3 = b"World";
        
        assert_eq!(fnv1a_hash(data1), fnv1a_hash(data2));
        assert_ne!(fnv1a_hash(data1), fnv1a_hash(data3));
    }
}
