//! Memory-mapped `SOUL.md` parser and writer for persistent self-learning state.
//! 
//! This module implements a zero-copy, non-blocking interface to the bot's
//! persistent memory file (`SOUL.md`). It uses `memmap2` for direct memory mapping
//! and atomic operations to ensure thread-safe updates without stalling the trading thread.
//!
//! **Latency Target:** < 50ns for reads, < 5µs for async writes.
//! **Memory Limit:** Strictly bounded by the mapped file size (configurable, default 64MB).

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;
use core::slice;

/// Cache line padding constant to prevent false sharing.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum size of the SOUL.md memory map (64MB).
const MAX_SOUL_SIZE: usize = 64 * 1024 * 1024;

/// Header structure for the SOUL.md memory map.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SoulHeader {
    /// Magic number for validation: 0x534F554C ("SOUL")
    pub magic: u32,
    /// Version of the SOUL.md format.
    pub version: u32,
    /// Total size of the mapped region in bytes.
    pub total_size: u64,
    /// Current write offset (head) in the log section.
    pub write_offset: AtomicU64,
    /// Sequence number for the last committed entry.
    pub sequence_num: AtomicU64,
    /// Flag indicating if the file is currently being compacted.
    pub is_compacting: AtomicBool,
    /// Reserved padding to reach 64 bytes.
    _padding: [u8; 38],
}

impl SoulHeader {
    /// Create a new header with default values.
    #[inline]
    pub const fn new() -> Self {
        Self {
            magic: 0x534F554C, // "SOUL" in little-endian
            version: 1,
            total_size: MAX_SOUL_SIZE as u64,
            write_offset: AtomicU64::new(core::mem::size_of::<Self>() as u64),
            sequence_num: AtomicU64::new(0),
            is_compacting: AtomicBool::new(false),
            _padding: [0u8; 38],
        }
    }

    /// Verify the magic number and version.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.magic == 0x534F554C && self.version == 1
    }
}

// Ensure header is exactly one cache line.
const _: () = assert!(core::mem::size_of::<SoulHeader>() == CACHE_LINE_SIZE);

/// The main SOUL.md memory manager.
/// 
/// Provides lock-free read access and serialized async write access.
pub struct SoulMemory {
    /// Pointer to the start of the memory-mapped region.
    base_ptr: *mut u8,
    /// Total size of the mapped region.
    mapped_size: usize,
    /// Reference to the header (cast from base_ptr).
    header: *mut SoulHeader,
    /// Flag indicating if the mapping is active.
    is_active: AtomicBool,
    /// Count of pending write operations.
    pending_writes: AtomicU64,
}

unsafe impl Send for SoulMemory {}
unsafe impl Sync for SoulMemory {}

impl SoulMemory {
    /// Initialize the SOUL.md memory map.
    /// 
    /// In a real implementation, this would use `memmap2::MmapMut` to map a file.
    /// For this zero-allocation prototype, we simulate the mapping with a static buffer
    /// or pre-allocated heap block that is never resized.
    #[inline]
    pub fn init() -> Result<Self, &'static str> {
        // Simulate memory mapping with a pre-allocated aligned buffer.
        // In production: let file = OpenOptions::new().read(true).write(true).open("SOUL.md")?;
        //                let mut mmap = MmapOptions::new().len(MAX_SOUL_SIZE).map_mut(&file)?;
        
        // Allocate aligned memory manually to avoid std::Vec reallocations
        let layout = std::alloc::Layout::from_size_align(MAX_SOUL_SIZE, CACHE_LINE_SIZE)
            .map_err(|_| "Invalid layout")?;
        
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("Failed to allocate SOUL.md memory map");
        }

        // Initialize header at the beginning
        let header = ptr as *mut SoulHeader;
        unsafe {
            ptr::write(header, SoulHeader::new());
        }

        Ok(Self {
            base_ptr: ptr,
            mapped_size: MAX_SOUL_SIZE,
            header,
            is_active: AtomicBool::new(true),
            pending_writes: AtomicU64::new(0),
        })
    }

    /// Read an entry from the SOUL.md log by sequence number.
    /// 
    /// Zero-copy read: returns a slice directly into the mapped memory.
    #[inline]
    pub fn read_entry(&self, seq: u64) -> Option<&[u8]> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        let header = unsafe { &*self.header };
        if seq >= header.sequence_num.load(Ordering::Acquire) {
            return None;
        }

        // Calculate offset based on sequence number (simplified: fixed-size entries)
        // In production, this would parse a variable-length log format.
        let entry_size = 256; // Fixed entry size for O(1) lookup
        let data_start = core::mem::size_of::<SoulHeader>() + (seq as usize * entry_size);
        
        if data_start + entry_size > self.mapped_size {
            return None;
        }

        unsafe {
            Some(slice::from_raw_parts(self.base_ptr.add(data_start), entry_size))
        }
    }

    /// Append a new entry to the SOUL.md log asynchronously.
    /// 
    /// Uses atomic operations to reserve space and write without locks.
    /// The actual disk sync happens in a background thread (not shown).
    #[inline]
    pub fn append_async(&self, data: &[u8]) -> Result<u64, &'static str> {
        if !self.is_active.load(Ordering::Acquire) {
            return Err("SOUL.md memory map is inactive");
        }

        if data.len() > 256 {
            return Err("Entry exceeds maximum size");
        }

        let header = unsafe { &*self.header };
        
        // Atomically increment sequence number to reserve a slot
        let seq = header.sequence_num.fetch_add(1, Ordering::AcqRel);
        
        // Calculate write offset
        let entry_size = 256;
        let offset = core::mem::size_of::<SoulHeader>() + (seq as usize * entry_size);
        
        if offset + data.len() > self.mapped_size {
            // Trigger compaction flag (handled by background thread)
            header.is_compacting.store(true, Ordering::Release);
            return Err("SOUL.md log full, compaction required");
        }

        // Write data directly to mapped memory
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.base_ptr.add(offset),
                data.len(),
            );
            // Zero-fill the rest of the entry
            ptr::write_bytes(self.base_ptr.add(offset + data.len()), 0, entry_size - data.len());
        }

        // Update write offset atomically
        header.write_offset.store((offset + entry_size) as u64, Ordering::Release);
        
        self.pending_writes.fetch_sub(1, Ordering::Relaxed);

        Ok(seq)
    }

    /// Get the current sequence number (latest entry).
    #[inline]
    pub fn latest_sequence(&self) -> u64 {
        if !self.is_active.load(Ordering::Acquire) {
            return 0;
        }
        unsafe { (*self.header).sequence_num.load(Ordering::Acquire) }
    }

    /// Check if compaction is needed.
    #[inline]
    pub fn needs_compaction(&self) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }
        unsafe { (*self.header).is_compacting.load(Ordering::Acquire) }
    }

    /// Shutdown and unmap the memory region.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
        
        // Wait for pending writes to complete (spin-wait for simplicity)
        while self.pending_writes.load(Ordering::Acquire) > 0 {
            core::hint::spin_loop();
        }

        // Deallocate memory
        if !self.base_ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(MAX_SOUL_SIZE, CACHE_LINE_SIZE).unwrap();
            unsafe {
                std::alloc::dealloc(self.base_ptr, layout);
            }
            self.base_ptr = ptr::null_mut();
        }
    }
}

impl Drop for SoulMemory {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soul_header_size() {
        assert_eq!(core::mem::size_of::<SoulHeader>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_soul_memory_init() {
        let soul = SoulMemory::init().unwrap();
        assert!(soul.is_active.load(Ordering::Acquire));
        assert_eq!(soul.latest_sequence(), 0);
    }

    #[test]
    fn test_append_and_read() {
        let soul = SoulMemory::init().unwrap();
        let data = b"Test entry for SOUL.md learning";
        let seq = soul.append_async(data).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(soul.latest_sequence(), 1);
        
        let read_data = soul.read_entry(0).unwrap();
        assert_eq!(&read_data[..data.len()], data);
    }
}
