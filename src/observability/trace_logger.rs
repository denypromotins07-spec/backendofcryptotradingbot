//! Zero-allocation structured tracing writing directly to a ring buffer.
//!
//! This module implements a lock-free trace logger that writes structured
//! events directly to a pre-allocated ring buffer without any heap allocation.
//! Designed for ultra-low-latency HFT workloads.
//!
//! **Latency Target:** < 100ns per trace event.
//! **Memory Limit:** Fixed-size ring buffer, no dynamic allocation.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Size of the trace ring buffer (16MB).
const TRACE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Maximum size of a single trace entry.
const MAX_ENTRY_SIZE: usize = 256;

/// Trace level enumeration.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Critical = 4,
}

/// Trace entry header.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceHeader {
    /// Timestamp (rdtsc cycles).
    pub timestamp: u64,
    /// Trace level.
    pub level: u8,
    /// Source ID (component identifier).
    pub source_id: u8,
    /// Event code.
    pub event_code: u16,
    /// Data length in bytes.
    pub data_len: u16,
    /// Sequence number.
    pub sequence: u64,
    /// Thread ID (simplified).
    pub thread_id: u32,
    /// Reserved padding.
    _padding: [u8; 34],
}

impl TraceHeader {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp: 0,
            level: 0,
            source_id: 0,
            event_code: 0,
            data_len: 0,
            sequence: 0,
            thread_id: 0,
            _padding: [0u8; 34],
        }
    }
}

// Ensure TraceHeader is exactly one cache line.
const _: () = assert!(core::mem::size_of::<TraceHeader>() == CACHE_LINE_SIZE);

/// The main trace logger with a ring buffer.
pub struct TraceLogger {
    /// Pre-allocated ring buffer.
    buffer: *mut u8,
    /// Total buffer size.
    buffer_size: usize,
    /// Write head (next write position).
    write_head: AtomicU64,
    /// Read head (for consumers).
    read_head: AtomicU64,
    /// Sequence counter.
    sequence: AtomicU64,
    /// Flag indicating if the logger is active.
    is_active: AtomicBool,
    /// Count of dropped entries (buffer full).
    dropped_count: AtomicU64,
    /// Padding.
    _padding: [u8; 48],
}

unsafe impl Send for TraceLogger {}
unsafe impl Sync for TraceLogger {}

impl TraceLogger {
    /// Create a new trace logger.
    #[inline]
    pub fn init() -> Result<Self, &'static str> {
        // Allocate aligned memory for the ring buffer
        let layout = std::alloc::Layout::from_size_align(TRACE_BUFFER_SIZE, CACHE_LINE_SIZE)
            .map_err(|_| "Invalid layout")?;
        
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("Failed to allocate trace buffer");
        }

        Ok(Self {
            buffer: ptr,
            buffer_size: TRACE_BUFFER_SIZE,
            write_head: AtomicU64::new(0),
            read_head: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            dropped_count: AtomicU64::new(0),
            _padding: [0u8; 48],
        })
    }

    /// Write a trace event to the ring buffer.
    ///
    /// Returns true if the event was written successfully, false if buffer is full.
    #[inline]
    pub fn trace(&self, level: TraceLevel, source_id: u8, event_code: u16, data: &[u8]) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        let data_len = data.len().min(MAX_ENTRY_SIZE - core::mem::size_of::<TraceHeader>());
        let total_size = core::mem::size_of::<TraceHeader>() + data_len;
        
        // Align to cache line
        let aligned_size = ((total_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

        // Reserve space atomically
        let old_head = self.write_head.fetch_add(aligned_size as u64, Ordering::AcqRel);
        let write_pos = (old_head as usize) % self.buffer_size;
        
        // Check for buffer overflow (simple check)
        let read_pos = self.read_head.load(Ordering::Acquire) as usize % self.buffer_size;
        let available = if write_pos >= read_pos {
            self.buffer_size - (write_pos - read_pos)
        } else {
            read_pos - write_pos
        };

        if available < aligned_size + CACHE_LINE_SIZE {
            // Buffer nearly full, drop this entry
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Get timestamp
        #[cfg(target_arch = "x86_64")]
        let timestamp = unsafe {
            use core::arch::x86_64::_rdtsc;
            _rdtsc()
        };
        #[cfg(not(target_arch = "x86_64"))]
        let timestamp = self.sequence.load(Ordering::Relaxed);

        // Write header
        let header_ptr = unsafe { self.buffer.add(write_pos) as *mut TraceHeader };
        let mut header = TraceHeader::new();
        header.timestamp = timestamp;
        header.level = level as u8;
        header.source_id = source_id;
        header.event_code = event_code;
        header.data_len = data_len as u16;
        header.sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        header.thread_id = 0; // Simplified

        unsafe {
            ptr::write(header_ptr, header);
            
            // Write data
            if data_len > 0 {
                let data_ptr = self.buffer.add(write_pos + core::mem::size_of::<TraceHeader>());
                ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data_len);
            }
        }

        true
    }

    /// Convenience method for debug traces.
    #[inline]
    pub fn debug(&self, source_id: u8, event_code: u16, data: &[u8]) -> bool {
        self.trace(TraceLevel::Debug, source_id, event_code, data)
    }

    /// Convenience method for info traces.
    #[inline]
    pub fn info(&self, source_id: u8, event_code: u16, data: &[u8]) -> bool {
        self.trace(TraceLevel::Info, source_id, event_code, data)
    }

    /// Convenience method for error traces.
    #[inline]
    pub fn error(&self, source_id: u8, event_code: u16, data: &[u8]) -> bool {
        self.trace(TraceLevel::Error, source_id, event_code, data)
    }

    /// Read the next entry from the buffer (for consumers).
    #[inline]
    pub fn read_next(&self) -> Option<(TraceHeader, Vec<u8>)> {
        let read_pos = self.read_head.load(Ordering::Acquire) as usize % self.buffer_size;
        let write_pos = self.write_head.load(Ordering::Acquire) as usize % self.buffer_size;

        if read_pos == write_pos {
            return None; // Buffer empty
        }

        unsafe {
            let header_ptr = self.buffer.add(read_pos) as *const TraceHeader;
            let header = ptr::read(header_ptr);
            
            if header.data_len == 0 {
                return Some((header, Vec::new()));
            }

            let data_ptr = self.buffer.add(read_pos + core::mem::size_of::<TraceHeader>());
            let mut data = vec![0u8; header.data_len as usize];
            ptr::copy_nonoverlapping(data_ptr, data.as_mut_ptr(), header.data_len as usize);

            // Update read head (aligned)
            let aligned_size = ((core::mem::size_of::<TraceHeader>() + header.data_len as usize + CACHE_LINE_SIZE - 1) 
                / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
            self.read_head.fetch_add(aligned_size as u64, Ordering::Release);

            Some((header, data))
        }
    }

    /// Get the number of dropped entries.
    #[inline]
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Acquire)
    }

    /// Shutdown the logger.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
        
        // Deallocate buffer
        if !self.buffer.is_null() {
            let layout = std::alloc::Layout::from_size_align(TRACE_BUFFER_SIZE, CACHE_LINE_SIZE).unwrap();
            unsafe {
                std::alloc::dealloc(self.buffer, layout);
            }
            self.buffer = ptr::null_mut();
        }
    }
}

impl Drop for TraceLogger {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_header_size() {
        assert_eq!(core::mem::size_of::<TraceHeader>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_trace_logger_init() {
        let logger = TraceLogger::init().unwrap();
        assert!(logger.is_active.load(Ordering::Acquire));
        assert_eq!(logger.dropped_count(), 0);
    }

    #[test]
    fn test_write_and_read() {
        let logger = TraceLogger::init().unwrap();
        
        let data = b"Test trace event";
        let result = logger.info(1, 100, data);
        assert!(result);
        
        let (header, read_data) = logger.read_next().unwrap();
        assert_eq!(header.level, TraceLevel::Info as u8);
        assert_eq!(header.source_id, 1);
        assert_eq!(header.event_code, 100);
        assert_eq!(&read_data, data);
    }

    #[test]
    fn test_multiple_levels() {
        let logger = TraceLogger::init().unwrap();
        
        logger.debug(1, 1, b"debug msg").unwrap();
        logger.info(1, 2, b"info msg").unwrap();
        logger.error(1, 3, b"error msg").unwrap();
        
        // Should have 3 entries
        let mut count = 0;
        while logger.read_next().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
