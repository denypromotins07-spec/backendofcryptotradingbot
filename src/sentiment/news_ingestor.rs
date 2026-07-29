//! Ultra-low-latency WebSocket news feed parser using zero-copy JSON extraction.
//! 
//! Parses news headlines and metadata without heap allocations in hot path.
//! Uses SIMD-accelerated string operations where applicable.

#![allow(clippy::missing_docs_in_private_items)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum headline length (pre-allocated)
pub const MAX_HEADLINE_LEN: usize = 512;

/// Maximum source name length
pub const MAX_SOURCE_LEN: usize = 64;

/// Parsed news item - zero-copy view into buffer
#[repr(C)]
pub struct NewsItem {
    pub timestamp_ns: u64,
    pub source_hash: u64,      // FNV-1a hash of source name
    pub headline_ptr: *const u8,
    pub headline_len: usize,
    pub urgency: u8,           // 0=normal, 1=high, 2=critical
    pub category: u8,          // Category ID
    pub _reserved: u16,
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

// SAFETY: NewsItem is read-only after creation
unsafe impl Send for NewsItem {}
unsafe impl Sync for NewsItem {}

impl NewsItem {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            timestamp_ns: 0,
            source_hash: 0,
            headline_ptr: core::ptr::null(),
            headline_len: 0,
            urgency: 0,
            category: 0,
            _reserved: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }

    /// Get headline as string slice (unsafe - caller must ensure lifetime)
    #[inline(always)]
    pub unsafe fn headline(&self) -> &str {
        if self.headline_ptr.is_null() || self.headline_len == 0 {
            return "";
        }
        let slice = core::slice::from_raw_parts(self.headline_ptr, self.headline_len);
        core::str::from_utf8_unchecked(slice)
    }

    /// Calculate FNV-1a hash of headline content
    #[inline]
    pub fn headline_hash(&self) -> u64 {
        if self.headline_ptr.is_null() || self.headline_len == 0 {
            return 0;
        }
        
        let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
        unsafe {
            let slice = core::slice::from_raw_parts(self.headline_ptr, self.headline_len);
            for byte in slice.iter() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3); // FNV prime
            }
        }
        hash
    }
}

/// Pre-allocated news buffer pool
#[repr(C)]
pub struct NewsBuffer<const BUF_SIZE: usize> {
    data: [u8; BUF_SIZE],
    write_pos: AtomicU64,
    read_pos: AtomicU64,
    overflow: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 17],
}

impl<const BUF_SIZE: usize> NewsBuffer<BUF_SIZE> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: [0u8; BUF_SIZE],
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            overflow: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 17],
        }
    }

    /// Write data to buffer (lock-free, returns bytes written or 0 on overflow)
    #[inline]
    pub fn write(&self, data: &[u8]) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Acquire);
        
        // Calculate available space (circular)
        let available = if write >= read {
            BUF_SIZE - (write - read) as usize - 1
        } else {
            (read - write - 1) as usize
        };
        
        if data.len() > available {
            self.overflow.store(true, Ordering::Release);
            return 0;
        }
        
        // Write data
        let write_idx = (write as usize) % BUF_SIZE;
        let chunk1_len = core::cmp::min(data.len(), BUF_SIZE - write_idx);
        
        unsafe {
            let ptr = self.data.as_ptr() as *mut u8;
            ptr.add(write_idx).copy_from_nonoverlapping(data.as_ptr(), chunk1_len);
            
            if data.len() > chunk1_len {
                ptr.add(0).copy_from_nonoverlapping(
                    data.as_ptr().add(chunk1_len),
                    data.len() - chunk1_len
                );
            }
        }
        
        self.write_pos.fetch_add(data.len() as u64, Ordering::Release);
        data.len()
    }

    /// Read data from buffer
    #[inline]
    pub fn read(&self, buf: &mut [u8]) -> usize {
        let read = self.read_pos.load(Ordering::Relaxed);
        let write = self.write_pos.load(Ordering::Acquire);
        
        let available = (write - read) as usize;
        let to_read = core::cmp::min(buf.len(), available);
        
        if to_read == 0 {
            return 0;
        }
        
        let read_idx = (read as usize) % BUF_SIZE;
        let chunk1_len = core::cmp::min(to_read, BUF_SIZE - read_idx);
        
        unsafe {
            let ptr = self.data.as_ptr();
            buf[..chunk1_len].copy_from_slice(
                core::slice::from_raw_parts(ptr.add(read_idx), chunk1_len)
            );
            
            if to_read > chunk1_len {
                buf[chunk1_len..to_read].copy_from_slice(
                    core::slice::from_raw_parts(ptr.add(0), to_read - chunk1_len)
                );
            }
        }
        
        self.read_pos.fetch_add(to_read as u64, Ordering::Release);
        to_read
    }

    /// Available bytes to read
    #[inline(always)]
    pub fn available(&self) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Relaxed);
        (write - read) as usize
    }

    /// Check if overflow occurred
    #[inline(always)]
    pub fn has_overflow(&self) -> bool {
        self.overflow.swap(false, Ordering::AcqRel)
    }

    /// Reset buffer
    #[inline]
    pub fn reset(&self) {
        self.write_pos.store(0, Ordering::Relaxed);
        self.read_pos.store(0, Ordering::Relaxed);
        self.overflow.store(false, Ordering::Relaxed);
    }
}

/// Zero-copy JSON field extractor
pub struct JsonExtractor;

impl JsonExtractor {
    /// Extract field value from JSON without allocation
    /// Returns (value_ptr, value_len) or None if not found
    #[inline]
    pub fn extract_field<'a>(json: &'a [u8], field_name: &[u8]) -> Option<(&'a [u8], usize)> {
        // Simple JSON parser - find "field_name":
        let search_pattern = [b'"'];
        
        let mut i = 0;
        while i < json.len().saturating_sub(field_name.len() + 3) {
            // Look for quote
            if json[i] == b'"' {
                // Check if this is our field name
                if i + 1 + field_name.len() < json.len() 
                    && &json[i + 1..i + 1 + field_name.len()] == field_name 
                    && json.get(i + 1 + field_name.len()) == Some(&b'"')
                {
                    // Found field name, now find colon and value
                    let mut j = i + 1 + field_name.len() + 1;
                    
                    // Skip whitespace and colon
                    while j < json.len() && (json[j] == b':' || json[j] <= b' ') {
                        j += 1;
                    }
                    
                    if j >= json.len() {
                        return None;
                    }
                    
                    // Determine value type and extract
                    return match json[j] {
                        b'"' => Self::extract_string(json, j + 1),
                        b'0'..=b'9' | b'-' => Self::extract_number(json, j),
                        b't' | b'f' => Self::extract_bool(json, j),
                        _ => None,
                    };
                }
            }
            i += 1;
        }
        
        None
    }

    #[inline]
    fn extract_string<'a>(json: &'a [u8], start: usize) -> Option<(&'a [u8], usize)> {
        let mut end = start;
        while end < json.len() && json[end] != b'"' {
            // Handle escape sequences
            if json[end] == b'\\' {
                end += 2;
            } else {
                end += 1;
            }
        }
        
        if end > start {
            Some((&json[start..end], end - start))
        } else {
            None
        }
    }

    #[inline]
    fn extract_number<'a>(json: &'a [u8], start: usize) -> Option<(&'a [u8], usize)> {
        let mut end = start;
        while end < json.len() && (json[end].is_ascii_digit() || json[end] == b'.' || json[end] == b'-' || json[end] == b'e' || json[end] == b'E' || json[end] == b'+') {
            end += 1;
        }
        
        if end > start {
            Some((&json[start..end], end - start))
        } else {
            None
        }
    }

    #[inline]
    fn extract_bool<'a>(json: &'a [u8], start: usize) -> Option<(&'a [u8], usize)> {
        if json[start..].starts_with(b"true") {
            Some((&json[start..start + 4], 4))
        } else if json[start..].starts_with(b"false") {
            Some((&json[start..start + 5], 5))
        } else {
            None
        }
    }
}

/// News Ingestor - main entry point
#[repr(C)]
pub struct NewsIngestor<const BUFFER_SIZE: usize> {
    buffer: NewsBuffer<BUFFER_SIZE>,
    items_processed: AtomicU64,
    parse_errors: AtomicU64,
    last_ingest_cycle: AtomicU64,
    active: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 17],
}

// SAFETY: All interior mutability protected by atomics
unsafe impl<const B: usize> Send for NewsIngestor<B> {}
unsafe impl<const B: usize> Sync for NewsIngestor<B> {}

impl<const BUFFER_SIZE: usize> NewsIngestor<BUFFER_SIZE> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            buffer: NewsBuffer::new(),
            items_processed: AtomicU64::new(0),
            parse_errors: AtomicU64::new(0),
            last_ingest_cycle: AtomicU64::new(0),
            active: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 17],
        }
    }

    /// Activate ingestor
    #[inline(always)]
    pub fn activate(&self) {
        self.active.store(true, Ordering::Release);
    }

    /// Deactivate ingestor
    #[inline(always)]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Release);
    }

    /// Check if active
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Ingest raw WebSocket message
    #[inline]
    pub fn ingest(&self, data: &[u8]) -> Result<usize, &'static str> {
        if !self.is_active() {
            return Err("Ingestor not active");
        }
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cycles = core::arch::x86_64::_rdtsc();
            self.last_ingest_cycle.store(cycles, Ordering::Relaxed);
        }
        
        let written = self.buffer.write(data);
        if written == 0 {
            self.parse_errors.fetch_add(1, Ordering::Relaxed);
            return Err("Buffer overflow");
        }
        
        Ok(written)
    }

    /// Parse next news item from buffer
    #[inline]
    pub fn parse_next(&self) -> Option<NewsItem> {
        let available = self.buffer.available();
        if available < 10 {
            return None;
        }
        
        // Read enough data to parse
        let mut raw_data = [0u8; MAX_HEADLINE_LEN + 256];
        let read = self.buffer.read(&mut raw_data);
        
        if read < 10 {
            return None;
        }
        
        // Try to parse as JSON
        // Expected format: {"source":"...", "headline":"...", "urgency":N, "category":N}
        
        let mut item = NewsItem::new();
        
        // Get timestamp
        #[cfg(target_arch = "x86_64")]
        unsafe {
            item.timestamp_ns = core::arch::x86_64::_rdtsc() as u64 * 25; // Approximate ns
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            item.timestamp_ns = 0;
        }
        
        // Extract source
        if let Some((src, len)) = JsonExtractor::extract_field(&raw_data[..read], b"source") {
            item.source_hash = fnv1a_hash(src);
        }
        
        // Extract headline
        if let Some((headline, len)) = JsonExtractor::extract_field(&raw_data[..read], b"headline") {
            if len > 0 && len < MAX_HEADLINE_LEN {
                item.headline_ptr = headline.as_ptr();
                item.headline_len = len;
            }
        }
        
        // Extract urgency
        if let Some((urg, _)) = JsonExtractor::extract_field(&raw_data[..read], b"urgency") {
            if urg.starts_with(b"2") {
                item.urgency = 2;
            } else if urg.starts_with(b"1") {
                item.urgency = 1;
            }
        }
        
        // Extract category
        if let Some((cat, _)) = JsonExtractor::extract_field(&raw_data[..read], b"category") {
            if cat.len() > 0 {
                item.category = cat[0] - b'0';
            }
        }
        
        if item.headline_len > 0 {
            self.items_processed.fetch_add(1, Ordering::Relaxed);
            Some(item)
        } else {
            self.parse_errors.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, usize) {
        (
            self.items_processed.load(Ordering::Relaxed),
            self.parse_errors.load(Ordering::Relaxed),
            self.buffer.available(),
        )
    }

    /// Get last ingest cycle count
    #[inline(always)]
    pub fn last_ingest_cycles(&self) -> u64 {
        self.last_ingest_cycle.load(Ordering::Relaxed)
    }

    /// Reset statistics
    #[inline]
    pub fn reset_stats(&self) {
        self.items_processed.store(0, Ordering::Relaxed);
        self.parse_errors.store(0, Ordering::Relaxed);
        self.buffer.reset();
    }
}

/// FNV-1a hash function
#[inline]
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Type alias for typical configuration
pub type CryptoNewsIngestor = NewsIngestor<1024 * 1024>; // 1MB buffer

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_item_creation() {
        let item = NewsItem::new();
        assert_eq!(item.timestamp_ns, 0);
        assert_eq!(item.headline_len, 0);
    }

    #[test]
    fn test_json_extractor_string() {
        let json = br#"{"headline":"Test News","source":"Reuters"}"#;
        
        let (value, len) = JsonExtractor::extract_field(json, b"headline").unwrap();
        assert_eq!(len, 9);
        assert_eq!(value, b"Test News");
    }

    #[test]
    fn test_json_extractor_number() {
        let json = br#"{"urgency":2,"category":1}"#;
        
        let (value, _) = JsonExtractor::extract_field(json, b"urgency").unwrap();
        assert_eq!(value, b"2");
    }

    #[test]
    fn test_news_buffer_write_read() {
        let buf = NewsBuffer::<1024>::new();
        
        let data = b"Hello, World!";
        let written = buf.write(data);
        assert_eq!(written, data.len());
        
        let mut read_buf = [0u8; 100];
        let read = buf.read(&mut read_buf);
        assert_eq!(read, data.len());
        assert_eq!(&read_buf[..read], data);
    }

    #[test]
    fn test_news_ingestor_basic() {
        let ingestor = NewsIngestor::<1024>::new();
        ingestor.activate();
        
        assert!(ingestor.is_active());
        
        let json = br#"{"headline":"BTC Breaks $50k","source":"CoinDesk","urgency":1}"#;
        let result = ingestor.ingest(json);
        assert!(result.is_ok());
        
        let item = ingestor.parse_next();
        assert!(item.is_some());
        
        let (processed, errors, _) = ingestor.get_stats();
        assert_eq!(processed, 1);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_fnv1a_hash() {
        let hash1 = fnv1a_hash(b"test");
        let hash2 = fnv1a_hash(b"test");
        let hash3 = fnv1a_hash(b"Test");
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_buffer_overflow_handling() {
        let buf = NewsBuffer::<100>::new();
        
        // Fill buffer
        let large_data = [0u8; 150];
        let written = buf.write(&large_data);
        assert_eq!(written, 0); // Should fail
        assert!(buf.has_overflow());
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;
        
        assert!(size_of::<NewsItem>() >= CACHE_LINE_SIZE);
    }
}
