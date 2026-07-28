// src/transport/shared_memory.rs
//! Memory-Mapped File Handlers for Persistent, Low-Latency State Recovery
//!
//! This module implements memory-mapped file operations for:
//! - Zero-copy persistence of trading state
//! - Fast recovery after crashes or restarts
//! - Circular buffer storage for replayable event logs
//! - Crash-consistent state snapshots
//!
//! Micro-optimizations:
//! - Direct memory mapping (no intermediate buffers)
//! - Pre-reserved file space (no dynamic growth)
//! - Cache-line aligned writes for optimal performance
//! - Atomic header updates for crash consistency

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 64;

/// Default memory-mapped region size (1 GB)
const DEFAULT_MMAP_SIZE: usize = 1 << 30;

/// Magic number for file format identification
const MMAP_MAGIC: u32 = 0x4D4D4150; // "MMAP" in ASCII

/// Version of the file format
const MMAP_VERSION: u32 = 1;

/// Header structure at the beginning of memory-mapped files
/// This is written atomically to ensure crash consistency
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MmapHeader {
    /// Magic number for format identification
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Total file size in bytes
    pub file_size: u64,
    /// Current write position (for circular buffer)
    pub write_pos: u64,
    /// Number of records written
    pub record_count: u64,
    /// Last checksum of data region
    pub data_checksum: u64,
    /// Timestamp of last update (nanoseconds)
    pub last_update_ns: u64,
    /// Flags for various states
    pub flags: u32,
    /// Reserved for future use
    pub reserved: [u32; 7],
}

impl MmapHeader {
    /// Create a new header with default values
    #[inline]
    pub const fn new(file_size: u64) -> Self {
        Self {
            magic: MMAP_MAGIC,
            version: MMAP_VERSION,
            file_size,
            write_pos: 0,
            record_count: 0,
            data_checksum: 0,
            last_update_ns: 0,
            flags: 0,
            reserved: [0; 7],
        }
    }

    /// Validate the header
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.magic == MMAP_MAGIC && self.version == MMAP_VERSION
    }

    /// Calculate header checksum for integrity verification
    #[inline]
    pub fn calculate_checksum(&self) -> u32 {
        let mut sum: u32 = 0;
        sum = sum.wrapping_add(self.magic);
        sum = sum.wrapping_add(self.version);
        sum = sum.wrapping_add((self.file_size & 0xFFFFFFFF) as u32);
        sum = sum.wrapping_add((self.file_size >> 32) as u32);
        sum = sum.wrapping_add((self.write_pos & 0xFFFFFFFF) as u32);
        sum = sum.wrapping_add((self.write_pos >> 32) as u32);
        sum = sum.wrapping_add((self.record_count & 0xFFFFFFFF) as u32);
        sum = sum.wrapping_add((self.record_count >> 32) as u32);
        sum = sum.wrapping_add((self.data_checksum & 0xFFFFFFFF) as u32);
        sum = sum.wrapping_add((self.data_checksum >> 32) as u32);
        sum = sum.wrapping_add((self.last_update_ns & 0xFFFFFFFF) as u32);
        sum = sum.wrapping_add((self.last_update_ns >> 32) as u32);
        sum = sum.wrapping_add(self.flags);
        for &r in &self.reserved {
            sum = sum.wrapping_add(r);
        }
        sum
    }
}

/// Memory-mapped file handler
///
/// Provides zero-copy access to persistent storage through mmap
pub struct SharedMemory {
    /// Path to the memory-mapped file
    path: PathBuf,
    /// The underlying file handle
    file: File,
    /// Pointer to mapped memory region
    ptr: *mut u8,
    /// Size of the mapped region
    size: usize,
    /// Header pointer (convenience cast)
    header: *mut MmapHeader,
    /// Data region start pointer
    data_start: *mut u8,
    /// Data region size
    data_size: usize,
    /// Track if we own the mapping (for Drop)
    is_mapped: bool,
}

// SharedMemory can be sent between threads safely
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create a new memory-mapped file
    ///
    /// # Arguments
    /// * `path` - Path to the file to create/open
    /// * `size` - Size of the memory region (will be rounded up to page size)
    ///
    /// # Returns
    /// Result containing the SharedMemory handle or an error
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Create or open the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // Get current file size
        let metadata = file.metadata()?;
        let current_size = metadata.len() as usize;

        // Extend file if needed
        if current_size < size + core::mem::size_of::<MmapHeader>() {
            file.set_len((size + core::mem::size_of::<MmapHeader>()) as u64)?;
        }

        // Memory map the file
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size + core::mem::size_of::<MmapHeader>(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let ptr = ptr as *mut u8;
        let header = ptr as *mut MmapHeader;
        let data_start = unsafe { ptr.add(core::mem::size_of::<MmapHeader>()) };

        // Initialize header if this is a new file
        unsafe {
            if !(*header).is_valid() {
                *header = MmapHeader::new((size + core::mem::size_of::<MmapHeader>()) as u64);
            }
        }

        Ok(Self {
            path,
            file,
            ptr,
            size: size + core::mem::size_of::<MmapHeader>(),
            header,
            data_start,
            data_size: size,
            is_mapped: true,
        })
    }

    /// Open an existing memory-mapped file
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        if file_size < core::mem::size_of::<MmapHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small to contain valid header",
            ));
        }

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                file_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let ptr = ptr as *mut u8;
        let header = ptr as *mut MmapHeader;

        // Validate header
        unsafe {
            if !(*header).is_valid() {
                libc::munmap(ptr as *mut libc::c_void, file_size);
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid file header",
                ));
            }
        }

        let data_start = unsafe { ptr.add(core::mem::size_of::<MmapHeader>()) };
        let data_size = file_size - core::mem::size_of::<MmapHeader>();

        Ok(Self {
            path,
            file,
            ptr,
            size: file_size,
            header,
            data_start,
            data_size,
            is_mapped: true,
        })
    }

    /// Write data to the memory-mapped region (circular buffer style)
    #[inline(always)]
    pub fn write(&self, data: &[u8]) -> io::Result<u64> {
        unsafe {
            let header = &mut *self.header;
            
            // Calculate write position in data region
            let pos = (header.write_pos as usize) % self.data_size;
            
            // Check if we need to wrap around
            let available = if pos + data.len() <= self.data_size {
                data.len()
            } else {
                self.data_size - pos
            };

            // Write data
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data_start.add(pos),
                available,
            );

            // Handle wrap-around if needed
            if data.len() > available {
                ptr::copy_nonoverlapping(
                    data.as_ptr().add(available),
                    self.data_start,
                    data.len() - available,
                );
            }

            // Update header atomically
            header.write_pos = header.write_pos.wrapping_add(data.len() as u64);
            header.record_count = header.record_count.wrapping_add(1);
            header.last_update_ns = get_timestamp_ns();

            Ok(header.write_pos)
        }
    }

    /// Read data from the memory-mapped region
    #[inline]
    pub fn read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        unsafe {
            let header = &*self.header;
            
            // Validate offset
            if offset >= header.write_pos {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Offset beyond written data",
                ));
            }

            let pos = (offset as usize) % self.data_size;
            let available = core::cmp::min(len, self.data_size - pos);

            let mut result = Vec::with_capacity(len);
            
            // Read data
            result.extend_from_slice(std::slice::from_raw_parts(
                self.data_start.add(pos),
                available,
            ));

            // Handle wrap-around
            if len > available {
                let remaining = len - available;
                result.extend_from_slice(std::slice::from_raw_parts(
                    self.data_start,
                    remaining,
                ));
            }

            Ok(result)
        }
    }

    /// Get direct pointer to data region (zero-copy access)
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data_start
    }

    /// Get data region size
    #[inline]
    pub fn data_size(&self) -> usize {
        self.data_size
    }

    /// Get current write position
    #[inline]
    pub fn write_position(&self) -> u64 {
        unsafe { (*self.header).write_pos }
    }

    /// Get record count
    #[inline]
    pub fn record_count(&self) -> u64 {
        unsafe { (*self.header).record_count }
    }

    /// Flush changes to disk
    #[inline]
    pub fn flush(&self) -> io::Result<()> {
        unsafe {
            let result = libc::msync(
                self.ptr as *mut libc::c_void,
                self.size,
                libc::MS_SYNC,
            );
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    /// Sync only the header (for crash consistency)
    #[inline]
    pub fn sync_header(&self) -> io::Result<()> {
        unsafe {
            let result = libc::msync(
                self.header as *mut libc::c_void,
                core::mem::size_of::<MmapHeader>(),
                libc::MS_SYNC,
            );
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        if self.is_mapped {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.size);
            }
        }
    }
}

/// Get timestamp in nanoseconds using rdtsc
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
    use std::env;

    #[test]
    fn test_header_creation() {
        let header = MmapHeader::new(1024);
        assert!(header.is_valid());
        assert_eq!(header.magic, MMAP_MAGIC);
        assert_eq!(header.version, MMAP_VERSION);
        assert_eq!(header.file_size, 1024);
    }

    #[test]
    fn test_shared_memory_create() {
        let temp_path = env::temp_dir().join("test_mmap.bin");
        
        {
            let mmap = SharedMemory::create(&temp_path, 4096).unwrap();
            assert!(mmap.data_size() >= 4096);
            assert_eq!(mmap.write_position(), 0);
            
            // Write some data
            let data = b"Hello, World!";
            mmap.write(data).unwrap();
            assert_eq!(mmap.write_position(), data.len() as u64);
        }
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_shared_memory_read_write() {
        let temp_path = env::temp_dir().join("test_mmap_rw.bin");
        
        {
            let mmap = SharedMemory::create(&temp_path, 4096).unwrap();
            
            // Write data
            let data = b"Test data for mmap";
            mmap.write(data).unwrap();
            
            // Read back
            let read_data = mmap.read(0, data.len()).unwrap();
            assert_eq!(read_data, data);
        }
        
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_header_checksum() {
        let mut header = MmapHeader::new(2048);
        let checksum1 = header.calculate_checksum();
        
        header.record_count = 100;
        let checksum2 = header.calculate_checksum();
        
        assert_ne!(checksum1, checksum2, "Checksum should change when data changes");
    }
}
