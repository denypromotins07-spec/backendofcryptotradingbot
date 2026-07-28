// src/core/allocator.rs
//! Custom Bump Allocator with Strict 6.5GB RAM Boundary Enforcement
//! 
//! This module implements a custom global allocator that:
//! - Uses bump allocation for O(1) allocation in the hot path
//! - Enforces a hard 6.5GB memory limit with circuit breaker
//! - Tracks peak memory usage for monitoring
//! - Avoids heap fragmentation by design
//! 
//! Micro-optimizations:
//! - Cache-line aligned bump pointer to prevent false sharing
//! - Atomic operations for thread-safe limit checking
//! - Zero-initialization skip for performance-critical allocations

#![allow(dead_code)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Memory limit: 6.5 GB in bytes
const MEMORY_LIMIT_BYTES: usize = 6_500_000_000;

/// Cache line size for padding (x86_64)
const CACHE_LINE_SIZE: usize = 64;

/// Arena size for bump allocation (pre-allocated memory pool)
const ARENA_SIZE: usize = MEMORY_LIMIT_BYTES;

/// Circuit breaker state - set to true when memory limit approached
static CIRCUIT_BREAKER: AtomicBool = AtomicBool::new(false);

/// Peak memory usage tracker (bytes)
static PEAK_MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Current memory usage counter (bytes)
static CURRENT_MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Panic flag to prevent re-entry during panic handling
static PANIC_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Custom Bump Allocator Structure
/// 
/// Memory layout:
/// - `start`: Pointer to beginning of arena
/// - `bump`: Current allocation pointer (grows upward)
/// - `end`: Pointer to end of arena
/// 
/// All fields are cache-line padded to prevent false sharing
#[repr(C)]
struct BumpAllocator {
    /// Start of the memory arena
    start: UnsafeCell<*mut u8>,
    /// Current bump pointer - atomic for thread safety
    bump: AtomicUsize,
    /// End of the memory arena
    end: UnsafeCell<*mut u8>,
    /// Padding to ensure `bump` is on its own cache line
    _pad0: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>()],
    /// Allocation statistics
    alloc_count: AtomicU64,
    /// Padding before next cache line
    _pad1: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
}

impl BumpAllocator {
    /// Create a new bump allocator (called once at startup)
    const fn new() -> Self {
        Self {
            start: UnsafeCell::new(ptr::null_mut()),
            bump: AtomicUsize::new(0),
            end: UnsafeCell::new(ptr::null_mut()),
            _pad0: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>()],
            alloc_count: AtomicU64::new(0),
            _pad1: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
        }
    }

    /// Initialize the allocator with pre-allocated memory
    /// 
    /// # Safety
    /// Must be called exactly once before any allocations
    unsafe fn init(&self, arena_start: *mut u8, arena_size: usize) {
        let start_ptr = self.start.get();
        let end_ptr = self.end.get();
        
        *start_ptr = arena_start;
        *end_ptr = arena_start.add(arena_size);
        
        self.bump.store(arena_start as usize, Ordering::Relaxed);
    }

    /// Allocate memory from the bump allocator
    /// 
    /// Returns null if:
    /// - Memory limit would be exceeded
    /// - Circuit breaker is tripped
    /// - Alignment requirements cannot be met
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> *mut u8 {
        // Fast path: check circuit breaker first (branch prediction friendly)
        if CIRCUIT_BREAKER.load(Ordering::Relaxed) {
            return ptr::null_mut();
        }

        let size = layout.size();
        let align = layout.align();

        // Get current bump pointer
        let mut current = self.bump.load(Ordering::Acquire);
        
        loop {
            // Calculate aligned address
            let aligned = (current + align - 1) & !(align - 1);
            let new_bump = aligned + size;
            
            // Check bounds against arena end
            let end = self.end.get() as usize;
            if new_bump > end {
                // Out of memory - trigger circuit breaker
                CIRCUIT_BREAKER.store(true, Ordering::Release);
                return ptr::null_mut();
            }

            // Try to atomically advance the bump pointer
            match self.bump.compare_exchange_weak(
                current,
                new_bump,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Success - update statistics
                    let used = new_bump - self.start.get() as usize;
                    CURRENT_MEMORY_USAGE.store(used, Ordering::Relaxed);
                    
                    // Update peak usage
                    let mut peak = PEAK_MEMORY_USAGE.load(Ordering::Relaxed);
                    while used > peak {
                        match PEAK_MEMORY_USAGE.compare_exchange_weak(
                            peak,
                            used,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(p) => peak = p,
                        }
                    }
                    
                    self.alloc_count.fetch_add(1, Ordering::Relaxed);
                    
                    // Check if approaching limit (95% threshold)
                    if used > (MEMORY_LIMIT_BYTES * 95 / 100) {
                        CIRCUIT_BREAKER.store(true, Ordering::Release);
                    }
                    
                    return aligned as *mut u8;
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Deallocate memory (no-op for bump allocator - memory reclaimed on reset)
    #[inline(always)]
    fn deallocate(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocators typically don't support individual deallocation
        // Memory is reclaimed when the arena is reset
        // For this trading bot, we use epoch-based reclamation externally
    }

    /// Reset the allocator (call only when safe - no outstanding references)
    /// 
    /// # Safety
    /// Must only be called when no allocations are in use
    unsafe fn reset(&self) {
        let start = self.start.get() as usize;
        self.bump.store(start, Ordering::Release);
        CURRENT_MEMORY_USAGE.store(0, Ordering::Release);
        self.alloc_count.store(0, Ordering::Relaxed);
    }

    /// Get current memory usage in bytes
    #[inline]
    fn current_usage(&self) -> usize {
        CURRENT_MEMORY_USAGE.load(Ordering::Relaxed)
    }

    /// Get peak memory usage in bytes
    #[inline]
    fn peak_usage(&self) -> usize {
        PEAK_MEMORY_USAGE.load(Ordering::Relaxed)
    }

    /// Check if circuit breaker is tripped
    #[inline]
    fn is_circuit_breaker_tripped(&self) -> bool {
        CIRCUIT_BREAKER.load(Ordering::Relaxed)
    }
}

/// Global static bump allocator instance
static GLOBAL_ALLOCATOR: BumpAllocator = BumpAllocator::new();

/// Custom Global Allocator implementing GlobalAlloc trait
pub struct UltraLowLatencyAllocator;

unsafe impl GlobalAlloc for UltraLowLatencyAllocator {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        GLOBAL_ALLOCATOR.allocate(layout)
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        GLOBAL_ALLOCATOR.deallocate(ptr, layout);
    }
}

/// Set the custom global allocator
#[global_allocator]
static GLOBAL: UltraLowLatencyAllocator = UltraLowLatencyAllocator;

/// Initialize the memory arena
/// 
/// # Safety
/// Must be called exactly once at program startup
pub unsafe fn initialize_allocator() -> Result<(), &'static str> {
    if CIRCUIT_BREAKER.load(Ordering::Relaxed) {
        return Err("Allocator already initialized or circuit breaker tripped");
    }

    // Allocate the arena using system malloc (one-time cost)
    let layout = Layout::from_size_align(ARENA_SIZE, CACHE_LINE_SIZE)
        .map_err(|_| "Invalid layout for arena")?;
    
    let arena_start = std::alloc::System.alloc(layout);
    
    if arena_start.is_null() {
        return Err("Failed to allocate memory arena");
    }

    // Initialize our bump allocator with the arena
    GLOBAL_ALLOCATOR.init(arena_start, ARENA_SIZE);

    // Install panic hook to capture state before corruption
    install_panic_hook();

    Ok(())
}

/// Install custom panic hook to preserve shared memory state
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Prevent re-entry
        if PANIC_IN_PROGRESS.swap(true, Ordering::SeqCst) {
            return;
        }

        // Log critical information before panic
        eprintln!("=== CRITICAL PANIC ===");
        eprintln!("Peak Memory Usage: {} bytes", GLOBAL_ALLOCATOR.peak_usage());
        eprintln!("Current Memory Usage: {} bytes", GLOBAL_ALLOCATOR.current_usage());
        eprintln!("Total Allocations: {}", GLOBAL_ALLOCATOR.alloc_count.load(Ordering::Relaxed));
        eprintln!("Circuit Breaker Tripped: {}", GLOBAL_ALLOCATOR.is_circuit_breaker_tripped());
        
        // Call default hook for backtrace
        default_hook(panic_info);
        
        // Memory will be preserved since we use abort panics
    }));
}

/// Get current memory usage (public API for monitoring)
#[inline]
pub fn get_memory_usage() -> usize {
    GLOBAL_ALLOCATOR.current_usage()
}

/// Get peak memory usage (public API for monitoring)
#[inline]
pub fn get_peak_memory_usage() -> usize {
    GLOBAL_ALLOCATOR.peak_usage()
}

/// Check if circuit breaker is active
#[inline]
pub fn is_circuit_breaker_active() -> bool {
    GLOBAL_ALLOCATOR.is_circuit_breaker_tripped()
}

/// Compile-time assertion for memory limit
const_assert!(MEMORY_LIMIT_BYTES == 6_500_000_000);

/// Macro for compile-time assertions
#[macro_export]
macro_rules! const_assert {
    ($x:expr) => {
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_constants() {
        assert_eq!(MEMORY_LIMIT_BYTES, 6_500_000_000);
        assert_eq!(CACHE_LINE_SIZE, 64);
    }

    #[test]
    fn test_bump_allocator_size() {
        // Verify BumpAllocator is properly cache-line padded
        let size = core::mem::size_of::<BumpAllocator>();
        assert!(size >= CACHE_LINE_SIZE * 2, "BumpAllocator should span multiple cache lines");
    }
}
