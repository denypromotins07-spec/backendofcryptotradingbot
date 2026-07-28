// src/transport/ring_buffer.rs
//! Wait-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
//!
//! This module implements a lock-free, wait-free ring buffer optimized for:
//! - Single producer, single consumer pattern
//! - Zero contention in the common case
//! - Sub-microsecond latency for push/pop operations
//! - Cache-line alignment to prevent false sharing
//!
//! Micro-optimizations:
//! - Atomic sequence counters with relaxed ordering where safe
//! - Power-of-2 size for fast modulo via bitwise AND
//! - Pre-allocated slots (no dynamic allocation)
//! - Separate cache lines for producer and consumer state

#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

/// Cache line size for padding (x86_64)
const CACHE_LINE_SIZE: usize = 64;

/// Default ring buffer capacity (must be power of 2)
const DEFAULT_CAPACITY: usize = 1 << 14; // 16384 slots

/// Ring buffer slot containing the actual data
/// 
/// Each slot is padded to cache line boundary to prevent
/// false sharing between adjacent slots during concurrent access
#[repr(C)]
struct Slot<T> {
    /// The actual data stored in this slot
    data: UnsafeCell<Option<T>>,
    /// Padding to ensure next slot starts on new cache line
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Option<T>>>()],
}

impl<T> Slot<T> {
    const fn new() -> Self {
        Self {
            data: UnsafeCell::new(None),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Option<T>>>()],
        }
    }
}

/// Wait-free SPSC Ring Buffer
///
/// Memory layout:
/// ```text
/// +------------------+     +------------------+
/// | Producer State   |     | Consumer State   |
/// | (cache line 0)   |     | (cache line N)   |
/// +------------------+     +------------------+
/// | Slot 0           |
/// | Slot 1           |
/// | ...              |
/// | Slot N-1         |
/// +------------------+
/// ```
///
/// The producer and consumer states are on separate cache lines
/// to prevent false sharing when both threads are active.
#[repr(C)]
pub struct SpscRingBuffer<T> {
    /// Producer's next write position
    /// Placed on its own cache line
    producer_pos: AtomicU64,
    _pad0: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
    
    /// Consumer's next read position
    /// Placed on its own cache line to avoid false sharing
    consumer_pos: AtomicU64,
    _pad1: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
    
    /// The actual ring buffer slots
    /// Capacity is always a power of 2 for efficient masking
    slots: Box<[Slot<T>; DEFAULT_CAPACITY]>,
    
    /// Mask for fast index calculation (capacity - 1)
    mask: usize,
    
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> SpscRingBuffer<T> {
    /// Create a new SPSC ring buffer
    ///
    /// # Panics
    /// Panics if memory allocation fails
    pub fn new() -> Self {
        // Initialize all slots with None
        let slots: Box<[Slot<T>; DEFAULT_CAPACITY]> = {
            let mut slots = Vec::with_capacity(DEFAULT_CAPACITY);
            for _ in 0..DEFAULT_CAPACITY {
                slots.push(Slot::new());
            }
            slots.into_boxed_slice().try_into().unwrap()
        };
        
        Self {
            producer_pos: AtomicU64::new(0),
            _pad0: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
            consumer_pos: AtomicU64::new(0),
            _pad1: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>()],
            slots,
            mask: DEFAULT_CAPACITY - 1,
            _phantom: PhantomData,
        }
    }
    
    /// Push an item into the ring buffer (producer side)
    ///
    /// This is a wait-free operation - it never blocks or spins.
    /// Returns `Some(item)` if the buffer was full (item not pushed),
    /// or `None` if successful.
    ///
    /// # Complexity
    /// O(1) time, no allocations
    #[inline(always)]
    pub fn push(&self, item: T) -> Option<T> {
        let prod_pos = self.producer_pos.load(Ordering::Relaxed);
        let cons_pos = self.consumer_pos.load(Ordering::Acquire);
        
        // Check if buffer is full
        // We need at least one empty slot to distinguish full from empty
        if prod_pos.wrapping_sub(cons_pos) >= DEFAULT_CAPACITY as u64 {
            return Some(item); // Buffer full
        }
        
        // Calculate slot index using bitmask (faster than modulo)
        let idx = (prod_pos as usize) & self.mask;
        let slot = &self.slots[idx];
        
        // Write the data
        unsafe {
            *slot.data.get() = Some(item);
        }
        
        // Update producer position with release semantics
        // This ensures the data write is visible before the position update
        self.producer_pos.store(prod_pos + 1, Ordering::Release);
        
        None // Success
    }
    
    /// Pop an item from the ring buffer (consumer side)
    ///
    /// This is a wait-free operation - it never blocks or spins.
    /// Returns `Some(item)` if successful, or `None` if buffer was empty.
    ///
    /// # Complexity
    /// O(1) time, no allocations
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let cons_pos = self.consumer_pos.load(Ordering::Relaxed);
        let prod_pos = self.producer_pos.load(Ordering::Acquire);
        
        // Check if buffer is empty
        if cons_pos >= prod_pos {
            return None; // Buffer empty
        }
        
        // Calculate slot index
        let idx = (cons_pos as usize) & self.mask;
        let slot = &self.slots[idx];
        
        // Read the data
        let item = unsafe {
            (*slot.data.get()).take()
        };
        
        // Update consumer position with release semantics
        self.consumer_pos.store(cons_pos + 1, Ordering::Release);
        
        item
    }
    
    /// Check if the buffer is empty (consumer perspective)
    ///
    /// Note: This is a snapshot - the state may change immediately after
    #[inline]
    pub fn is_empty(&self) -> bool {
        let cons_pos = self.consumer_pos.load(Ordering::Acquire);
        let prod_pos = self.producer_pos.load(Ordering::Acquire);
        cons_pos >= prod_pos
    }
    
    /// Check if the buffer is full (producer perspective)
    ///
    /// Note: This is a snapshot - the state may change immediately after
    #[inline]
    pub fn is_full(&self) -> bool {
        let prod_pos = self.producer_pos.load(Ordering::Acquire);
        let cons_pos = self.consumer_pos.load(Ordering::Acquire);
        prod_pos.wrapping_sub(cons_pos) >= DEFAULT_CAPACITY as u64
    }
    
    /// Get the current number of items in the buffer
    ///
    /// Note: This is a snapshot - the actual count may change
    #[inline]
    pub fn len(&self) -> usize {
        let prod_pos = self.producer_pos.load(Ordering::Acquire);
        let cons_pos = self.consumer_pos.load(Ordering::Acquire);
        (prod_pos.wrapping_sub(cons_pos)) as usize
    }
    
    /// Get the buffer capacity
    #[inline]
    pub const fn capacity(&self) -> usize {
        DEFAULT_CAPACITY
    }
}

impl<T> Default for SpscRingBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time assertion for capacity being power of 2
const_assert!(DEFAULT_CAPACITY.is_power_of_two());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_single() {
        let buffer = SpscRingBuffer::<i32>::new();
        assert!(buffer.is_empty());
        
        assert!(buffer.push(42).is_none());
        assert!(!buffer.is_empty());
        
        let value = buffer.pop();
        assert_eq!(value, Some(42));
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_push_pop_multiple() {
        let buffer = SpscRingBuffer::<i32>::new();
        
        for i in 0..100 {
            assert!(buffer.push(i).is_none());
        }
        
        for i in 0..100 {
            assert_eq!(buffer.pop(), Some(i));
        }
        
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_buffer_full() {
        let buffer = SpscRingBuffer::<i32>::new();
        
        // Fill the buffer
        for i in 0..DEFAULT_CAPACITY {
            let result = buffer.push(i as i32);
            assert!(result.is_none(), "Should accept item {}", i);
        }
        
        // Next push should fail (buffer full)
        let result = buffer.push(-1);
        assert_eq!(result, Some(-1), "Should reject item when full");
        
        // Pop one item
        let value = buffer.pop();
        assert!(value.is_some());
        
        // Now we can push again
        assert!(buffer.push(999).is_none());
    }
    
    #[test]
    fn test_wraparound() {
        let buffer = SpscRingBuffer::<i32>::new();
        
        // Push and pop to advance positions
        for _ in 0..DEFAULT_CAPACITY + 10 {
            assert!(buffer.push(1).is_none());
            assert_eq!(buffer.pop(), Some(1));
        }
        
        // Buffer should still work correctly after wraparound
        assert!(buffer.is_empty());
        assert!(buffer.push(42).is_none());
        assert_eq!(buffer.pop(), Some(42));
    }
    
    #[test]
    fn test_len() {
        let buffer = SpscRingBuffer::<i32>::new();
        assert_eq!(buffer.len(), 0);
        
        for i in 0..50 {
            buffer.push(i);
            assert_eq!(buffer.len(), i + 1);
        }
        
        for i in 0..50 {
            buffer.pop();
            assert_eq!(buffer.len(), 49 - i);
        }
    }
}
