//! Lock-free ring buffer for market data events.
//! 
//! Single-producer single-consumer (SPSC) ring buffer with zero allocations.

#![allow(dead_code)]

use core::sync::atomic::{AtomicUsize, Ordering};

/// Ring buffer capacity (must be power of 2)
const CAPACITY: usize = 4096;
const MASK: usize = CAPACITY - 1;

/// Result type for ring buffer operations
pub type RingResult<T> = Result<(), T>;

/// SPSC Ring Buffer
pub struct RingBuffer<T: Copy> {
    buffer: [T; CAPACITY],
    head: AtomicUsize, // Write position
    tail: AtomicUsize, // Read position
}

unsafe impl<T: Copy + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for RingBuffer<T> {}

impl<T: Copy> RingBuffer<T> {
    pub const fn new() -> Self {
        use core::mem::MaybeUninit;
        // SAFETY: We assume T can be zero-initialized
        Self {
            buffer: unsafe { MaybeUninit::zeroed().assume_init() },
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    /// Push an item to the buffer
    #[inline]
    pub fn push(&self, item: T) -> RingResult<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head.wrapping_sub(tail) >= CAPACITY {
            return Err(item); // Buffer full
        }
        
        unsafe {
            *self.buffer.get_unchecked_mut(head & MASK) = item;
        }
        
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
    
    /// Pop an item from the buffer
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Buffer empty
        }
        
        let item = unsafe { *self.buffer.get_unchecked(tail & MASK) };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }
    
    /// Get current size
    #[inline]
    pub fn size(&self) -> usize {
        self.head.load(Ordering::Relaxed).wrapping_sub(self.tail.load(Ordering::Relaxed))
    }
}

impl<T: Copy> Default for RingBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}
