// src/core/event_loop.rs
//! Lock-Free LMAX Disruptor-Style Event Bus for Microsecond Message Routing
//!
//! This module implements a high-performance event bus inspired by the LMAX Disruptor pattern:
//! - Lock-free ring buffer for event storage
//! - Single producer, multiple consumer support
//! - Batch processing for instruction cache optimization
//! - Cache-line padding to prevent false sharing
//!
//! Micro-optimizations:
//! - Sequence barriers using atomic operations
//! - Pre-allocated event slots (no dynamic allocation in hot path)
//! - Branch prediction hints via ordering semantics
//! - Vectorized batch processing where applicable

#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Maximum number of events in the ring buffer (must be power of 2)
const RING_BUFFER_SIZE: usize = 1 << 16; // 65536 events

/// Cache line size for padding
const CACHE_LINE_SIZE: usize = 64;

/// Event types supported by the event bus
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    MarketData = 0,
    OrderSubmission = 1,
    OrderCancellation = 2,
    OrderFill = 3,
    Heartbeat = 4,
    SystemSignal = 5,
}

/// Event structure - cache-line aligned for optimal memory access
/// 
/// Layout designed to fit within single cache line where possible
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Event {
    /// Event type discriminator
    pub event_type: EventType,
    /// Sequence number for ordering guarantees
    pub sequence: u64,
    /// Timestamp captured via rdtsc (nanosecond precision)
    pub timestamp_ns: u64,
    /// Payload size (for variable-length data)
    pub payload_size: u32,
    /// Symbol ID for routing
    pub symbol_id: u32,
    /// Inline payload (fits common small messages)
    pub payload: [u8; 32],
}

impl Event {
    /// Create a new event with rdtsc timestamp
    #[inline(always)]
    pub fn new(event_type: EventType, sequence: u64, symbol_id: u32) -> Self {
        Self {
            event_type,
            sequence,
            timestamp_ns: read_rdtsc(),
            payload_size: 0,
            symbol_id,
            payload: [0u8; 32],
        }
    }

    /// Set payload data (up to 32 bytes inline)
    #[inline(always)]
    pub fn with_payload(mut self, data: &[u8]) -> Self {
        let len = core::cmp::min(data.len(), 32);
        self.payload[..len].copy_from_slice(&data[..len]);
        self.payload_size = len as u32;
        self
    }
}

/// Read time-stamp counter for nanosecond-precision timestamps
/// Uses rdtsc instruction directly - no syscall overhead
#[inline(always)]
fn read_rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_rdtsc;
        _rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

/// Ring buffer slot - padded to cache line boundary
#[repr(C)]
struct RingSlot {
    /// The event data
    event: UnsafeCell<Event>,
    /// Sequence number for this slot (tracks publication state)
    sequence: AtomicU64,
    /// Padding to ensure next slot starts on cache line
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Event>>() - core::mem::size_of::<AtomicU64>()],
}

impl RingSlot {
    const fn new() -> Self {
        Self {
            event: UnsafeCell::new(Event {
                event_type: EventType::Heartbeat,
                sequence: 0,
                timestamp_ns: 0,
                payload_size: 0,
                symbol_id: 0,
                payload: [0u8; 32],
            }),
            sequence: AtomicU64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Event>>() - core::mem::size_of::<AtomicU64>()],
        }
    }
}

/// Lock-free ring buffer for event storage
/// 
/// Uses sequence numbers to track publication and consumption state
/// without requiring locks or mutexes
#[repr(C)]
pub struct EventRingBuffer {
    /// The ring buffer slots
    slots: [RingSlot; RING_BUFFER_SIZE],
    /// Mask for fast modulo operation (size - 1)
    mask: usize,
    /// Producer sequence counter
    producer_seq: AtomicU64,
    /// Consumer sequence counter (for single consumer)
    consumer_seq: AtomicU64,
    /// Padding between producer and consumer counters
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 2],
    /// Total events published (statistics)
    total_published: AtomicU64,
    /// Total events consumed (statistics)
    total_consumed: AtomicU64,
}

impl EventRingBuffer {
    /// Create a new event ring buffer
    pub const fn new() -> Self {
        // Initialize all slots with default values
        const EMPTY_SLOT: RingSlot = RingSlot::new();
        Self {
            slots: [EMPTY_SLOT; RING_BUFFER_SIZE],
            mask: RING_BUFFER_SIZE - 1,
            producer_seq: AtomicU64::new(0),
            consumer_seq: AtomicU64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicU64>() * 2],
            total_published: AtomicU64::new(0),
            total_consumed: AtomicU64::new(0),
        }
    }

    /// Publish an event to the ring buffer (single producer)
    /// 
    /// Returns the sequence number if successful, None if buffer is full
    #[inline(always)]
    pub fn publish(&self, event: Event) -> Option<u64> {
        // Reserve a sequence number
        let seq = self.producer_seq.fetch_add(1, Ordering::AcqRel);
        
        // Check if we're too far ahead of consumer (buffer full)
        let consumer = self.consumer_seq.load(Ordering::Acquire);
        if seq.wrapping_sub(consumer) >= RING_BUFFER_SIZE as u64 {
            // Buffer full - roll back
            self.producer_seq.fetch_sub(1, Ordering::Release);
            return None;
        }

        // Calculate slot index
        let idx = (seq as usize) & self.mask;
        let slot = &self.slots[idx];

        // Write event data
        unsafe {
            *slot.event.get() = event;
        }

        // Mark slot as available (release semantics ensures visibility)
        slot.sequence.store(seq + 1, Ordering::Release);

        // Update statistics
        self.total_published.fetch_add(1, Ordering::Relaxed);

        Some(seq)
    }

    /// Consume an event from the ring buffer (single consumer)
    /// 
    /// Returns the event if available, None if buffer is empty
    #[inline(always)]
    pub fn consume(&self) -> Option<Event> {
        let consumer = self.consumer_seq.load(Ordering::Acquire);
        let producer = self.producer_seq.load(Ordering::Acquire);

        // Check if buffer is empty
        if consumer >= producer {
            return None;
        }

        // Calculate slot index
        let idx = (consumer as usize) & self.mask;
        let slot = &self.slots[idx];

        // Wait for slot to be ready (spin briefly)
        while slot.sequence.load(Ordering::Acquire) < consumer + 1 {
            core::hint::spin_loop();
        }

        // Read event data
        let event = unsafe { *slot.event.get() };

        // Advance consumer sequence
        self.consumer_seq.fetch_add(1, Ordering::Release);
        self.total_consumed.fetch_add(1, Ordering::Relaxed);

        Some(event)
    }

    /// Consume a batch of events (maximizes instruction cache hits)
    /// 
    /// Processes up to `max_count` events in a single call
    /// Returns the number of events processed
    #[inline(always)]
    pub fn consume_batch<F>(&self, mut handler: F, max_count: usize) -> usize
    where
        F: FnMut(&Event),
    {
        let mut count = 0;
        let consumer = self.consumer_seq.load(Ordering::Acquire);
        let producer = self.producer_seq.load(Ordering::Acquire);
        let available = producer.wrapping_sub(consumer) as usize;
        let batch_size = core::cmp::min(available, max_count);

        for i in 0..batch_size {
            let seq = consumer + i as u64;
            let idx = (seq as usize) & self.mask;
            let slot = &self.slots[idx];

            // Wait for slot to be ready
            while slot.sequence.load(Ordering::Acquire) < seq + 1 {
                core::hint::spin_loop();
            }

            // Process event
            let event = unsafe { &*slot.event.get() };
            handler(event);
            count += 1;
        }

        // Advance consumer sequence by batch size
        if count > 0 {
            self.consumer_seq.fetch_add(count as u64, Ordering::Release);
            self.total_consumed.fetch_add(count as u64, Ordering::Relaxed);
        }

        count
    }

    /// Get current buffer depth (number of unconsumed events)
    #[inline]
    pub fn depth(&self) -> usize {
        let producer = self.producer_seq.load(Ordering::Acquire);
        let consumer = self.consumer_seq.load(Ordering::Acquire);
        (producer.wrapping_sub(consumer)) as usize
    }

    /// Get total published events
    #[inline]
    pub fn total_published(&self) -> u64 {
        self.total_published.load(Ordering::Relaxed)
    }

    /// Get total consumed events
    #[inline]
    pub fn total_consumed(&self) -> u64 {
        self.total_consumed.load(Ordering::Relaxed)
    }
}

/// Event Bus - wraps the ring buffer with additional functionality
pub struct EventBus {
    /// The underlying ring buffer
    buffer: EventRingBuffer,
    /// Batch size for processing optimization
    batch_size: usize,
}

impl EventBus {
    /// Create a new event bus with specified batch size
    pub const fn new(batch_size: usize) -> Self {
        Self {
            buffer: EventRingBuffer::new(),
            batch_size,
        }
    }

    /// Publish an event
    #[inline(always)]
    pub fn publish(&self, event: Event) -> Option<u64> {
        self.buffer.publish(event)
    }

    /// Process events in batches
    #[inline(always)]
    pub fn process_batch<F>(&self, handler: F) -> usize
    where
        F: FnMut(&Event),
    {
        self.buffer.consume_batch(handler, self.batch_size)
    }

    /// Get buffer depth
    #[inline]
    pub fn depth(&self) -> usize {
        self.buffer.depth()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(64) // Default batch size optimized for L1 cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = Event::new(EventType::MarketData, 0, 1);
        assert_eq!(event.event_type, EventType::MarketData);
        assert_eq!(event.symbol_id, 1);
        assert!(event.timestamp_ns > 0);
    }

    #[test]
    fn test_ring_buffer_publish_consume() {
        let buffer = EventRingBuffer::new();
        let event = Event::new(EventType::OrderSubmission, 0, 42);
        
        assert!(buffer.publish(event).is_some());
        assert_eq!(buffer.depth(), 1);
        
        let consumed = buffer.consume();
        assert!(consumed.is_some());
        let consumed = consumed.unwrap();
        assert_eq!(consumed.event_type, EventType::OrderSubmission);
        assert_eq!(consumed.symbol_id, 42);
        assert_eq!(buffer.depth(), 0);
    }

    #[test]
    fn test_batch_processing() {
        let buffer = EventRingBuffer::new();
        
        // Publish 10 events
        for i in 0..10 {
            let event = Event::new(EventType::MarketData, i as u64, i as u32);
            buffer.publish(event);
        }
        
        // Consume in batch
        let mut count = 0;
        buffer.consume_batch(|_| { count += 1; }, 100);
        
        assert_eq!(count, 10);
        assert_eq!(buffer.depth(), 0);
    }

    #[test]
    fn test_cache_line_alignment() {
        // Verify Event fits within expected size
        let event_size = core::mem::size_of::<Event>();
        assert!(event_size <= 128, "Event should fit within 2 cache lines");
        
        // Verify RingSlot is cache-line aligned
        let slot_size = core::mem::size_of::<RingSlot>();
        assert_eq!(slot_size % CACHE_LINE_SIZE, 0, "RingSlot should be cache-line aligned");
    }
}
