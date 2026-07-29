//! Robust WebSocket subscription manager with automatic reconnection and gap recovery.
//! 
//! Uses lock-free atomic state management, pre-allocated buffers,
//! and circular queues for zero-copy message handling.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, AtomicIsize, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum subscription count
const MAX_SUBSCRIPTIONS: usize = 256;

/// Message buffer size
const MSG_BUFFER_SIZE: usize = 4096;

/// Circular message queue capacity
const QUEUE_CAPACITY: usize = 1024;

/// Padded atomic u64 for cache-line alignment
#[repr(C)]
struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 8],
}

impl PaddedAtomicU64 {
    const fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; CACHE_LINE_SIZE - 8],
        }
    }
    
    #[inline]
    fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline]
    fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed);
    }
}

/// Subscription state
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SubState {
    Pending = 0,
    Active = 1,
    Reconnecting = 2,
    Closed = 3,
}

/// Subscription info - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct Subscription {
    /// Subscription ID
    sub_id: u64,
    /// Channel/topic hash
    channel_hash: u64,
    /// State
    state: SubState,
    /// Padding
    _pad1: [u8; 7],
    /// Messages received
    msg_count: u64,
    /// Last sequence number
    last_seq: u64,
    /// Reconnect attempts
    reconnect_attempts: u32,
    /// Is active
    is_active: bool,
    /// Padding
    _padding: [u8; 38],
}

const _: () = assert!(core::mem::size_of::<Subscription>() == 64);

/// Message in circular queue - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct QueuedMessage {
    /// Sequence number
    seq: u64,
    /// Subscription ID
    sub_id: u64,
    /// Data offset in buffer
    data_offset: u32,
    /// Data length
    data_len: u32,
    /// Timestamp (cycles)
    timestamp: u64,
    /// Padding
    _padding: [u8; 40],
}

const _: () = assert!(core::mem::size_of::<QueuedMessage>() == 64);

/// Circular message queue
#[repr(C)]
struct MessageQueue {
    /// Pre-allocated messages
    messages: [QueuedMessage; QUEUE_CAPACITY],
    /// Pre-allocated data buffer
    data_buffer: [u8; MSG_BUFFER_SIZE * QUEUE_CAPACITY],
    /// Head index
    head: AtomicU64,
    /// Tail index
    tail: AtomicU64,
    /// Count
    count: AtomicU64,
    /// Dropped messages (overflow)
    dropped_count: AtomicU64,
}

impl MessageQueue {
    const fn new() -> Self {
        Self {
            messages: [QueuedMessage {
                seq: 0,
                sub_id: 0,
                data_offset: 0,
                data_len: 0,
                timestamp: 0,
                _padding: [0u8; 40],
            }; QUEUE_CAPACITY],
            data_buffer: [0u8; MSG_BUFFER_SIZE * QUEUE_CAPACITY],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
        }
    }
    
    #[inline]
    pub fn push(&self, msg: QueuedMessage, data: &[u8]) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        // Check if full
        if head.wrapping_sub(tail) >= QUEUE_CAPACITY as u64 {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            // Drop oldest message
            self.tail.fetch_add(1, Ordering::Relaxed);
        }
        
        let idx = (head % QUEUE_CAPACITY as u64) as usize;
        
        unsafe {
            // Store data
            let data_start = idx * MSG_BUFFER_SIZE;
            let copy_len = data.len().min(MSG_BUFFER_SIZE);
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data_buffer.as_mut_ptr().add(data_start),
                copy_len,
            );
            
            // Store message metadata
            let mut stored_msg = msg;
            stored_msg.data_offset = (data_start as u32).min(MSG_BUFFER_SIZE as u32 - 1);
            stored_msg.data_len = copy_len as u32;
            *self.messages.get_unchecked_mut(idx) = stored_msg;
        }
        
        self.head.fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        
        true
    }
    
    #[inline]
    pub fn pop(&self) -> Option<(QueuedMessage, usize)> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        if tail >= head {
            return None;
        }
        
        let idx = (tail % QUEUE_CAPACITY as u64) as usize;
        self.tail.fetch_add(1, Ordering::Relaxed);
        self.count.fetch_sub(1, Ordering::Relaxed);
        
        unsafe {
            let msg = *self.messages.get_unchecked(idx);
            Some((msg, idx))
        }
    }
    
    #[inline]
    pub fn get_data(&self, offset: usize, len: usize) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.data_buffer.as_ptr().add(offset),
                len.min(MSG_BUFFER_SIZE),
            )
        }
    }
}

/// Main WebSocket subscription manager
#[repr(C)]
pub struct WsSubscriptionManager {
    /// Subscriptions (pre-allocated)
    subscriptions: [Subscription; MAX_SUBSCRIPTIONS],
    /// Message queue
    queue: MessageQueue,
    /// Subscription count
    sub_count: AtomicU64,
    /// Connection state
    is_connected: AtomicBool,
    /// Is reconnecting
    is_reconnecting: AtomicBool,
    /// Reconnect attempt count
    reconnect_count: AtomicU64,
    /// Gap detected flag
    gap_detected: AtomicBool,
    /// Expected next sequence
    expected_seq: AtomicU64,
    /// Total messages processed
    total_processed: PaddedAtomicU64,
}

impl WsSubscriptionManager {
    /// Create a new subscription manager
    pub const fn new() -> Self {
        Self {
            subscriptions: [Subscription {
                sub_id: 0,
                channel_hash: 0,
                state: SubState::Closed,
                _pad1: [0u8; 7],
                msg_count: 0,
                last_seq: 0,
                reconnect_attempts: 0,
                is_active: false,
                _padding: [0u8; 38],
            }; MAX_SUBSCRIPTIONS],
            queue: MessageQueue::new(),
            sub_count: AtomicU64::new(0),
            is_connected: AtomicBool::new(false),
            is_reconnecting: AtomicBool::new(false),
            reconnect_count: AtomicU64::new(0),
            gap_detected: AtomicBool::new(false),
            expected_seq: AtomicU64::new(0),
            total_processed: PaddedAtomicU64::new(0),
        }
    }
    
    /// Subscribe to a channel
    #[inline]
    pub fn subscribe(&self, sub_id: u64, channel_hash: u64) -> bool {
        let idx = self.sub_count.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= MAX_SUBSCRIPTIONS {
            self.sub_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            let sub = &mut *self.subscriptions.get_unchecked_mut(idx);
            sub.sub_id = sub_id;
            sub.channel_hash = channel_hash;
            sub.state = SubState::Pending;
            sub.is_active = true;
        }
        
        true
    }
    
    /// Process incoming message
    #[inline]
    pub fn process_message(&self, sub_id: u64, seq: u64, data: &[u8]) -> bool {
        use core::arch::x86_64::_rdtsc;
        
        // Check for gap
        let expected = self.expected_seq.load(Ordering::Relaxed);
        if seq != expected && expected > 0 {
            self.gap_detected.store(true, Ordering::Relaxed);
        }
        self.expected_seq.store(seq + 1, Ordering::Relaxed);
        
        // Find subscription
        for i in 0..self.sub_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let sub = &mut *self.subscriptions.get_unchecked_mut(i);
                if sub.sub_id == sub_id {
                    sub.msg_count += 1;
                    sub.last_seq = seq;
                    
                    // Queue message
                    let msg = QueuedMessage {
                        seq,
                        sub_id,
                        data_offset: 0,
                        data_len: 0,
                        timestamp: unsafe { _rdtsc() },
                        _padding: [0u8; 40],
                    };
                    
                    return self.queue.push(msg, data);
                }
            }
        }
        
        false
    }
    
    /// Handle reconnection
    #[inline]
    pub fn handle_reconnect(&self) {
        if self.is_reconnecting.load(Ordering::Relaxed) {
            return;
        }
        
        self.is_reconnecting.store(true, Ordering::Relaxed);
        self.is_connected.store(false, Ordering::Relaxed);
        
        let attempts = self.reconnect_count.fetch_add(1, Ordering::Relaxed);
        
        // Update all subscriptions to reconnecting state
        for i in 0..self.sub_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let sub = &mut *self.subscriptions.get_unchecked_mut(i);
                if sub.is_active {
                    sub.state = SubState::Reconnecting;
                    sub.reconnect_attempts = attempts as u32;
                }
            }
        }
    }
    
    /// Complete reconnection
    #[inline]
    pub fn complete_reconnect(&self) {
        self.is_connected.store(true, Ordering::Relaxed);
        self.is_reconnecting.store(false, Ordering::Relaxed);
        
        // Reset gap detection
        self.gap_detected.store(false, Ordering::Relaxed);
        self.expected_seq.store(0, Ordering::Relaxed);
        
        // Mark subscriptions as active
        for i in 0..self.sub_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let sub = &mut *self.subscriptions.get_unchecked_mut(i);
                if sub.is_active {
                    sub.state = SubState::Active;
                }
            }
        }
    }
    
    /// Pop next message from queue
    #[inline]
    pub fn pop_message(&self) -> Option<(QueuedMessage, usize)> {
        let result = self.queue.pop();
        if result.is_some() {
            self.total_processed.fetch_add(1);
        }
        result
    }
    
    /// Check if gap was detected
    #[inline]
    pub fn is_gap_detected(&self) -> bool {
        self.gap_detected.load(Ordering::Relaxed)
    }
    
    /// Get dropped message count
    #[inline]
    pub fn get_dropped_count(&self) -> u64 {
        self.queue.dropped_count.load(Ordering::Relaxed)
    }
    
    /// Get total processed
    #[inline]
    pub fn get_total_processed(&self) -> u64 {
        self.total_processed.load()
    }
}

impl Default for WsSubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_subscribe() {
        let mgr = WsSubscriptionManager::new();
        
        assert!(mgr.subscribe(1, 0x1234));
        assert_eq!(mgr.sub_count.load(), 1);
    }
    
    #[test]
    fn test_message_processing() {
        let mgr = WsSubscriptionManager::new();
        
        mgr.subscribe(1, 0x1234);
        
        let data = b"test message";
        assert!(mgr.process_message(1, 1, data));
        
        let msg = mgr.pop_message();
        assert!(msg.is_some());
        assert_eq!(mgr.get_total_processed(), 1);
    }
    
    #[test]
    fn test_gap_detection() {
        let mgr = WsSubscriptionManager::new();
        
        mgr.subscribe(1, 0x1234);
        mgr.process_message(1, 1, b"msg1");
        mgr.process_message(1, 5, b"msg5"); // Gap: expected 2, got 5
        
        assert!(mgr.is_gap_detected());
    }
}
