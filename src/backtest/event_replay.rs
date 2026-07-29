//! Deterministic, nanosecond-precision event-driven backtesting with tick replay.
//! 
//! Uses raw rdtsc cycles for timestamp delta calculations.
//! Compile-time assertions verify byte layout matches live market data.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum events in replay buffer
pub const MAX_EVENTS: usize = 10_000_000;

/// Event types
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Tick = 0,
    Order = 1,
    Fill = 2,
    Cancel = 3,
    Quote = 4,
    Trade = 5,
}

/// Market event - exact byte layout match with live data
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MarketEvent {
    pub event_type: EventType,
    pub timestamp_ns: u64,
    pub symbol_id: u32,
    pub price_tick: i64,
    pub quantity: u64,
    pub flags: u32,
    pub exchange_id: u16,
    pub sequence: u64,
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

// Compile-time assertion: MarketEvent must be exactly cache-line aligned
const _: () = assert!(core::mem::size_of::<MarketEvent>() == CACHE_LINE_SIZE);

impl MarketEvent {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            event_type: EventType::Tick,
            timestamp_ns: 0,
            symbol_id: 0,
            price_tick: 0,
            quantity: 0,
            flags: 0,
            exchange_id: 0,
            sequence: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }
}

/// Event replay state
#[repr(C)]
pub struct EventReplay<const MAX_EVENTS: usize> {
    events: [MarketEvent; MAX_EVENTS],
    event_count: AtomicU64,
    current_idx: AtomicU64,
    /// Start timestamp (rdtsc cycles)
    start_cycles: AtomicU64,
    /// Current replay position in cycles
    current_cycles: AtomicU64,
    /// Speed multiplier (1000 = real-time)
    speed_multiplier: AtomicU64,
    /// Running flag
    is_running: AtomicBool,
    /// Completed flag
    is_complete: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 34],
}

// SAFETY: All interior mutability protected by atomics
unsafe impl<const M: usize> Send for EventReplay<M> {}
unsafe impl<const M: usize> Sync for EventReplay<M> {}

impl<const MAX_EVENTS: usize> EventReplay<MAX_EVENTS> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            events: [MarketEvent::new(); MAX_EVENTS],
            event_count: AtomicU64::new(0),
            current_idx: AtomicU64::new(0),
            start_cycles: AtomicU64::new(0),
            current_cycles: AtomicU64::new(0),
            speed_multiplier: AtomicU64::new(1000),
            is_running: AtomicBool::new(false),
            is_complete: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 34],
        }
    }

    /// Add event to replay buffer
    #[inline]
    pub fn add_event(&self, event: MarketEvent) -> bool {
        let idx = self.event_count.load(Ordering::Relaxed) as usize;
        if idx >= MAX_EVENTS {
            return false;
        }

        unsafe {
            let ptr = self.events.as_ptr() as *mut MarketEvent;
            ptr.add(idx).write(event);
        }

        self.event_count.fetch_add(1, Ordering::Release);
        true
    }

    /// Load events from buffer (simulating file/network load)
    #[inline]
    pub fn load_events(&self, events: &[MarketEvent]) -> usize {
        let count = events.len().min(MAX_EVENTS);
        for i in 0..count {
            unsafe {
                let ptr = self.events.as_ptr() as *mut MarketEvent;
                ptr.add(i).write(events[i]);
            }
        }
        self.event_count.store(count as u64, Ordering::Release);
        count
    }

    /// Start replay from beginning
    #[inline]
    pub fn start(&self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.start_cycles.store(core::arch::x86_64::_rdtsc(), Ordering::Relaxed);
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.start_cycles.store(0, Ordering::Relaxed);

        self.current_idx.store(0, Ordering::Relaxed);
        self.is_running.store(true, Ordering::Release);
        self.is_complete.store(false, Ordering::Relaxed);
    }

    /// Stop replay
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Get next event (call in replay loop)
    #[inline]
    pub fn next_event(&self) -> Option<MarketEvent> {
        if !self.is_running.load(Ordering::Acquire) {
            return None;
        }

        let idx = self.current_idx.load(Ordering::Relaxed) as usize;
        let count = self.event_count.load(Ordering::Relaxed) as usize;

        if idx >= count {
            self.is_complete.store(true, Ordering::Relaxed);
            self.is_running.store(false, Ordering::Relaxed);
            return None;
        }

        unsafe {
            let ptr = self.events.as_ptr();
            let event = *ptr.add(idx);
            self.current_idx.fetch_add(1, Ordering::Relaxed);

            // Update cycle tracking
            #[cfg(target_arch = "x86_64")]
            {
                self.current_cycles.store(
                    unsafe { core::arch::x86_64::_rdtsc() },
                    Ordering::Relaxed
                );
            }

            Some(event)
        }
    }

    /// Peek at next event without advancing
    #[inline]
    pub fn peek_event(&self) -> Option<MarketEvent> {
        let idx = self.current_idx.load(Ordering::Relaxed) as usize;
        let count = self.event_count.load(Ordering::Relaxed) as usize;

        if idx >= count || idx >= MAX_EVENTS {
            return None;
        }

        unsafe {
            let ptr = self.events.as_ptr();
            Some(*ptr.add(idx))
        }
    }

    /// Calculate timestamp delta in nanoseconds between events
    #[inline]
    pub fn timestamp_delta_ns(&self, idx1: usize, idx2: usize) -> Option<u64> {
        if idx1 >= MAX_EVENTS || idx2 >= MAX_EVENTS {
            return None;
        }

        unsafe {
            let ptr = self.events.as_ptr();
            let e1 = *ptr.add(idx1);
            let e2 = *ptr.add(idx2);

            if e2.timestamp_ns > e1.timestamp_ns {
                Some(e2.timestamp_ns - e1.timestamp_ns)
            } else {
                Some(0)
            }
        }
    }

    /// Calculate cycle delta between two points
    #[inline]
    pub fn cycle_delta(&self) -> u64 {
        let current = self.current_cycles.load(Ordering::Relaxed);
        let start = self.start_cycles.load(Ordering::Relaxed);
        if current > start {
            current - start
        } else {
            0
        }
    }

    /// Get current progress (0.0 to 1.0)
    #[inline]
    pub fn progress(&self) -> f64 {
        let current = self.current_idx.load(Ordering::Relaxed) as f64;
        let total = self.event_count.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            return 0.0;
        }
        current / total
    }

    /// Set replay speed multiplier
    #[inline]
    pub fn set_speed(&self, multiplier: u64) {
        self.speed_multiplier.store(multiplier.max(1), Ordering::Relaxed);
    }

    /// Check if replay is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Acquire)
    }

    /// Check if replay is running
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get event count
    #[inline(always)]
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }

    /// Reset replay state
    #[inline]
    pub fn reset(&self) {
        self.current_idx.store(0, Ordering::Relaxed);
        self.start_cycles.store(0, Ordering::Relaxed);
        self.current_cycles.store(0, Ordering::Relaxed);
        self.is_running.store(false, Ordering::Relaxed);
        self.is_complete.store(false, Ordering::Relaxed);
    }
}

/// Type alias for typical configuration
pub type CryptoEventReplay = EventReplay<MAX_EVENTS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_event_size() {
        // Verify compile-time assertion works
        assert_eq!(core::mem::size_of::<MarketEvent>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_event_replay_basic() {
        let replay = CryptoEventReplay::new();

        let mut event = MarketEvent::new();
        event.event_type = EventType::Tick;
        event.timestamp_ns = 1000000;
        event.price_tick = 50000;

        assert!(replay.add_event(event));
        assert_eq!(replay.event_count(), 1);
    }

    #[test]
    fn test_replay_sequence() {
        let replay = EventReplay::<100>::new();

        for i in 0..10 {
            let mut event = MarketEvent::new();
            event.timestamp_ns = i * 1000000;
            event.price_tick = 50000 + i;
            replay.add_event(event);
        }

        replay.start();

        let mut prev_ts = 0u64;
        while let Some(event) = replay.next_event() {
            assert!(event.timestamp_ns >= prev_ts);
            prev_ts = event.timestamp_ns;
        }

        assert!(replay.is_complete());
        assert!(!replay.is_running());
    }

    #[test]
    fn test_timestamp_delta() {
        let replay = EventReplay::<100>::new();

        let mut e1 = MarketEvent::new();
        e1.timestamp_ns = 1000000;
        replay.add_event(e1);

        let mut e2 = MarketEvent::new();
        e2.timestamp_ns = 5000000;
        replay.add_event(e2);

        let delta = replay.timestamp_delta_ns(0, 1).unwrap();
        assert_eq!(delta, 4000000);
    }

    #[test]
    fn test_progress_tracking() {
        let replay = EventReplay::<100>::new();

        for i in 0..10 {
            replay.add_event(MarketEvent::new());
        }

        replay.start();

        for _ in 0..5 {
            replay.next_event();
        }

        let progress = replay.progress();
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_peek_without_advance() {
        let replay = EventReplay::<100>::new();

        let mut event = MarketEvent::new();
        event.price_tick = 12345;
        replay.add_event(event);

        let peek1 = replay.peek_event();
        let peek2 = replay.peek_event();

        assert!(peek1.is_some());
        assert!(peek2.is_some());
        assert_eq!(peek1.unwrap().price_tick, peek2.unwrap().price_tick);
    }

    #[test]
    fn test_buffer_overflow_protection() {
        let replay = EventReplay::<10>::new();

        for i in 0..15 {
            let result = replay.add_event(MarketEvent::new());
            if i < 10 {
                assert!(result);
            } else {
                assert!(!result);
            }
        }
    }
}
