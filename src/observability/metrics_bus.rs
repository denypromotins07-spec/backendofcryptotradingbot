//! Lock-free metrics aggregation bus using atomic counters and histograms.
//!
//! This module provides a zero-allocation metrics collection system using
//! atomic operations and pre-allocated histogram buckets. All metrics are
//! aggregated in a lock-free manner suitable for HFT workloads.
//!
//! **Latency Target:** < 50ns per metric update.
//! **Memory Limit:** Pre-allocated buffers only, no heap allocation.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of metric keys supported.
const MAX_METRICS: usize = 256;

/// Number of histogram buckets per metric.
const HISTOGRAM_BUCKETS: usize = 32;

/// A single metric slot with atomic counters.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetricSlot {
    /// Counter value (for counters).
    pub counter: AtomicU64,
    /// Sum of values (for gauges/averages).
    pub sum: AtomicU64,
    /// Count of observations (for averages).
    pub count: AtomicU64,
    /// Minimum value observed (fixed-point).
    pub min_val: AtomicU64,
    /// Maximum value observed (fixed-point).
    pub max_val: AtomicU64,
    /// Histogram bucket counts (pre-allocated).
    pub histogram: [AtomicU64; HISTOGRAM_BUCKETS],
    /// Last update timestamp.
    pub last_update_ts: AtomicU64,
    /// Flag indicating if this slot is active.
    pub is_active: AtomicBool,
}

impl MetricSlot {
    #[inline]
    pub const fn new() -> Self {
        const EMPTY_ATOMIC: AtomicU64 = AtomicU64::new(0);
        const EMPTY_HIST: [AtomicU64; HISTOGRAM_BUCKETS] = [EMPTY_ATOMIC; HISTOGRAM_BUCKETS];
        
        Self {
            counter: EMPTY_ATOMIC,
            sum: EMPTY_ATOMIC,
            count: EMPTY_ATOMIC,
            min_val: AtomicU64::new(u64::MAX),
            max_val: AtomicU64::new(0),
            histogram: EMPTY_HIST,
            last_update_ts: EMPTY_ATOMIC,
            is_active: AtomicBool::new(false),
        }
    }

    /// Increment the counter.
    #[inline]
    pub fn inc(&self, delta: u64) {
        self.counter.fetch_add(delta, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Record an observation for histogram/average.
    #[inline]
    pub fn observe(&self, value: u64) {
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        
        // Update min/max branchlessly
        let current_min = self.min_val.load(Ordering::Relaxed);
        let current_max = self.max_val.load(Ordering::Relaxed);
        
        self.min_val.fetch_min(value, Ordering::Relaxed);
        self.max_val.fetch_max(value, Ordering::Relaxed);
        
        // Update histogram bucket
        let bucket = (value.trailing_zeros() as usize).min(HISTOGRAM_BUCKETS - 1);
        self.histogram[bucket].fetch_add(1, Ordering::Relaxed);
        
        self.update_timestamp();
    }

    /// Get the current average.
    #[inline]
    pub fn average(&self) -> f64 {
        let sum = self.sum.load(Ordering::Acquire) as f64;
        let count = self.count.load(Ordering::Acquire) as f64;
        if count < 1.0 {
            return 0.0;
        }
        sum / count
    }

    /// Update the timestamp.
    #[inline]
    fn update_timestamp(&self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::_rdtsc;
            self.last_update_ts.store(_rdtsc(), Ordering::Relaxed);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.last_update_ts.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Ensure MetricSlot fits within reasonable bounds (may exceed one cache line due to histogram)
const _: () = assert!(core::mem::size_of::<MetricSlot>() <= 256);

/// The main metrics bus.
///
/// Provides lock-free metric registration and updates.
pub struct MetricsBus {
    /// Pre-allocated metric slots.
    metrics: [MetricSlot; MAX_METRICS],
    /// Count of registered metrics.
    registered_count: AtomicU64,
    /// Flag indicating if the bus is active.
    is_active: AtomicBool,
    /// Padding.
    _padding: [u8; 56],
}

unsafe impl Send for MetricsBus {}
unsafe impl Sync for MetricsBus {}

impl MetricsBus {
    /// Create a new metrics bus.
    #[inline]
    pub const fn new() -> Self {
        Self {
            metrics: [MetricSlot::new(); MAX_METRICS],
            registered_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 56],
        }
    }

    /// Register a new metric by name hash.
    #[inline]
    pub fn register(&self, name_hash: u64) -> Option<usize> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        let idx = self.registered_count.load(Ordering::Acquire) as usize;
        if idx >= MAX_METRICS {
            return None;
        }

        // Try to claim this slot atomically
        let claimed = self.registered_count.compare_exchange(
            idx as u64,
            (idx + 1) as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match claimed {
            Ok(_) => {
                self.metrics[idx].is_active.store(true, Ordering::Release);
                Some(idx)
            }
            Err(_) => None, // Another thread claimed it
        }
    }

    /// Get a metric slot by index.
    #[inline]
    pub fn get_metric(&self, idx: usize) -> Option<&MetricSlot> {
        if idx >= MAX_METRICS {
            return None;
        }
        if !self.metrics[idx].is_active.load(Ordering::Acquire) {
            return None;
        }
        Some(&self.metrics[idx])
    }

    /// Increment a counter metric.
    #[inline]
    pub fn increment_counter(&self, idx: usize, delta: u64) {
        if let Some(metric) = self.get_metric(idx) {
            metric.inc(delta);
        }
    }

    /// Record an observation for a histogram metric.
    #[inline]
    pub fn record_observation(&self, idx: usize, value: u64) {
        if let Some(metric) = self.get_metric(idx) {
            metric.observe(value);
        }
    }

    /// Get all metric summaries.
    #[inline]
    pub fn get_summaries(&self, output: &mut [(u64, f64, u64, u64); MAX_METRICS]) -> usize {
        let count = self.registered_count.load(Ordering::Acquire) as usize;
        for i in 0..count {
            let metric = &self.metrics[i];
            output[i] = (
                metric.counter.load(Ordering::Acquire),
                metric.average(),
                metric.min_val.load(Ordering::Acquire),
                metric.max_val.load(Ordering::Acquire),
            );
        }
        count
    }

    /// Shutdown the metrics bus.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

impl Default for MetricsBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_slot_init() {
        let slot = MetricSlot::new();
        assert!(!slot.is_active.load(Ordering::Acquire));
        assert_eq!(slot.counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_bus_register() {
        let bus = MetricsBus::new();
        let idx = bus.register(0x12345678).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(bus.registered_count.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_increment_and_observe() {
        let bus = MetricsBus::new();
        let idx = bus.register(0x12345678).unwrap();
        
        bus.increment_counter(idx, 10);
        bus.increment_counter(idx, 5);
        
        bus.record_observation(idx, 100);
        bus.record_observation(idx, 200);
        bus.record_observation(idx, 300);
        
        let metric = bus.get_metric(idx).unwrap();
        assert_eq!(metric.counter.load(Ordering::Acquire), 15);
        assert!((metric.average() - 200.0).abs() < 0.01);
        assert_eq!(metric.min_val.load(Ordering::Acquire), 100);
        assert_eq!(metric.max_val.load(Ordering::Acquire), 300);
    }
}
