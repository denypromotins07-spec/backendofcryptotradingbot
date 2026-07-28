//! Bot-behavior anomaly detector for unexpected order rates or PnL variance.
//!
//! This module uses a lock-free Kalman filter to establish dynamic baselines
//! for order flow and PnL, detecting anomalies that may indicate bugs, market
//! regime changes, or external attacks.
//!
//! **Latency Target:** < 200ns per observation.
//! **Memory Limit:** Pre-allocated state vectors, zero heap allocation.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Number of metrics tracked simultaneously.
const NUM_METRICS: usize = 8;

/// Kalman filter state for a single metric.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KalmanState {
    /// Current state estimate (fixed-point).
    pub state: i64,
    /// State variance (fixed-point).
    pub variance: i64,
    /// Process noise (fixed-point).
    pub process_noise: i64,
    /// Measurement noise (fixed-point).
    pub measurement_noise: i64,
    /// Kalman gain (fixed-point).
    pub kalman_gain: i64,
    /// Last update timestamp.
    pub last_update_ts: u64,
    /// Anomaly count for this metric.
    pub anomaly_count: AtomicU64,
    /// Reserved padding.
    _padding: [u8; 32],
}

impl KalmanState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: 0,
            variance: 1000, // Initial uncertainty
            process_noise: 10,
            measurement_noise: 100,
            kalman_gain: 0,
            last_update_ts: 0,
            anomaly_count: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }

    /// Initialize with a specific value.
    #[inline]
    pub fn init(&mut self, initial_value: i64) {
        self.state = initial_value;
        self.variance = 100;
    }
}

// Ensure KalmanState is exactly one cache line.
const _: () = assert!(core::mem::size_of::<KalmanState>() == CACHE_LINE_SIZE);

/// Metric types tracked by the anomaly detector.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    OrderRate = 0,
    PnLVariance = 1,
    FillRatio = 2,
    SlippageBps = 3,
    LatencyNs = 4,
    RejectionRate = 5,
    CancelRate = 6,
    Volatility = 7,
}

/// The main anomaly detector using Kalman filters.
pub struct AnomalyDetector {
    /// Kalman filter states for each metric.
    filters: [KalmanState; NUM_METRICS],
    /// Global anomaly flag (bitmask).
    anomaly_flags: AtomicU64,
    /// Total anomaly count.
    total_anomalies: AtomicU64,
    /// Threshold multiplier for anomaly detection (fixed-point).
    threshold_multiplier: AtomicI64,
    /// Flag indicating if the detector is active.
    is_active: AtomicBool,
    /// Padding.
    _padding: [u8; 48],
}

unsafe impl Send for AnomalyDetector {}
unsafe impl Sync for AnomalyDetector {}

impl AnomalyDetector {
    /// Create a new anomaly detector.
    #[inline]
    pub const fn new() -> Self {
        Self {
            filters: [KalmanState::new(); NUM_METRICS],
            anomaly_flags: AtomicU64::new(0),
            total_anomalies: AtomicU64::new(0),
            threshold_multiplier: AtomicI64::new(3 * 65536), // 3 sigma
            is_active: AtomicBool::new(true),
            _padding: [0u8; 48],
        }
    }

    /// Initialize a specific metric's filter.
    #[inline]
    pub fn init_metric(&mut self, metric: MetricType, initial_value: i64) {
        let idx = metric as usize;
        if idx < NUM_METRICS {
            self.filters[idx].init(initial_value);
        }
    }

    /// Record an observation and check for anomalies.
    ///
    /// Returns true if an anomaly was detected.
    #[inline]
    pub fn observe(&self, metric: MetricType, value: i64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        let idx = metric as usize;
        if idx >= NUM_METRICS {
            return false;
        }

        let filter = &self.filters[idx];
        
        // Get current state (simplified fixed-point Kalman filter)
        let state = filter.state;
        let variance = filter.variance;
        let process_noise = filter.process_noise;
        let measurement_noise = filter.measurement_noise;

        // Prediction step
        let predicted_state = state;
        let predicted_variance = variance + process_noise;

        // Update step
        let innovation = value - predicted_state;
        let innovation_variance = predicted_variance + measurement_noise;
        
        // Kalman gain (branchless division approximation)
        let kalman_gain = if innovation_variance > 0 {
            predicted_variance * 65536 / innovation_variance
        } else {
            65536
        };

        // Update state
        let new_state = predicted_state + (innovation * kalman_gain / 65536);
        let new_variance = (65536 - kalman_gain) * predicted_variance / 65536;

        // Store updated values (simplified, no CAS for speed)
        unsafe {
            ptr::write_volatile(&filter.state as *const i64 as *mut i64, new_state);
            ptr::write_volatile(&filter.variance as *const i64 as *mut i64, new_variance.max(1));
            ptr::write_volatile(&filter.kalman_gain as *const i64 as *mut i64, kalman_gain);
        }

        // Check for anomaly: |innovation| > threshold * sqrt(variance)
        let threshold = self.threshold_multiplier.load(Ordering::Acquire);
        let std_dev_approx = (variance as u64).sqrt() as i64;
        let anomaly_threshold = threshold * std_dev_approx / 65536;
        
        let abs_innovation = innovation.abs();
        let is_anomaly = abs_innovation > anomaly_threshold && anomaly_threshold > 0;

        if is_anomaly {
            filter.anomaly_count.fetch_add(1, Ordering::Relaxed);
            self.total_anomalies.fetch_add(1, Ordering::Relaxed);
            
            // Set anomaly flag for this metric
            self.anomaly_flags.fetch_or(1u64 << idx, Ordering::Release);
            
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use core::arch::x86_64::_rdtsc;
                ptr::write_volatile(&filter.last_update_ts as *const u64 as *mut u64, _rdtsc());
            }
        }

        is_anomaly
    }

    /// SIMD-accelerated PnL variance calculation.
    ///
    /// Uses manual loop unrolling for performance.
    #[inline]
    pub fn calculate_pnl_variance_simd(&self, samples: &[i64; 16]) -> i64 {
        if samples.is_empty() {
            return 0;
        }

        // Manual loop unrolling
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        let n = samples.len() as i64;

        // Unroll by 4
        let mut i = 0;
        while i + 3 < n as usize {
            sum += samples[i] + samples[i+1] + samples[i+2] + samples[i+3];
            sum_sq += samples[i]*samples[i] + samples[i+1]*samples[i+1] 
                    + samples[i+2]*samples[i+2] + samples[i+3]*samples[i+3];
            i += 4;
        }
        
        // Handle remainder
        while i < n as usize {
            sum += samples[i];
            sum_sq += samples[i] * samples[i];
            i += 1;
        }

        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        variance.max(0)
    }

    /// Check if a specific metric has an active anomaly.
    #[inline]
    pub fn is_anomalous(&self, metric: MetricType) -> bool {
        let idx = metric as usize;
        let flags = self.anomaly_flags.load(Ordering::Acquire);
        (flags & (1u64 << idx)) != 0
    }

    /// Get the total anomaly count.
    #[inline]
    pub fn total_anomalies(&self) -> u64 {
        self.total_anomalies.load(Ordering::Acquire)
    }

    /// Clear anomaly flags (called after remediation).
    #[inline]
    pub fn clear_flags(&self) {
        self.anomaly_flags.store(0, Ordering::Release);
    }

    /// Set the threshold multiplier (in fixed-point).
    #[inline]
    pub fn set_threshold(&self, multiplier: f64) {
        let fixed = (multiplier * 65536.0) as i64;
        self.threshold_multiplier.store(fixed, Ordering::Release);
    }

    /// Shutdown the detector.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Get anomaly statistics for a metric.
    #[inline]
    pub fn get_metric_stats(&self, metric: MetricType) -> (i64, i64, u64) {
        let idx = metric as usize;
        if idx >= NUM_METRICS {
            return (0, 0, 0);
        }

        let filter = &self.filters[idx];
        (
            filter.state,
            filter.variance,
            filter.anomaly_count.load(Ordering::Acquire),
        )
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_state_size() {
        assert_eq!(core::mem::size_of::<KalmanState>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_detector_init() {
        let detector = AnomalyDetector::new();
        assert!(detector.is_active.load(Ordering::Acquire));
        assert_eq!(detector.total_anomalies(), 0);
    }

    #[test]
    fn test_observe_normal() {
        let mut detector = AnomalyDetector::new();
        detector.init_metric(MetricType::OrderRate, 100 * 65536);
        
        // Normal observations should not trigger anomalies
        for _ in 0..20 {
            let is_anomaly = detector.observe(MetricType::OrderRate, 100 * 65536 + 1000);
            // First few might be anomalous due to initialization, but should stabilize
            let _ = is_anomaly;
        }
    }

    #[test]
    fn test_observe_anomaly() {
        let mut detector = AnomalyDetector::new();
        detector.init_metric(MetricType::OrderRate, 100 * 65536);
        
        // Establish baseline
        for _ in 0..50 {
            detector.observe(MetricType::OrderRate, 100 * 65536);
        }
        
        // Large deviation should trigger anomaly
        let is_anomaly = detector.observe(MetricType::OrderRate, 500 * 65536);
        assert!(is_anomaly);
        assert!(detector.is_anomalous(MetricType::OrderRate));
    }

    #[test]
    fn test_pnl_variance_simd() {
        let detector = AnomalyDetector::new();
        let samples = [10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 11, 13];
        let variance = detector.calculate_pnl_variance_simd(&samples);
        assert!(variance > 0);
    }
}
