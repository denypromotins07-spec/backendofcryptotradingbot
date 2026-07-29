//! Chapter 2: Global Event Clock & Causality Ordering
//! Clock drift and time-warp anomaly detector to prevent stale data execution.

use core::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum allowed clock drift in microseconds (100us)
const MAX_DRIFT_US: i64 = 100;

/// Time-warp detection threshold (nanoseconds going backwards)
const TIME_WARP_THRESHOLD_NS: i64 = -1_000_000; // -1ms

/// Circuit breaker trip threshold (consecutive anomalies)
const CIRCUIT_BREAKER_THRESHOLD: u64 = 3;

#[repr(C, align(64))]
pub struct TimeWarpGuard {
    /// Last observed timestamp (nanoseconds)
    last_timestamp_ns: AtomicI64,
    /// Reference clock (TSC cycles)
    reference_tsc: AtomicU64,
    /// Accumulated drift (nanoseconds)
    accumulated_drift_ns: AtomicI64,
    /// Anomaly counter for circuit breaker
    anomaly_count: AtomicU64,
    /// Circuit breaker tripped flag
    circuit_breaker_tripped: AtomicBool,
    /// Trading halted flag
    trading_halted: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8 - 2 * 1],
}

#[repr(C, align(64))]
pub struct ClockDriftMonitor {
    /// Local clock offset from reference (nanoseconds)
    offset_ns: AtomicI64,
    /// Drift rate (nanoseconds per second)
    drift_rate_ns_per_sec: AtomicI64,
    /// Last synchronization timestamp
    last_sync_ts: AtomicU64,
    /// Synchronization interval (microseconds)
    sync_interval_us: AtomicU64,
    /// Maximum observed drift
    max_drift_observed: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 5 * 8],
}

#[repr(C, align(64))]
pub struct TimestampValidator {
    /// Minimum acceptable timestamp
    min_valid_ts: AtomicI64,
    /// Maximum acceptable timestamp (prevents future timestamps)
    max_valid_ts: AtomicI64,
    /// Stale data threshold (microseconds)
    stale_threshold_us: AtomicU64,
    /// Rejected timestamp count
    rejected_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

// Compile-time assertions
const _: () = {
    assert!(MAX_DRIFT_US > 0, "Max drift must be positive");
    assert!(TIME_WARP_THRESHOLD_NS < 0, "Time warp threshold must be negative");
};

impl Default for TimeWarpGuard {
    fn default() -> Self {
        Self {
            last_timestamp_ns: AtomicI64::new(0),
            reference_tsc: AtomicU64::new(0),
            accumulated_drift_ns: AtomicI64::new(0),
            anomaly_count: AtomicU64::new(0),
            circuit_breaker_tripped: AtomicBool::new(false),
            trading_halted: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8 - 2 * 1],
        }
    }
}

impl Default for ClockDriftMonitor {
    fn default() -> Self {
        Self {
            offset_ns: AtomicI64::new(0),
            drift_rate_ns_per_sec: AtomicI64::new(0),
            last_sync_ts: AtomicU64::new(0),
            sync_interval_us: AtomicU64::new(1000), // 1ms default
            max_drift_observed: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 5 * 8],
        }
    }
}

impl Default for TimestampValidator {
    fn default() -> Self {
        Self {
            min_valid_ts: AtomicI64::new(0),
            max_valid_ts: AtomicI64::new(i64::MAX),
            stale_threshold_us: AtomicU64::new(10_000), // 10ms default
            rejected_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl TimeWarpGuard {
    /// Initialize guard with current timestamp
    #[inline]
    pub fn init(&self, initial_ts_ns: i64) {
        self.last_timestamp_ns.store(initial_ts_ns, Ordering::Relaxed);
        self.reference_tsc.store(unsafe { _rdtsc() }, Ordering::Relaxed);
        self.anomaly_count.store(0, Ordering::Relaxed);
        self.circuit_breaker_tripped.store(false, Ordering::Relaxed);
        self.trading_halted.store(false, Ordering::Relaxed);
    }

    /// Validate incoming timestamp and detect time-warps
    /// Returns: true if timestamp is valid, false if anomaly detected
    #[inline]
    pub fn validate_timestamp(&self, timestamp_ns: i64) -> bool {
        // Check circuit breaker first (branchless early exit simulation)
        let tripped = self.circuit_breaker_tripped.load(Ordering::Relaxed);
        if tripped {
            return false;
        }

        let last = self.last_timestamp_ns.load(Ordering::Relaxed);
        let delta = timestamp_ns - last;

        // Detect time-warp (timestamp going backwards beyond threshold)
        let is_time_warp = delta < TIME_WARP_THRESHOLD_NS;

        // Branchless anomaly counting
        let anomaly_increment = is_time_warp as u64;
        let new_count = self.anomaly_count.fetch_add(anomaly_increment, Ordering::Relaxed) + anomaly_increment;

        // Check circuit breaker threshold
        if new_count >= CIRCUIT_BREAKER_THRESHOLD {
            self.trip_circuit_breaker();
            return false;
        }

        // Update last timestamp if valid
        if !is_time_warp {
            self.last_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);

            // Update drift estimate
            let current_tsc = unsafe { _rdtsc() };
            let ref_tsc = self.reference_tsc.load(Ordering::Relaxed);
            let tsc_delta = current_tsc.wrapping_sub(ref_tsc);

            // Approximate TSC frequency (assuming ~2.5GHz = 2.5 cycles/ns)
            let expected_ns = (tsc_delta / 2) as i64;
            let drift = delta - expected_ns;
            let accum = self.accumulated_drift_ns.load(Ordering::Relaxed);
            self.accumulated_drift_ns.store(accum + drift, Ordering::Relaxed);
        }

        !is_time_warp
    }

    /// Trip the circuit breaker to halt trading
    #[inline]
    pub fn trip_circuit_breaker(&self) {
        self.circuit_breaker_tripped.store(true, Ordering::SeqCst);
        self.trading_halted.store(true, Ordering::SeqCst);
    }

    /// Reset circuit breaker (requires manual intervention)
    #[inline]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker_tripped.store(false, Ordering::SeqCst);
        self.trading_halted.store(false, Ordering::SeqCst);
        self.anomaly_count.store(0, Ordering::Relaxed);
    }

    /// Check if trading is halted
    #[inline]
    pub fn is_trading_halted(&self) -> bool {
        self.trading_halted.load(Ordering::Relaxed)
    }

    /// Get accumulated drift
    #[inline]
    pub fn get_accumulated_drift_ns(&self) -> i64 {
        self.accumulated_drift_ns.load(Ordering::Relaxed)
    }

    /// Get anomaly count
    #[inline]
    pub fn get_anomaly_count(&self) -> u64 {
        self.anomaly_count.load(Ordering::Relaxed)
    }

    /// SIMD-accelerated multi-source timestamp validation
    #[inline]
    pub fn simd_validate_multiple<const N: usize>(&self, timestamps: &[i64; N]) -> u32
    where [i64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let last = self.last_timestamp_ns.load(Ordering::Relaxed);
        let mut valid_mask = 0u32;

        unsafe {
            if N == 4 {
                let ts_vec = _mm256_load_si256(timestamps.as_ptr() as *const __m256i);
                let last_vec = _mm256_set1_epi64x(last);
                let threshold_vec = _mm256_set1_epi64x(TIME_WARP_THRESHOLD_NS);

                // Calculate deltas
                let delta_vec = _mm256_sub_epi64(ts_vec, last_vec);

                // Check if delta >= threshold (not a time-warp)
                let cmp_vec = _mm256_cmpgt_epi64(delta_vec, threshold_vec);

                // Extract mask
                let mask = _mm256_movemask_epi8(cmp_vec);
                valid_mask = ((mask & 0x1) | ((mask >> 4) & 0x2) | ((mask >> 7) & 0x4) | ((mask >> 11) & 0x8)) as u32;

                _mm256_zeroupper();
            } else {
                for i in 0..N {
                    let delta = timestamps[i] - last;
                    let bit = ((delta >= TIME_WARP_THRESHOLD_NS) as u32) << i;
                    valid_mask |= bit;
                }
            }
        }

        valid_mask
    }

    /// Force halt trading (external trigger)
    #[inline]
    pub fn force_halt(&self) {
        self.trip_circuit_breaker();
    }
}

impl ClockDriftMonitor {
    /// Synchronize with reference clock
    #[inline]
    pub fn sync(&self, reference_ts_ns: i64, local_ts_ns: i64, current_ts_us: u64) {
        let offset = local_ts_ns - reference_ts_ns;
        self.offset_ns.store(offset, Ordering::Relaxed);

        // Calculate drift rate
        let last_sync = self.last_sync_ts.load(Ordering::Relaxed);
        if last_sync > 0 {
            let elapsed_us = current_ts_us.saturating_sub(last_sync);
            if elapsed_us > 0 {
                let elapsed_sec = elapsed_us as i64 / 1_000_000;
                if elapsed_sec > 0 {
                    let drift_rate = offset / elapsed_sec;
                    self.drift_rate_ns_per_sec.store(drift_rate, Ordering::Relaxed);

                    // Track maximum drift
                    let max_drift = self.max_drift_observed.load(Ordering::Relaxed);
                    let abs_drift = offset.abs();
                    if abs_drift > max_drift {
                        self.max_drift_observed.store(abs_drift, Ordering::Relaxed);
                    }
                }
            }
        }

        self.last_sync_ts.store(current_ts_us, Ordering::Relaxed);
    }

    /// Get current offset from reference
    #[inline]
    pub fn get_offset_ns(&self) -> i64 {
        self.offset_ns.load(Ordering::Relaxed)
    }

    /// Get drift rate
    #[inline]
    pub fn get_drift_rate_ns_per_sec(&self) -> i64 {
        self.drift_rate_ns_per_sec.load(Ordering::Relaxed)
    }

    /// Predict future offset based on drift rate
    #[inline]
    pub fn predict_offset_ns(&self, seconds_ahead: i64) -> i64 {
        let current = self.offset_ns.load(Ordering::Relaxed);
        let rate = self.drift_rate_ns_per_sec.load(Ordering::Relaxed);
        current + rate * seconds_ahead
    }

    /// Check if drift exceeds threshold
    #[inline]
    pub fn is_drift_excessive(&self, threshold_ns: i64) -> bool {
        let offset = self.offset_ns.load(Ordering::Relaxed).abs();
        offset > threshold_ns
    }

    /// Get maximum observed drift
    #[inline]
    pub fn get_max_drift_observed(&self) -> i64 {
        self.max_drift_observed.load(Ordering::Relaxed)
    }

    /// Set synchronization interval
    #[inline]
    pub fn set_sync_interval_us(&self, interval_us: u64) {
        self.sync_interval_us.store(interval_us, Ordering::Relaxed);
    }

    /// Check if synchronization is due
    #[inline]
    pub fn is_sync_due(&self, current_ts_us: u64) -> bool {
        let last_sync = self.last_sync_ts.load(Ordering::Relaxed);
        let interval = self.sync_interval_us.load(Ordering::Relaxed);
        current_ts_us.saturating_sub(last_sync) >= interval
    }
}

impl TimestampValidator {
    /// Set valid timestamp range
    #[inline]
    pub fn set_valid_range(&self, min_ts: i64, max_ts: i64) {
        self.min_valid_ts.store(min_ts, Ordering::Relaxed);
        self.max_valid_ts.store(max_ts, Ordering::Relaxed);
    }

    /// Validate timestamp against bounds and staleness
    #[inline]
    pub fn validate(&self, timestamp_ns: i64, current_time_ns: i64) -> bool {
        let min_ts = self.min_valid_ts.load(Ordering::Relaxed);
        let max_ts = self.max_valid_ts.load(Ordering::Relaxed);
        let stale_us = self.stale_threshold_us.load(Ordering::Relaxed);

        // Branchless validation
        let in_range = (timestamp_ns >= min_ts) & (timestamp_ns <= max_ts);
        let age_ns = current_time_ns - timestamp_ns;
        let age_us = age_ns / 1000;
        let not_stale = (age_us as u64) < stale_us;

        let is_valid = in_range & not_stale;

        // Increment rejection counter (branchless)
        let rejected = (is_valid == 0) as u64;
        self.rejected_count.fetch_add(rejected, Ordering::Relaxed);

        is_valid != 0
    }

    /// Update minimum valid timestamp (sliding window)
    #[inline]
    pub fn slide_min_window(&self, new_min: i64) {
        let current = self.min_valid_ts.load(Ordering::Relaxed);
        if new_min > current {
            self.min_valid_ts.store(new_min, Ordering::Relaxed);
        }
    }

    /// Get rejection count
    #[inline]
    pub fn get_rejected_count(&self) -> u64 {
        self.rejected_count.load(Ordering::Relaxed)
    }

    /// Set stale threshold
    #[inline]
    pub fn set_stale_threshold_us(&self, threshold_us: u64) {
        self.stale_threshold_us.store(threshold_us, Ordering::Relaxed);
    }

    /// Check if timestamp is stale
    #[inline]
    pub fn is_stale(&self, timestamp_ns: i64, current_time_ns: i64) -> bool {
        let age_ns = current_time_ns - timestamp_ns;
        let age_us = age_ns / 1000;
        let threshold = self.stale_threshold_us.load(Ordering::Relaxed);
        age_us as u64 >= threshold
    }
}

/// rdtsc wrapper for TSC-based timing
#[inline]
fn get_tsc_cycles() -> u64 {
    unsafe { _rdtsc() }
}

/// Convert TSC cycles to nanoseconds (approximate, assuming 2.5GHz)
#[inline]
fn tsc_to_ns(cycles: u64) -> i64 {
    (cycles / 2) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_warp_detection() {
        let guard = TimeWarpGuard::default();
        guard.init(1_000_000_000); // 1 second in ns

        // Valid forward timestamp
        assert!(guard.validate_timestamp(1_001_000_000));

        // Time-warp (going backwards significantly)
        assert!(!guard.validate_timestamp(500_000_000));

        assert!(guard.get_anomaly_count() >= 1);
    }

    #[test]
    fn test_circuit_breaker() {
        let guard = TimeWarpGuard::default();
        guard.init(1_000_000_000);

        // Trigger multiple anomalies
        for _ in 0..CIRCUIT_BREAKER_THRESHOLD + 1 {
            guard.validate_timestamp(500_000_000); // Time-warp each time
        }

        assert!(guard.is_trading_halted());

        // Verify circuit breaker blocks further validation
        assert!(!guard.validate_timestamp(2_000_000_000));
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let guard = TimeWarpGuard::default();
        guard.init(1_000_000_000);

        // Trip the breaker
        for _ in 0..CIRCUIT_BREAKER_THRESHOLD + 1 {
            guard.validate_timestamp(500_000_000);
        }

        assert!(guard.is_trading_halted());

        // Reset
        guard.reset_circuit_breaker();
        assert!(!guard.is_trading_halted());
        assert_eq!(guard.get_anomaly_count(), 0);
    }

    #[test]
    fn test_clock_drift_monitor() {
        let monitor = ClockDriftMonitor::default();

        // Initial sync
        monitor.sync(1_000_000_000, 1_000_100_000, 1000); // 100ms offset
        assert_eq!(monitor.get_offset_ns(), 100_000_000);

        // Second sync to calculate drift rate
        monitor.sync(2_000_000_000, 2_000_200_000, 2000); // Still 200ms offset after 1ms
        assert!(monitor.get_drift_rate_ns_per_sec() != 0 || monitor.get_offset_ns() == 200_000_000);
    }

    #[test]
    fn test_timestamp_validator() {
        let validator = TimestampValidator::default();
        validator.set_valid_range(0, 10_000_000_000_000); // 0 to 10000 seconds

        let current = 5_000_000_000_000; // 5000 seconds

        // Valid timestamp
        assert!(validator.validate(4_999_000_000_000, current));

        // Stale timestamp (older than 10ms)
        assert!(!validator.validate(4_000_000_000_000, current));

        // Future timestamp beyond max
        assert!(!validator.validate(11_000_000_000_000, current));

        assert!(validator.get_rejected_count() >= 2);
    }

    #[test]
    fn test_simd_multi_validation() {
        let guard = TimeWarpGuard::default();
        guard.init(1_000_000_000);

        let timestamps = [
            1_001_000_000, // Valid
            500_000_000,   // Time-warp
            1_002_000_000, // Valid
            400_000_000,   // Time-warp
        ];

        let mask = guard.simd_validate_multiple(&timestamps);

        // Expected: [true, false, true, false] = 0b0101 = 5
        assert_eq!(mask, 0b0101);
    }

    #[test]
    fn test_force_halt() {
        let guard = TimeWarpGuard::default();
        guard.init(1_000_000_000);

        assert!(!guard.is_trading_halted());

        guard.force_halt();

        assert!(guard.is_trading_halted());
        assert!(guard.circuit_breaker_tripped.load(Ordering::Relaxed));
    }

    #[test]
    fn test_drift_prediction() {
        let monitor = ClockDriftMonitor::default();

        // Set up known drift
        monitor.offset_ns.store(100_000, Ordering::Relaxed); // 100us offset
        monitor.drift_rate_ns_per_sec.store(1000, Ordering::Relaxed); // 1us/sec drift

        let predicted = monitor.predict_offset_ns(10); // 10 seconds ahead
        assert_eq!(predicted, 110_000); // 100us + 10us
    }
}
