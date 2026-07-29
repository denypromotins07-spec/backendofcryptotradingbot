//! Chapter 1: Margin, Leverage, and Open Interest Dynamics
//! Predictive funding rate model and countdown timer for perpetual swaps.

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Fixed-point scaling factor (1e9)
const FP_SCALE: i64 = 1_000_000_000;

/// Funding rate interval in seconds (8 hours typical)
const FUNDING_INTERVAL_SEC: u64 = 28_800;

#[repr(C, align(64))]
pub struct FundingRateModel {
    /// Current funding rate (scaled by 1e9)
    current_rate: AtomicI64,
    /// Predicted next funding rate
    predicted_rate: AtomicI64,
    /// Premium index accumulator
    premium_sum: AtomicI64,
    /// Sample count for premium average
    sample_count: AtomicU64,
    /// Last funding timestamp
    last_funding_ts: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 6 * 8],
}

#[repr(C, align(64))]
pub struct FundingCountdown {
    /// Next funding timestamp in microseconds
    next_funding_us: AtomicU64,
    /// Current timestamp in microseconds
    current_ts_us: AtomicU64,
    /// Time remaining until funding (microseconds)
    time_remaining_us: AtomicU64,
    /// Funding period identifier
    period_id: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 5 * 8],
}

#[repr(C, align(64))]
pub struct RollingFundingBuffer {
    /// Circular buffer for funding rate history (capacity 64)
    buffer: [i64; 64],
    /// Head index
    head: AtomicU64,
    /// Sum for O(1) average calculation
    sum: AtomicI64,
    /// Variance accumulator for volatility estimation
    variance_sum: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 64 * 8 - 3 * 8],
}

impl Default for FundingRateModel {
    fn default() -> Self {
        Self {
            current_rate: AtomicI64::new(0),
            predicted_rate: AtomicI64::new(0),
            premium_sum: AtomicI64::new(0),
            sample_count: AtomicU64::new(0),
            last_funding_ts: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 6 * 8],
        }
    }
}

impl Default for FundingCountdown {
    fn default() -> Self {
        Self {
            next_funding_us: AtomicU64::new(0),
            current_ts_us: AtomicU64::new(0),
            time_remaining_us: AtomicU64::new(0),
            period_id: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 5 * 8],
        }
    }
}

impl Default for RollingFundingBuffer {
    fn default() -> Self {
        Self {
            buffer: [0i64; 64],
            head: AtomicU64::new(0),
            sum: AtomicI64::new(0),
            variance_sum: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 64 * 8 - 3 * 8],
        }
    }
}

impl FundingRateModel {
    /// Update premium index sample for funding rate calculation
    #[inline]
    pub fn update_premium(&self, premium: i64) {
        self.premium_sum.fetch_add(premium, Ordering::Relaxed);
        self.sample_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Calculate funding rate from accumulated premiums (fixed-point)
    #[inline]
    pub fn calc_funding_rate(&self) -> i64 {
        let sum = self.premium_sum.load(Ordering::Relaxed);
        let count = self.sample_count.load(Ordering::Relaxed);
        
        if count == 0 {
            return 0;
        }

        // Funding Rate = clamp(premium_avg, -0.75%, 0.75%) per 8h
        // Scaled: clamp(sum / count, -7_500_000, 7_500_000)
        let avg = sum / count as i64;
        
        // Branchless clamp
        let clamped = avg.max(-7_500_000).min(7_500_000);
        
        clamped
    }

    /// Predict next funding rate using linear regression on recent samples
    #[inline]
    pub fn predict_next_rate(&self, buffer: &RollingFundingBuffer) -> i64 {
        let count = buffer.head.load(Ordering::Relaxed).min(64);
        if count < 2 {
            return self.current_rate.load(Ordering::Relaxed);
        }

        // Simple momentum prediction: rate + (rate - prev_rate) * decay
        let idx1 = ((count - 1) % 64) as usize;
        let idx2 = ((count - 2) % 64) as usize;
        
        unsafe {
            let current = *buffer.buffer.get_unchecked(idx1);
            let previous = *buffer.buffer.get_unchecked(idx2);
            let delta = current - previous;
            
            // Apply decay factor (0.7 scaled to 1e9)
            let decay = 700_000_000;
            let prediction = current + (delta * decay / FP_SCALE);
            
            // Clamp prediction
            prediction.max(-10_000_000).min(10_000_000)
        }
    }

    /// Update current and predicted rates
    #[inline]
    pub fn update_rates(&self, buffer: &RollingFundingBuffer) {
        let new_rate = self.calc_funding_rate();
        self.current_rate.store(new_rate, Ordering::Relaxed);
        
        let predicted = self.predict_next_rate(buffer);
        self.predicted_rate.store(predicted, Ordering::Relaxed);
        
        // Reset premium accumulator for next period
        self.premium_sum.store(0, Ordering::Relaxed);
        self.sample_count.store(0, Ordering::Relaxed);
    }

    /// Get current funding rate
    #[inline]
    pub fn get_current_rate(&self) -> i64 {
        self.current_rate.load(Ordering::Relaxed)
    }

    /// Get predicted funding rate
    #[inline]
    pub fn get_predicted_rate(&self) -> i64 {
        self.predicted_rate.load(Ordering::Relaxed)
    }

    /// SIMD-accelerated funding rate comparison across multiple venues
    #[inline]
    pub fn simd_funding_spread<const N: usize>(
        &self, 
        venue_rates: &[i64; N], 
        benchmark: i64
    ) -> [i64; N] 
    where [i64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut result = [0i64; N];
        
        unsafe {
            if N == 4 {
                let rates_vec = _mm256_load_si256(venue_rates.as_ptr() as *const __m256i);
                let bench_vec = _mm256_set1_epi64x(benchmark);
                let spread_vec = _mm256_sub_epi64(rates_vec, bench_vec);
                _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, spread_vec);
            } else {
                for i in 0..N {
                    result[i] = venue_rates[i] - benchmark;
                }
            }
        }
        result
    }
}

impl FundingCountdown {
    /// Initialize countdown with current timestamp
    #[inline]
    pub fn init(&self, current_ts_us: u64) {
        self.current_ts_us.store(current_ts_us, Ordering::Relaxed);
        
        // Calculate next funding boundary (aligned to 8h intervals)
        let interval_us = FUNDING_INTERVAL_SEC * 1_000_000;
        let elapsed = current_ts_us % interval_us;
        let remaining = interval_us - elapsed;
        let next_funding = current_ts_us + remaining;
        
        self.next_funding_us.store(next_funding, Ordering::Relaxed);
        self.time_remaining_us.store(remaining, Ordering::Relaxed);
        self.period_id.fetch_add(1, Ordering::Relaxed);
    }

    /// Update current timestamp and recalculate remaining time
    #[inline]
    pub fn tick(&self, current_ts_us: u64) -> u64 {
        self.current_ts_us.store(current_ts_us, Ordering::Relaxed);
        
        let next = self.next_funding_us.load(Ordering::Relaxed);
        let remaining = if current_ts_us >= next {
            // Funding occurred, calculate next interval
            let interval_us = FUNDING_INTERVAL_SEC * 1_000_000;
            let new_next = next + interval_us;
            self.next_funding_us.store(new_next, Ordering::Relaxed);
            self.period_id.fetch_add(1, Ordering::Relaxed);
            new_next - current_ts_us
        } else {
            next - current_ts_us
        };
        
        self.time_remaining_us.store(remaining, Ordering::Relaxed);
        remaining
    }

    /// Get time remaining until next funding (microseconds)
    #[inline]
    pub fn time_remaining(&self) -> u64 {
        self.time_remaining_us.load(Ordering::Relaxed)
    }

    /// Get time remaining in human-readable format (branchless)
    #[inline]
    pub fn time_remaining_formatted(&self) -> (u32, u32, u32) {
        let remaining_us = self.time_remaining_us.load(Ordering::Relaxed);
        let hours = (remaining_us / 3_600_000_000) as u32;
        let minutes = ((remaining_us % 3_600_000_000) / 60_000_000) as u32;
        let seconds = ((remaining_us % 60_000_000) / 1_000_000) as u32;
        (hours, minutes, seconds)
    }

    /// Check if funding event is imminent (< 1 minute)
    #[inline]
    pub fn is_imminent(&self) -> bool {
        self.time_remaining_us.load(Ordering::Relaxed) < 60_000_000
    }

    /// Get current period ID
    #[inline]
    pub fn get_period_id(&self) -> u64 {
        self.period_id.load(Ordering::Relaxed)
    }
}

impl RollingFundingBuffer {
    /// Push new funding rate into circular buffer with O(1) operations
    #[inline]
    pub fn push(&self, rate: i64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let idx = (head % 64) as usize;
        
        // Load old value for sum update
        let old_value = unsafe { *self.buffer.get_unchecked(idx) };
        
        // Update sum
        let old_sum = self.sum.load(Ordering::Relaxed);
        self.sum.store(old_sum - old_value + rate, Ordering::Relaxed);
        
        // Update variance sum (simplified E[X^2] tracking)
        let old_sq = old_value * old_value / FP_SCALE;
        let new_sq = rate * rate / FP_SCALE;
        let old_var_sum = self.variance_sum.load(Ordering::Relaxed);
        self.variance_sum.store(old_var_sum - old_sq + new_sq, Ordering::Relaxed);
        
        // Store new value
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = rate;
        }
    }

    /// Get rolling average funding rate
    #[inline]
    pub fn rolling_average(&self) -> i64 {
        let count = self.head.load(Ordering::Relaxed).min(64);
        if count == 0 {
            return 0;
        }
        self.sum.load(Ordering::Relaxed) / count as i64
    }

    /// Get funding rate volatility estimate
    #[inline]
    pub fn volatility_estimate(&self) -> i64 {
        let count = self.head.load(Ordering::Relaxed).min(64);
        if count < 2 {
            return 0;
        }
        
        let mean = self.rolling_average();
        let mean_sq = mean * mean / FP_SCALE;
        let var_sum = self.variance_sum.load(Ordering::Relaxed);
        
        // Variance = E[X^2] - E[X]^2
        let variance = (var_sum / count as i64 - mean_sq).max(0);
        
        // Simplified sqrt approximation for std dev
        // Using Newton-Raphson iteration (2 iterations for speed)
        if variance == 0 {
            return 0;
        }
        
        let mut guess = variance / 2;
        guess = (guess + variance / guess) / 2;
        guess = (guess + variance / guess) / 2;
        
        guess
    }

    /// Get historical funding rate at specific offset
    #[inline]
    pub fn get_at_offset(&self, offset: u64) -> i64 {
        let count = self.head.load(Ordering::Relaxed);
        if offset >= count || offset >= 64 {
            return 0;
        }
        let idx = ((count - 1 - offset) % 64) as usize;
        unsafe { *self.buffer.get_unchecked(idx) }
    }
}

/// Funding rate arbitrage opportunity detector
#[repr(C, align(64))]
pub struct FundingArbDetector {
    /// Threshold for arb opportunity (scaled by 1e9)
    threshold: AtomicI64,
    /// Last detected spread
    last_spread: AtomicI64,
    /// Opportunity counter
    opportunity_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 3 * 8],
}

impl Default for FundingArbDetector {
    fn default() -> Self {
        Self {
            threshold: AtomicI64::new(1_000_000), // 0.1% threshold
            last_spread: AtomicI64::new(0),
            opportunity_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 3 * 8],
        }
    }
}

impl FundingArbDetector {
    /// Check for funding rate arbitrage between two venues
    #[inline]
    pub fn check_arb(&self, rate_a: i64, rate_b: i64) -> bool {
        let spread = (rate_a - rate_b).abs();
        self.last_spread.store(spread, Ordering::Relaxed);
        
        let threshold = self.threshold.load(Ordering::Relaxed);
        
        // Branchless comparison
        let is_opportunity = (spread > threshold) as u64;
        self.opportunity_count.fetch_add(is_opportunity, Ordering::Relaxed);
        
        is_opportunity != 0
    }

    /// Get opportunity count
    #[inline]
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunity_count.load(Ordering::Relaxed)
    }

    /// Set detection threshold
    #[inline]
    pub fn set_threshold(&self, threshold: i64) {
        self.threshold.store(threshold.max(0), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_rate_calculation() {
        let model = FundingRateModel::default();
        
        // Add premium samples
        for _ in 0..100 {
            model.update_premium(500_000); // 0.05% premium
        }
        
        let rate = model.calc_funding_rate();
        assert_eq!(rate, 500_000); // Should equal average premium
    }

    #[test]
    fn test_funding_countdown() {
        let countdown = FundingCountdown::default();
        countdown.init(100_000_000); // 100ms
        
        let remaining = countdown.tick(150_000_000);
        assert!(remaining > 0);
        assert!(!countdown.is_imminent());
    }

    #[test]
    fn test_rolling_buffer_operations() {
        let buffer = RollingFundingBuffer::default();
        
        for i in 0..100 {
            buffer.push(i * 10_000);
        }
        
        // After 100 pushes, should have last 64 values (36..99)
        let avg = buffer.rolling_average();
        assert_eq!(avg, (36 + 99) * 5_000); // Average of 36..99
    }

    #[test]
    fn test_funding_arb_detection() {
        let detector = FundingArbDetector::default();
        
        // No arb when spread below threshold
        assert!(!detector.check_arb(500_000, 500_500));
        
        // Arb when spread above threshold
        assert!(detector.check_arb(500_000, 502_000));
        
        assert!(detector.get_opportunity_count() >= 1);
    }

    #[test]
    fn test_simd_funding_spread() {
        let model = FundingRateModel::default();
        let rates = [100_000, 200_000, 300_000, 400_000];
        let benchmark = 250_000;
        
        let spreads = model.simd_funding_spread(&rates, benchmark);
        
        assert_eq!(spreads[0], -150_000);
        assert_eq!(spreads[1], -50_000);
        assert_eq!(spreads[2], 50_000);
        assert_eq!(spreads[3], 150_000);
    }

    #[test]
    fn test_rate_clamping() {
        let model = FundingRateModel::default();
        
        // Add extreme premium samples
        for _ in 0..10 {
            model.update_premium(100_000_000); // 10% premium (should clamp)
        }
        
        let rate = model.calc_funding_rate();
        assert!(rate <= 7_500_000); // Should be clamped to 0.75%
    }
}
