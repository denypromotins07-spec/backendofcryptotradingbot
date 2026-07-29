//! Chapter 1: Margin, Leverage, and Open Interest Dynamics
//! Real-time OI delta and long/short ratio tracker using lock-free accumulators.

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
pub struct OiAccumulator {
    /// Long open interest in fixed-point (scaled by 1e9)
    long_oi: AtomicI64,
    /// Short open interest in fixed-point (scaled by 1e9)
    short_oi: AtomicI64,
    /// Delta accumulator for OI changes
    oi_delta: AtomicI64,
    /// Tick counter for rolling window
    tick_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

#[repr(C, align(64))]
pub struct RollingOiBuffer {
    /// Circular buffer for OI history (capacity 256)
    buffer: [i64; 256],
    /// Head index for circular buffer
    head: AtomicU64,
    /// Sum accumulator for O(1) rolling calculations
    sum: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 256 * 8 - 2 * 8],
}

impl Default for OiAccumulator {
    fn default() -> Self {
        Self {
            long_oi: AtomicI64::new(0),
            short_oi: AtomicI64::new(0),
            oi_delta: AtomicI64::new(0),
            tick_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl Default for RollingOiBuffer {
    fn default() -> Self {
        Self {
            buffer: [0i64; 256],
            head: AtomicU64::new(0),
            sum: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 256 * 8 - 2 * 8],
        }
    }
}

impl OiAccumulator {
    /// Update long OI with lock-free atomic operation
    #[inline]
    pub fn update_long(&self, delta: i64) {
        self.long_oi.fetch_add(delta, Ordering::Relaxed);
        self.oi_delta.fetch_add(delta.abs(), Ordering::Relaxed);
    }

    /// Update short OI with lock-free atomic operation
    #[inline]
    pub fn update_short(&self, delta: i64) {
        self.short_oi.fetch_add(delta, Ordering::Relaxed);
        self.oi_delta.fetch_add(delta.abs(), Ordering::Relaxed);
    }

    /// Get current long/short ratio in fixed-point (scaled by 1e9)
    #[inline]
    pub fn long_short_ratio(&self) -> i64 {
        let long = self.long_oi.load(Ordering::Relaxed);
        let short = self.short_oi.load(Ordering::Relaxed);
        if short == 0 {
            return i64::MAX;
        }
        // Fixed-point division: (long * 1e9) / short
        (long * 1_000_000_000) / short
    }

    /// Get total open interest
    #[inline]
    pub fn total_oi(&self) -> i64 {
        self.long_oi.load(Ordering::Relaxed) + self.short_oi.load(Ordering::Relaxed)
    }

    /// Get OI delta since last reset
    #[inline]
    pub fn get_delta(&self) -> i64 {
        self.oi_delta.load(Ordering::Relaxed)
    }

    /// Reset delta accumulator
    #[inline]
    pub fn reset_delta(&self) {
        self.oi_delta.store(0, Ordering::Relaxed);
    }
}

impl RollingOiBuffer {
    /// Push new OI value into circular buffer with O(1) sum update
    #[inline]
    pub fn push(&self, oi_value: i64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let idx = (head % 256) as usize;
        
        // Load old value and update sum atomically
        let old_value = unsafe { *self.buffer.get_unchecked(idx) };
        let new_sum = self.sum.load(Ordering::Relaxed) - old_value + oi_value;
        self.sum.store(new_sum, Ordering::Relaxed);
        
        // Store new value
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = oi_value;
        }
    }

    /// Get rolling average OI in O(1)
    #[inline]
    pub fn rolling_average(&self, count: u64) -> i64 {
        let actual_count = count.min(256);
        if actual_count == 0 {
            return 0;
        }
        self.sum.load(Ordering::Relaxed) / actual_count as i64
    }

    /// SIMD-accelerated OI delta calculation across multiple symbols
    #[inline]
    pub fn simd_oi_delta<const N: usize>(&self, current: &[i64; N], previous: &[i64; N]) -> [i64; N] 
    where [i64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut result = [0i64; N];
        
        unsafe {
            if N == 4 {
                let curr_vec = _mm256_load_si256(current.as_ptr() as *const __m256i);
                let prev_vec = _mm256_load_si256(previous.as_ptr() as *const __m256i);
                let delta_vec = _mm256_sub_epi64(curr_vec, prev_vec);
                _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, delta_vec);
            } else {
                for i in 0..N {
                    result[i] = current[i] - previous[i];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oi_accumulator_basic() {
        let acc = OiAccumulator::default();
        acc.update_long(1_000_000_000);
        acc.update_short(500_000_000);
        
        assert_eq!(acc.total_oi(), 1_500_000_000);
        assert_eq!(acc.long_short_ratio(), 2_000_000_000); // 2:1 ratio scaled
    }

    #[test]
    fn test_rolling_buffer_o1() {
        let buffer = RollingOiBuffer::default();
        for i in 0..300 {
            buffer.push(i * 1_000_000);
        }
        // After 300 pushes, only last 256 remain
        let avg = buffer.rolling_average(256);
        assert_eq!(avg, (44 + 299) * 500_000); // Average of 44..299
    }

    #[test]
    fn test_lock_free_concurrent_access() {
        use std::thread;
        let acc = OiAccumulator::default();
        
        let handles: Vec<_> = (0..10).map(|_| {
            thread::spawn(|| {
                for _ in 0..1000 {
                    acc.update_long(100);
                    acc.update_short(50);
                }
            })
        }).collect();
        
        for h in handles {
            h.join().unwrap();
        }
        
        assert_eq!(acc.total_oi(), 10 * 1000 * 150);
    }
}
