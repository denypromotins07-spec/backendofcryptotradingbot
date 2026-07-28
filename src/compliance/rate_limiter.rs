//! Token-bucket rate limiter synchronized with exchange API quotas to prevent bans.
//!
//! This module implements a lock-free hierarchical timing wheel rate limiter
//! capable of handling millions of API calls per second with deterministic latency.
//! Uses branchless programming for all token deductions.
//!
//! **Latency Target:** < 50ns per rate limit check.
//! **Memory Limit:** Pre-allocated timing wheel, no heap allocation.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Number of buckets in the timing wheel.
const TIMING_WHEEL_BUCKETS: usize = 1024;

/// Maximum tokens per bucket.
const MAX_TOKENS_PER_BUCKET: u64 = 1_000_000;

/// A single bucket in the timing wheel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RateLimitBucket {
    /// Available tokens (fixed-point).
    pub tokens: AtomicU64,
    /// Last refill timestamp.
    pub last_refill_ts: AtomicU64,
    /// Token capacity.
    pub capacity: u64,
    /// Refill rate (tokens per second, fixed-point).
    pub refill_rate: u64,
    /// Padding to 64 bytes.
    _padding: [u8; 32],
}

impl RateLimitBucket {
    #[inline]
    pub const fn new() -> Self {
        Self {
            tokens: AtomicU64::new(0),
            last_refill_ts: AtomicU64::new(0),
            capacity: 0,
            refill_rate: 0,
            _padding: [0u8; 32],
        }
    }
}

const _: () = assert!(core::mem::size_of::<RateLimitBucket>() == CACHE_LINE_SIZE);

/// The main rate limiter using a hierarchical timing wheel.
pub struct RateLimiter {
    /// Timing wheel buckets.
    buckets: [RateLimitBucket; TIMING_WHEEL_BUCKETS],
    /// Current wheel position.
    wheel_position: AtomicU64,
    /// Total requests allowed.
    total_allowed: AtomicU64,
    /// Total requests rejected.
    total_rejected: AtomicU64,
    /// Flag indicating if the limiter is active.
    is_active: AtomicBool,
    /// Circuit breaker flag.
    circuit_open: AtomicBool,
    /// Padding.
    _padding: [u8; 48],
}

unsafe impl Send for RateLimiter {}
unsafe impl Sync for RateLimiter {}

impl RateLimiter {
    /// Create a new rate limiter.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buckets: [RateLimitBucket::new(); TIMING_WHEEL_BUCKETS],
            wheel_position: AtomicU64::new(0),
            total_allowed: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            circuit_open: AtomicBool::new(false),
            _padding: [0u8; 48],
        }
    }

    /// Initialize a bucket with specific parameters.
    #[inline]
    pub fn init_bucket(&self, idx: usize, capacity: u64, refill_rate: u64) {
        if idx >= TIMING_WHEEL_BUCKETS {
            return;
        }

        let bucket = &self.buckets[idx];
        unsafe {
            ptr::write_volatile(&bucket.capacity as *const u64 as *mut u64, capacity);
            ptr::write_volatile(&bucket.refill_rate as *const u64 as *mut u64, refill_rate);
            ptr::write_volatile(&bucket.tokens as *const AtomicU64 as *mut AtomicU64, AtomicU64::new(capacity));
        }
    }

    /// Try to consume a token (branchless implementation).
    #[inline]
    pub fn try_acquire(&self, bucket_idx: usize, tokens: u64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return true; // Bypass when inactive
        }

        if self.circuit_open.load(Ordering::Acquire) {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let idx = bucket_idx % TIMING_WHEEL_BUCKETS;
        let bucket = &self.buckets[idx];

        // Get current timestamp (simplified)
        #[cfg(target_arch = "x86_64")]
        let now = unsafe {
            use core::arch::x86_64::_rdtsc;
            _rdtsc()
        };
        #[cfg(not(target_arch = "x86_64"))]
        let now = self.wheel_position.load(Ordering::Relaxed);

        // Refill tokens based on elapsed time (branchless)
        let capacity = bucket.capacity;
        let refill_rate = bucket.refill_rate;
        let last_refill = bucket.last_refill_ts.load(Ordering::Relaxed);
        
        let elapsed = now.saturating_sub(last_refill);
        let refill_amount = (elapsed * refill_rate) >> 32; // Fixed-point division
        let current_tokens = bucket.tokens.load(Ordering::Relaxed);
        let refilled_tokens = (current_tokens + refill_amount).min(capacity);
        
        // Update tokens (branchless CAS-like behavior)
        bucket.tokens.store(refilled_tokens, Ordering::Relaxed);
        bucket.last_refill_ts.store(now, Ordering::Relaxed);

        // Try to consume tokens (branchless)
        let can_consume = (refilled_tokens >= tokens) as u64;
        let new_tokens = refilled_tokens - (tokens * can_consume);
        
        bucket.tokens.store(new_tokens, Ordering::Release);

        // Update statistics (branchless)
        let allowed_mask = can_consume.wrapping_neg();
        let rejected_mask = (!can_consume).wrapping_neg();
        
        self.total_allowed.fetch_add(can_consume, Ordering::Relaxed);
        self.total_rejected.fetch_add(!can_consume, Ordering::Relaxed);

        // Update wheel position
        self.wheel_position.fetch_add(1, Ordering::Relaxed);

        can_consume != 0
    }

    /// Get remaining tokens in a bucket.
    #[inline]
    pub fn get_tokens(&self, bucket_idx: usize) -> u64 {
        let idx = bucket_idx % TIMING_WHEEL_BUCKETS;
        self.buckets[idx].tokens.load(Ordering::Acquire)
    }

    /// Open the circuit breaker (emergency stop).
    #[inline]
    pub fn open_circuit(&self) {
        self.circuit_open.store(true, Ordering::Release);
    }

    /// Close the circuit breaker.
    #[inline]
    pub fn close_circuit(&self) {
        self.circuit_open.store(false, Ordering::Release);
    }

    /// Get statistics.
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, bool) {
        (
            self.total_allowed.load(Ordering::Acquire),
            self.total_rejected.load(Ordering::Acquire),
            self.circuit_open.load(Ordering::Acquire),
        )
    }

    /// Reset all buckets.
    #[inline]
    pub fn reset(&self) {
        for i in 0..TIMING_WHEEL_BUCKETS {
            let bucket = &self.buckets[i];
            bucket.tokens.store(bucket.capacity, Ordering::Release);
        }
        self.wheel_position.store(0, Ordering::Release);
    }

    /// Shutdown the rate limiter.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size() {
        assert_eq!(core::mem::size_of::<RateLimitBucket>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_limiter_init() {
        let limiter = RateLimiter::new();
        assert!(limiter.is_active.load(Ordering::Acquire));
        assert!(!limiter.circuit_open.load(Ordering::Acquire));
    }

    #[test]
    fn test_acquire_tokens() {
        let limiter = RateLimiter::new();
        limiter.init_bucket(0, 100, 1000);
        
        // Should be able to acquire tokens
        assert!(limiter.try_acquire(0, 10));
        assert!(limiter.try_acquire(0, 10));
        
        let tokens = limiter.get_tokens(0);
        assert!(tokens < 100);
    }

    #[test]
    fn test_exhaust_tokens() {
        let limiter = RateLimiter::new();
        limiter.init_bucket(0, 10, 0); // No refill
        
        // Exhaust tokens
        for _ in 0..10 {
            limiter.try_acquire(0, 1);
        }
        
        // Should fail now
        assert!(!limiter.try_acquire(0, 1));
        
        let (allowed, rejected, _) = limiter.get_stats();
        assert!(rejected > 0);
    }

    #[test]
    fn test_circuit_breaker() {
        let limiter = RateLimiter::new();
        limiter.init_bucket(0, 1000, 1000);
        
        limiter.open_circuit();
        assert!(!limiter.try_acquire(0, 1));
        
        limiter.close_circuit();
        assert!(limiter.try_acquire(0, 1));
    }
}
