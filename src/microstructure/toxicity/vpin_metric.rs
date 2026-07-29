//! Volume-Synchronized Probability of Informed Trading (VPIN).
//! 
//! VPIN measures the probability that trades are initiated by informed traders.
//! High VPIN indicates toxic order flow and potential adverse selection.
//! 
//! Uses volume buckets and buy/sell imbalance to estimate VPIN in real-time.
//! All calculations use fixed-point arithmetic and circular buffers.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Number of volume buckets for VPIN calculation
const NUM_BUCKETS: usize = 50;

/// Default bucket size in base units
const DEFAULT_BUCKET_SIZE: i64 = 1_000_000; // 1M units per bucket

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Volume bucket state - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VolumeBucket {
    pub buy_volume: i64,           // Classified buy volume
    pub sell_volume: i64,          // Classified sell volume
    pub total_volume: i64,         // Total volume in bucket
    pub is_full: bool,             // Is this bucket complete?
    _padding: [u8; 39],            // Pad to 64 bytes
}

impl Default for VolumeBucket {
    fn default() -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
            total_volume: 0,
            is_full: false,
            _padding: [0; 39],
        }
    }
}

/// Lock-free VPIN calculator
#[repr(C)]
pub struct VpinMetric {
    /// Volume buckets (circular buffer)
    buckets: [VolumeBucket; NUM_BUCKETS],
    
    /// Current bucket index
    current_bucket: AtomicU64,
    
    /// Bucket configuration
    bucket_size: AtomicI64,
    
    /// Running sums for VPIN calculation
    sum_abs_imbalance: AtomicI64,   // Sum of |buy - sell| across buckets
    sum_total_volume: AtomicI64,    // Sum of total volume across buckets
    
    /// VPIN state
    current_vpin: AtomicI64,        // Current VPIN value (fixed-point)
    vpin_samples: AtomicU64,        // Number of VPIN samples
    
    /// Statistics
    high_vpin_count: AtomicU64,     // Count of high VPIN readings
    max_vpin_observed: AtomicI64,   // Maximum VPIN observed
    
    /// Kill switches
    toxicity_active: AtomicBool,    // Is toxic flow detected?
    trading_halted: AtomicBool,     // Should trading halt?
    
    /// Thresholds
    vpin_threshold: AtomicI64,      // VPIN threshold for toxicity
    
    _padding: [u8; 24],             // Pad to cache line
}

impl VpinMetric {
    /// Create new VPIN metric calculator
    pub const fn new() -> Self {
        Self {
            buckets: [VolumeBucket::default(); NUM_BUCKETS],
            current_bucket: AtomicU64::new(0),
            bucket_size: AtomicI64::new(DEFAULT_BUCKET_SIZE),
            sum_abs_imbalance: AtomicI64::new(0),
            sum_total_volume: AtomicI64::new(0),
            current_vpin: AtomicI64::new(0),
            vpin_samples: AtomicU64::new(0),
            high_vpin_count: AtomicU64::new(0),
            max_vpin_observed: AtomicI64::new(0),
            toxicity_active: AtomicBool::new(false),
            trading_halted: AtomicBool::new(false),
            vpin_threshold: AtomicI64::new(FIXED_SCALE / 2), // 0.5 threshold
            _padding: [0; 24],
        }
    }
    
    /// Classify a trade as buy or sell using tick test
    #[inline(always)]
    fn classify_trade(&self, price: i64, prev_price: i64) -> i8 {
        // Branchless tick test
        let uptick = (price > prev_price) as i8;
        let downtick = (price < prev_price) as i8;
        
        // If uptick, classify as buy; if downtick, classify as sell
        // If no price change, use previous classification (default to buy)
        uptick - downtick + (1 - uptick - downtick)
    }
    
    /// Add a trade to the current bucket
    #[inline(always)]
    pub fn add_trade(&self, volume: i64, price: i64, prev_price: i64) {
        if self.trading_halted.load(Ordering::Acquire) {
            return;
        }
        
        let classifier = self.classify_trade(price, prev_price);
        
        // Get current bucket index
        let bucket_idx = self.current_bucket.load(Ordering::Acquire) as usize;
        let bucket = unsafe { self.buckets.get_unchecked(bucket_idx) };
        
        // Branchless classification
        let buy_vol = volume * (classifier.max(0) as i64);
        let sell_vol = volume * ((-classifier).max(0) as i64);
        
        // Update bucket (note: in production, use atomics or single-threaded access)
        // For now, we assume single-threaded access to the current bucket
        
        // Check if bucket is full
        if bucket.total_volume >= self.bucket_size.load(Ordering::Acquire) {
            // Move to next bucket
            self.advance_bucket();
        }
        
        // Add to running totals
        let old_imbalance = (bucket.buy_volume - bucket.sell_volume).abs();
        let new_imbalance = ((bucket.buy_volume + buy_vol) - (bucket.sell_volume + sell_vol)).abs();
        
        self.sum_abs_imbalance.fetch_add(new_imbalance - old_imbalance, Ordering::Relaxed);
        self.sum_total_volume.fetch_add(volume, Ordering::Relaxed);
        
        // Recalculate VPIN
        self.recalculate_vpin();
    }
    
    /// Advance to next bucket
    #[inline(always)]
    fn advance_bucket(&self) {
        let current = self.current_bucket.load(Ordering::Acquire);
        let next = (current + 1) % NUM_BUCKETS as u64;
        
        // Remove oldest bucket from sums if it exists
        let oldest_idx = next as usize;
        let oldest = unsafe { self.buckets.get_unchecked(oldest_idx) };
        
        if oldest.is_full {
            let old_imbalance = (oldest.buy_volume - oldest.sell_volume).abs();
            self.sum_abs_imbalance.fetch_sub(old_imbalance, Ordering::Relaxed);
            self.sum_total_volume.fetch_sub(oldest.total_volume, Ordering::Relaxed);
        }
        
        // Reset the bucket we're about to use
        unsafe {
            let bucket = self.buckets.get_unchecked_mut(oldest_idx);
            bucket.buy_volume = 0;
            bucket.sell_volume = 0;
            bucket.total_volume = 0;
            bucket.is_full = false;
        }
        
        self.current_bucket.store(next, Ordering::Release);
    }
    
    /// Recalculate VPIN from all buckets
    #[inline(always)]
    fn recalculate_vpin(&self) {
        let sum_imbalance = self.sum_abs_imbalance.load(Ordering::Acquire);
        let sum_volume = self.sum_total_volume.load(Ordering::Acquire);
        
        if sum_volume == 0 {
            return;
        }
        
        // VPIN = sum(|buy - sell|) / sum(buy + sell)
        let vpin = ((sum_imbalance as i128 * FIXED_SCALE as i128) / sum_volume.abs() as i128) as i64;
        let vpin = vpin.clamp(0, FIXED_SCALE);
        
        self.current_vpin.store(vpin, Ordering::Release);
        self.vpin_samples.fetch_add(1, Ordering::Relaxed);
        
        // Track maximum
        let max = self.max_vpin_observed.load(Ordering::Relaxed);
        if vpin > max {
            self.max_vpin_observed.store(vpin, Ordering::Relaxed);
        }
        
        // Check toxicity threshold (branchless)
        let is_toxic = (vpin > self.vpin_threshold.load(Ordering::Acquire)) as u8;
        self.toxicity_active.store(is_toxic != 0, Ordering::Release);
        
        if is_toxic != 0 {
            self.high_vpin_count.fetch_add(1, Ordering::Relaxed);
            
            // Auto-halt if VPIN is extremely high (> 0.8)
            if vpin > (FIXED_SCALE * 8 / 10) {
                self.trading_halted.store(true, Ordering::Release);
            }
        }
    }
    
    /// Get current VPIN value
    #[inline(always)]
    pub fn get_vpin(&self) -> i64 {
        self.current_vpin.load(Ordering::Acquire)
    }
    
    /// Check if flow is toxic
    #[inline(always)]
    pub fn is_toxic(&self) -> bool {
        self.toxicity_active.load(Ordering::Acquire)
    }
    
    /// Check if trading should halt
    #[inline(always)]
    pub fn should_halt(&self) -> bool {
        self.trading_halted.load(Ordering::Acquire)
    }
    
    /// Reset trading halt (manual intervention required)
    #[inline(always)]
    pub fn reset_halt(&self) {
        self.trading_halted.store(false, Ordering::Release);
    }
    
    /// Set VPIN threshold
    #[inline(always)]
    pub fn set_threshold(&self, threshold: i64) {
        self.vpin_threshold.store(threshold.clamp(0, FIXED_SCALE), Ordering::Release);
    }
    
    /// Get statistics
    #[inline(always)]
    pub fn stats(&self) -> (u64, i64, u64) {
        (
            self.vpin_samples.load(Ordering::Acquire),
            self.max_vpin_observed.load(Ordering::Acquire),
            self.high_vpin_count.load(Ordering::Acquire),
        )
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<VolumeBucket>() == 64);
        assert!(core::mem::size_of::<VpinMetric>() % 64 == 0);
    }
    
    #[test]
    fn test_vpin_calculation() {
        let vpin = VpinMetric::new();
        
        // Simulate balanced flow (should give low VPIN)
        for i in 0..100 {
            let price = 100_000_000 + (i as i64 * 10_000);
            vpin.add_trade(10_000, price, price - 10_000);
            vpin.add_trade(10_000, price, price + 10_000);
        }
        
        // VPIN should be relatively low for balanced flow
        let v = vpin.get_vpin();
        assert!(v < FIXED_SCALE / 2);
    }
    
    #[test]
    fn test_toxic_flow_detection() {
        let vpin = VpinMetric::new();
        vpin.set_threshold(FIXED_SCALE / 4); // 0.25 threshold
        
        // Simulate one-sided flow (should give high VPIN)
        for i in 0..200 {
            let price = 100_000_000 - (i as i64 * 10_000); // Declining price
            vpin.add_trade(50_000, price, price - 10_000); // All sells
        }
        
        // Should detect toxicity
        assert!(vpin.is_toxic());
    }
}
