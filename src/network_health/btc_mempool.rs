//! Bitcoin mempool fee estimator and congestion tracker for settlement risk.
//! 
//! Uses fixed-point arithmetic, rdtsc-based timestamp deltas for microsecond precision,
//! and lock-free atomic state management for deterministic latency.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of mempool entries tracked
const MAX_MEMPOOL_ENTRIES: usize = 4096;

/// Number of fee buckets in histogram
const FEE_BUCKETS: usize = 64;

/// Fixed-point scale (8 decimal precision for BTC)
const FIXED_SCALE: u64 = 100_000_000;

/// Minimum feerate in sat/vB
const MIN_FEERATE_SAT_VB: u64 = 1;

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

/// Mempool transaction entry - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct MempoolEntry {
    /// Transaction ID hash
    txid_hash: u64,
    /// Fee in satoshis
    fee_sats: u64,
    /// Virtual size in bytes
    vsize_bytes: u32,
    /// Feerate in sat/vB (fixed-point)
    feerate_fixed: u64,
    /// Entry time (rdtsc cycles)
    entry_time_cycles: u64,
    /// Ancestor count
    ancestor_count: u32,
    /// Descendant count
    descendant_count: u32,
    /// Is RBF replaceable
    is_rbf: bool,
    /// Padding to reach 64 bytes
    _padding: [u8; 35],
}

const _: () = assert!(core::mem::size_of::<MempoolEntry>() == 64);

/// Fee bucket for histogram - cache-line aligned
#[repr(C)]
struct FeeBucket {
    /// Minimum feerate (sat/vB)
    min_feerate: u64,
    /// Maximum feerate (sat/vB)
    max_feerate: u64,
    /// Transaction count in bucket
    tx_count: AtomicU64,
    /// Total size in bytes
    total_size: AtomicU64,
    /// Padding
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

impl FeeBucket {
    const fn new(min_rate: u64, max_rate: u64) -> Self {
        Self {
            min_feerate: min_rate,
            max_feerate: max_rate,
            tx_count: AtomicU64::new(0),
            total_size: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }
}

/// Circular buffer for rolling mempool size calculations
#[repr(C)]
struct MempoolSizeBuffer {
    /// Pre-allocated buffer
    buffer: [u64; 1440], // 24 hours at 1-minute intervals
    /// Head index
    head: AtomicU64,
    /// Sum for rolling average
    sum: AtomicU64,
    /// Window size
    window_size: usize,
}

impl MempoolSizeBuffer {
    const fn new(window_size: usize) -> Self {
        Self {
            buffer: [0u64; 1440],
            head: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            window_size,
        }
    }
    
    #[inline]
    pub fn push(&self, size: u64) -> u64 {
        let head = self.head.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = head % self.window_size;
        
        let old = unsafe { *self.buffer.get_unchecked(idx) };
        let delta = if size >= old { size - old } else { 0 };
        
        let new_sum = self.sum.fetch_add(delta, Ordering::Relaxed) + delta;
        
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = size;
        }
        
        new_sum
    }
    
    #[inline]
    pub fn get_average(&self) -> u64 {
        let count = core::cmp::min(self.head.load(Ordering::Relaxed) as usize, self.window_size);
        if count == 0 { return 0; }
        self.sum.load(Ordering::Relaxed) / count as u64
    }
}

/// Read CPU timestamp counter (rdtsc)
#[inline]
fn rdtsc() -> u64 {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            core::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0 // Fallback for non-x86
        }
    }
}

/// Main Bitcoin mempool tracker
#[repr(C)]
pub struct BtcMempoolTracker {
    /// Mempool entries (pre-allocated)
    entries: [MempoolEntry; MAX_MEMPOOL_ENTRIES],
    /// Fee histogram buckets
    fee_buckets: [FeeBucket; FEE_BUCKETS],
    /// Entry count
    entry_count: AtomicU64,
    /// Total mempool size in bytes
    total_size_bytes: PaddedAtomicU64,
    /// Total fees in satoshis
    total_fees_sats: PaddedAtomicU64,
    /// Rolling mempool size buffer
    size_history: MempoolSizeBuffer,
    /// Congestion level (0-100)
    congestion_level: AtomicU64,
    /// Settlement risk score (0-10000)
    settlement_risk: AtomicU64,
    /// Last block height
    last_block_height: AtomicU64,
    /// Last update time (rdtsc)
    last_update_cycles: AtomicU64,
    /// Is ready
    is_ready: AtomicBool,
}

impl BtcMempoolTracker {
    /// Create a new mempool tracker
    pub const fn new() -> Self {
        // Initialize fee buckets with logarithmic ranges
        const fn init_buckets() -> [FeeBucket; FEE_BUCKETS] {
            let mut buckets = [FeeBucket::new(0, 0); FEE_BUCKETS];
            let mut i = 0;
            while i < FEE_BUCKETS {
                let min = if i == 0 { 1 } else { (1u64 << (i / 4)) };
                let max = min * 2;
                buckets[i] = FeeBucket::new(min, max);
                i += 1;
            }
            buckets
        }
        
        Self {
            entries: [MempoolEntry {
                txid_hash: 0,
                fee_sats: 0,
                vsize_bytes: 0,
                feerate_fixed: 0,
                entry_time_cycles: 0,
                ancestor_count: 0,
                descendant_count: 0,
                is_rbf: false,
                _padding: [0u8; 35],
            }; MAX_MEMPOOL_ENTRIES],
            fee_buckets: init_buckets(),
            entry_count: AtomicU64::new(0),
            total_size_bytes: PaddedAtomicU64::new(0),
            total_fees_sats: PaddedAtomicU64::new(0),
            size_history: MempoolSizeBuffer::new(1440),
            congestion_level: AtomicU64::new(0),
            settlement_risk: AtomicU64::new(0),
            last_block_height: AtomicU64::new(0),
            last_update_cycles: AtomicU64::new(0),
            is_ready: AtomicBool::new(false),
        }
    }
    
    /// Add a transaction to the mempool
    #[inline]
    pub fn add_entry(&self, entry: MempoolEntry) -> bool {
        let idx = self.entry_count.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= MAX_MEMPOOL_ENTRIES {
            self.entry_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            *self.entries.get_unchecked_mut(idx) = entry;
        }
        
        // Update totals (branchless)
        self.total_size_bytes.fetch_add(entry.vsize_bytes as u64);
        self.total_fees_sats.fetch_add(entry.fee_sats);
        
        // Update fee bucket
        self.update_fee_bucket(entry.feerate_fixed, entry.vsize_bytes as u64);
        
        // Record timestamp
        self.last_update_cycles.store(rdtsc(), Ordering::Relaxed);
        
        true
    }
    
    /// Remove a transaction from mempool (e.g., confirmed)
    #[inline]
    pub fn remove_entry(&self, idx: usize) {
        if idx >= MAX_MEMPOOL_ENTRIES { return; }
        
        unsafe {
            let entry = *self.entries.get_unchecked(idx);
            if entry.txid_hash != 0 {
                self.total_size_bytes.fetch_sub(entry.vsize_bytes as u64);
                self.total_fees_sats.fetch_sub(entry.fee_sats);
                
                // Zero out entry
                *self.entries.get_unchecked_mut(idx) = MempoolEntry {
                    txid_hash: 0,
                    fee_sats: 0,
                    vsize_bytes: 0,
                    feerate_fixed: 0,
                    entry_time_cycles: 0,
                    ancestor_count: 0,
                    descendant_count: 0,
                    is_rbf: false,
                    _padding: [0u8; 35],
                };
            }
        }
    }
    
    /// Update fee histogram bucket
    #[inline]
    fn update_fee_bucket(&self, feerate_fixed: u64, size: u64) {
        // Convert feerate to sat/vB
        let feerate_sat_vb = feerate_fixed / FIXED_SCALE;
        
        // Find appropriate bucket (binary search would be ideal, but branchless for speed)
        let bucket_idx = ((feerate_sat_vb as f64).log2() * 4.0) as usize;
        let bucket_idx = bucket_idx.min(FEE_BUCKETS - 1);
        
        unsafe {
            let bucket = self.fee_buckets.get_unchecked(bucket_idx);
            bucket.tx_count.fetch_add(1, Ordering::Relaxed);
            bucket.total_size.fetch_add(size, Ordering::Relaxed);
        }
    }
    
    /// Get estimated feerate for target confirmation blocks
    #[inline]
    pub fn estimate_feerate(&self, target_blocks: u32) -> u64 {
        // Simple percentile-based estimation
        // Higher target_blocks = lower feerate acceptable
        
        let total_entries = self.entry_count.load(Ordering::Relaxed);
        if total_entries == 0 { return MIN_FEERATE_SAT_VB; }
        
        // Calculate percentile based on target
        let percentile = (100 - (target_blocks * 10).min(90)) as u64;
        
        // Find feerate at percentile
        let mut cumulative = 0u64;
        let threshold = (total_entries * percentile) / 100;
        
        for i in 0..FEE_BUCKETS {
            unsafe {
                let bucket = self.fee_buckets.get_unchecked(i);
                cumulative += bucket.tx_count.load(Ordering::Relaxed);
                if cumulative >= threshold {
                    return bucket.min_feerate;
                }
            }
        }
        
        MIN_FEERATE_SAT_VB
    }
    
    /// Update congestion and settlement risk metrics
    #[inline]
    pub fn update_metrics(&self) {
        let total_size = self.total_size_bytes.load();
        let entry_count = self.entry_count.load(Ordering::Relaxed);
        
        // Update rolling history
        self.size_history.push(total_size);
        
        // Congestion calculation (branchless)
        // Based on mempool size relative to typical capacity (~100MB)
        let typical_capacity = 100_000_000; // 100 MB
        let congestion = (total_size * 100 / typical_capacity).min(100);
        self.congestion_level.store(congestion, Ordering::Relaxed);
        
        // Settlement risk calculation
        // Factors: congestion, average wait time, fee pressure
        let avg_size = self.size_history.get_average();
        let size_trend = if total_size > avg_size {
            ((total_size - avg_size) * 100 / avg_size).min(500)
        } else {
            0
        };
        
        let fee_pressure = if entry_count > 0 {
            self.total_fees_sats.load() * 100 / entry_count
        } else {
            0
        };
        
        // Combined risk score (0-10000)
        let risk = (congestion * 50 + size_trend * 30 + (fee_pressure / 1000).min(200)) .min(10000);
        self.settlement_risk.store(risk, Ordering::Relaxed);
        
        self.is_ready.store(true, Ordering::Relaxed);
    }
    
    /// Get time delta since last update in CPU cycles
    #[inline]
    pub fn get_update_delta_cycles(&self) -> u64 {
        let last = self.last_update_cycles.load(Ordering::Relaxed);
        if last == 0 { return 0; }
        rdtsc() - last
    }
    
    /// Get current mempool size in bytes
    #[inline]
    pub fn get_mempool_size(&self) -> u64 {
        self.total_size_bytes.load()
    }
    
    /// Get entry count
    #[inline]
    pub fn get_entry_count(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }
    
    /// Get congestion level (0-100)
    #[inline]
    pub fn get_congestion_level(&self) -> u64 {
        self.congestion_level.load(Ordering::Relaxed)
    }
    
    /// Get settlement risk score (0-10000)
    #[inline]
    pub fn get_settlement_risk(&self) -> u64 {
        self.settlement_risk.load(Ordering::Relaxed)
    }
    
    /// Get rolling average mempool size
    #[inline]
    pub fn get_avg_mempool_size(&self) -> u64 {
        self.size_history.get_average()
    }
    
    /// Update last known block height
    #[inline]
    pub fn update_block_height(&self, height: u64) {
        self.last_block_height.store(height, Ordering::Relaxed);
    }
    
    /// Get last block height
    #[inline]
    pub fn get_block_height(&self) -> u64 {
        self.last_block_height.load(Ordering::Relaxed)
    }
}

impl Default for BtcMempoolTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add_entry() {
        let tracker = BtcMempoolTracker::new();
        
        let entry = MempoolEntry {
            txid_hash: 0x1234567890ABCDEF,
            fee_sats: 1000,
            vsize_bytes: 250,
            feerate_fixed: 4 * FIXED_SCALE, // 4 sat/vB
            entry_time_cycles: rdtsc(),
            ancestor_count: 0,
            descendant_count: 0,
            is_rbf: true,
            _padding: [0u8; 35],
        };
        
        assert!(tracker.add_entry(entry));
        assert_eq!(tracker.get_entry_count(), 1);
        assert_eq!(tracker.get_mempool_size(), 250);
    }
    
    #[test]
    fn test_feerate_estimation() {
        let tracker = BtcMempoolTracker::new();
        
        // Add several entries with different feerates
        for i in 1..=10 {
            let entry = MempoolEntry {
                txid_hash: i as u64,
                fee_sats: i * 100,
                vsize_bytes: 100,
                feerate_fixed: (i * FIXED_SCALE) as u64,
                entry_time_cycles: rdtsc(),
                ancestor_count: 0,
                descendant_count: 0,
                is_rbf: false,
                _padding: [0u8; 35],
            };
            tracker.add_entry(entry);
        }
        
        // Estimate for next block (should be higher)
        let feerate_1 = tracker.estimate_feerate(1);
        
        // Estimate for 6 blocks (should be lower)
        let feerate_6 = tracker.estimate_feerate(6);
        
        assert!(feerate_1 >= feerate_6);
    }
    
    #[test]
    fn test_congestion_calculation() {
        let tracker = BtcMempoolTracker::new();
        
        // Add many entries to increase congestion
        for i in 0..100 {
            let entry = MempoolEntry {
                txid_hash: i,
                fee_sats: 500,
                vsize_bytes: 1_000_000, // 1MB each
                feerate_fixed: 5 * FIXED_SCALE,
                entry_time_cycles: rdtsc(),
                ancestor_count: 0,
                descendant_count: 0,
                is_rbf: false,
                _padding: [0u8; 35],
            };
            tracker.add_entry(entry);
        }
        
        tracker.update_metrics();
        
        let congestion = tracker.get_congestion_level();
        assert!(congestion > 0);
        
        let risk = tracker.get_settlement_risk();
        assert!(risk > 0);
    }
}
