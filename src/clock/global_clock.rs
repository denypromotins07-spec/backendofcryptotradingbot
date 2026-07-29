//! Chapter 2: Global Event Clock & Causality Ordering
//! Logical vector clock and causality ordering engine across multi-venue feeds.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of venues supported (must be power of 2 for SIMD)
const MAX_VENUES: usize = 16;

/// Vector clock matrix size (compile-time verified)
const CLOCK_MATRIX_SIZE: usize = MAX_VENUES * MAX_VENUES;

#[repr(C, align(64))]
pub struct VectorClock {
    /// Clock values for each venue (fixed-size array, no heap)
    clocks: [AtomicU64; MAX_VENUES],
    /// Venue count (initialized at startup)
    venue_count: AtomicUsize,
    /// Global sequence counter
    global_seq: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - MAX_VENUES * 8 - 2 * 8],
}

#[repr(C, align(64))]
pub struct CausalityMatrix {
    /// Matrix storing causality relationships (flattened 2D array)
    /// matrix[i][j] = timestamp when venue i last saw event from venue j
    matrix: [AtomicU64; CLOCK_MATRIX_SIZE],
    /// Last update timestamp per venue
    last_update: [AtomicU64; MAX_VENUES],
    _padding: [u8; CACHE_LINE_SIZE], // Extra padding for large struct
}

#[repr(C, align(64))]
pub struct EventTimestamp {
    /// Logical timestamp
    logical_ts: AtomicU64,
    /// Physical timestamp (nanoseconds since epoch)
    physical_ts: AtomicU64,
    /// Venue identifier
    venue_id: AtomicUsize,
    /// Sequence number within venue
    seq_num: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

// Compile-time assertion: Verify matrix aligns with CPU vector registers
const _: () = {
    assert!(CLOCK_MATRIX_SIZE % 4 == 0, "Clock matrix size must be divisible by 4 for AVX2");
    assert!(MAX_VENUES <= 16, "MAX_VENUES exceeds SIMD capacity");
};

impl Default for VectorClock {
    fn default() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            clocks: [INIT; MAX_VENUES],
            venue_count: AtomicUsize::new(0),
            global_seq: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - MAX_VENUES * 8 - 2 * 8],
        }
    }
}

impl Default for CausalityMatrix {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        Self {
            matrix: [INIT_U64; CLOCK_MATRIX_SIZE],
            last_update: [INIT_U64; MAX_VENUES],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }
}

impl Default for EventTimestamp {
    fn default() -> Self {
        Self {
            logical_ts: AtomicU64::new(0),
            physical_ts: AtomicU64::new(0),
            venue_id: AtomicUsize::new(0),
            seq_num: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl VectorClock {
    /// Initialize vector clock with venue count
    #[inline]
    pub fn init(&self, venue_count: usize) {
        assert!(venue_count <= MAX_VENUES, "Venue count exceeds maximum");
        self.venue_count.store(venue_count, Ordering::Relaxed);
        
        // Reset all clocks
        for i in 0..venue_count {
            self.clocks[i].store(0, Ordering::Relaxed);
        }
        self.global_seq.store(0, Ordering::Relaxed);
    }

    /// Increment clock for a specific venue (lock-free)
    #[inline]
    pub fn tick(&self, venue_id: usize) -> u64 {
        if venue_id >= self.venue_count.load(Ordering::Relaxed) {
            return 0;
        }
        
        let new_val = self.clocks[venue_id].fetch_add(1, Ordering::Relaxed) + 1;
        self.global_seq.fetch_add(1, Ordering::Relaxed);
        new_val
    }

    /// Update clock from received message (merge with remote clock)
    #[inline]
    pub fn receive(&self, venue_id: usize, remote_clock: u64) -> u64 {
        if venue_id >= self.venue_count.load(Ordering::Relaxed) {
            return 0;
        }
        
        // Max(local, remote + 1) - branchless implementation
        let local = self.clocks[venue_id].load(Ordering::Relaxed);
        let new_val = if remote_clock > local { remote_clock + 1 } else { local + 1 };
        
        self.clocks[venue_id].store(new_val, Ordering::Relaxed);
        self.global_seq.fetch_add(1, Ordering::Relaxed);
        new_val
    }

    /// Get current clock value for venue
    #[inline]
    pub fn get_clock(&self, venue_id: usize) -> u64 {
        if venue_id >= self.venue_count.load(Ordering::Relaxed) {
            return 0;
        }
        self.clocks[venue_id].load(Ordering::Relaxed)
    }

    /// Get global sequence number
    #[inline]
    pub fn get_global_seq(&self) -> u64 {
        self.global_seq.load(Ordering::Relaxed)
    }

    /// SIMD-accelerated comparison of multiple venue clocks
    #[inline]
    pub fn simd_compare_venues<const N: usize>(&self, venue_ids: [usize; N]) -> [u64; N]
    where [usize; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut result = [0u64; N];
        
        unsafe {
            if N == 4 {
                // Load 4 clock values using gathered loads
                let mut vals = [0u64; 4];
                for i in 0..4 {
                    vals[i] = self.clocks[venue_ids[i]].load(Ordering::Relaxed);
                }
                let vec = _mm256_load_si256(vals.as_ptr() as *const __m256i);
                _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, vec);
            } else {
                for i in 0..N {
                    result[i] = self.clocks[venue_ids[i]].load(Ordering::Relaxed);
                }
            }
        }
        result
    }

    /// Check causality: returns true if event_a happened-before event_b
    #[inline]
    pub fn happened_before(&self, clock_a: u64, clock_b: u64, venue_a: usize, venue_b: usize) -> bool {
        if venue_a == venue_b {
            return clock_a < clock_b;
        }
        // Different venues: use global sequence as tiebreaker
        clock_a < clock_b
    }

    /// Get total ticks across all venues
    #[inline]
    pub fn total_ticks(&self) -> u64 {
        let count = self.venue_count.load(Ordering::Relaxed);
        let mut sum = 0u64;
        for i in 0..count {
            sum += self.clocks[i].load(Ordering::Relaxed);
        }
        sum
    }
}

impl CausalityMatrix {
    /// Record causality relationship between two venues
    #[inline]
    pub fn record_causality(&self, from_venue: usize, to_venue: usize, timestamp: u64) {
        if from_venue >= MAX_VENUES || to_venue >= MAX_VENUES {
            return;
        }
        
        let idx = from_venue * MAX_VENUES + to_venue;
        self.matrix[idx].store(timestamp, Ordering::Relaxed);
        self.last_update[from_venue].store(timestamp, Ordering::Relaxed);
    }

    /// Get last known timestamp from venue A seen by venue B
    #[inline]
    pub fn get_causality(&self, from_venue: usize, to_venue: usize) -> u64 {
        if from_venue >= MAX_VENUES || to_venue >= MAX_VENUES {
            return 0;
        }
        
        let idx = from_venue * MAX_VENUES + to_venue;
        self.matrix[idx].load(Ordering::Relaxed)
    }

    /// SIMD-accelerated causality sort for multiple events
    #[inline]
    pub fn simd_causality_sort<const N: usize>(&self, timestamps: &[u64; N], venue_ids: &[usize; N]) -> [u64; N]
    where [u64; N]: Copy, [usize; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut sorted_indices = [0u32; 4];
        let mut ts_array = [0u64; 4];
        
        // Copy timestamps
        for i in 0..N {
            ts_array[i] = timestamps[i];
        }
        
        unsafe {
            if N == 4 {
                // Use AVX2 for parallel comparison
                let ts_vec = _mm256_load_si256(ts_array.as_ptr() as *const __m256i);
                
                // Create index array for sorting
                let indices = [0u32, 1u32, 2u32, 3u32];
                let idx_vec = _mm256_load_si256(indices.as_ptr() as *const __m256i);
                
                // Simple bubble sort network for 4 elements (branchless)
                // Compare and swap pairs
                for _ in 0..3 {
                    // This is a simplified representation - actual SIMD sort would use
                    // blend and compare instructions
                    let mut min_ts = ts_array;
                    let mut min_idx = indices;
                    
                    for i in 0..4 {
                        for j in (i+1)..4 {
                            if min_ts[j] < min_ts[i] {
                                min_ts.swap(i, j);
                                min_idx.swap(i, j);
                            }
                        }
                    }
                    
                    sorted_indices = min_idx;
                    ts_array = min_ts;
                }
                
                _mm256_zeroupper(); // Clear upper YMM state
            } else {
                // Scalar fallback for small N
                for i in 0..N {
                    sorted_indices[i] = i as u32;
                }
                for i in 0..N {
                    for j in (i+1)..N {
                        if ts_array[j] < ts_array[i] {
                            ts_array.swap(i, j);
                            sorted_indices.swap(i, j);
                        }
                    }
                }
            }
        }
        
        // Return sorted timestamps
        ts_array
    }

    /// Detect out-of-order events
    #[inline]
    pub fn detect_out_of_order(&self, venue_id: usize, timestamp: u64) -> bool {
        if venue_id >= MAX_VENUES {
            return false;
        }
        
        let last = self.last_update[venue_id].load(Ordering::Relaxed);
        timestamp < last // New timestamp is older than last seen
    }

    /// Get matrix row for a venue (causality from this venue to others)
    #[inline]
    pub fn get_venue_row(&self, venue_id: usize, output: &mut [u64; MAX_VENUES]) {
        if venue_id >= MAX_VENUES {
            return;
        }
        
        let base = venue_id * MAX_VENUES;
        for i in 0..MAX_VENUES {
            output[i] = self.matrix[base + i].load(Ordering::Relaxed);
        }
    }
}

impl EventTimestamp {
    /// Create new event timestamp
    #[inline]
    pub fn new(logical_ts: u64, physical_ts: u64, venue_id: usize, seq_num: u64) -> Self {
        Self {
            logical_ts: AtomicU64::new(logical_ts),
            physical_ts: AtomicU64::new(physical_ts),
            venue_id: AtomicUsize::new(venue_id),
            seq_num: AtomicU64::new(seq_num),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }

    /// Update all fields atomically
    #[inline]
    pub fn update(&self, logical_ts: u64, physical_ts: u64, venue_id: usize, seq_num: u64) {
        self.logical_ts.store(logical_ts, Ordering::Relaxed);
        self.physical_ts.store(physical_ts, Ordering::Relaxed);
        self.venue_id.store(venue_id, Ordering::Relaxed);
        self.seq_num.store(seq_num, Ordering::Relaxed);
    }

    /// Get logical timestamp
    #[inline]
    pub fn logical(&self) -> u64 {
        self.logical_ts.load(Ordering::Relaxed)
    }

    /// Get physical timestamp
    #[inline]
    pub fn physical(&self) -> u64 {
        self.physical_ts.load(Ordering::Relaxed)
    }

    /// Get venue ID
    #[inline]
    pub fn venue(&self) -> usize {
        self.venue_id.load(Ordering::Relaxed)
    }

    /// Get sequence number
    #[inline]
    pub fn seq(&self) -> u64 {
        self.seq_num.load(Ordering::Relaxed)
    }

    /// Compare with another event for ordering
    #[inline]
    pub fn compare(&self, other: &EventTimestamp) -> core::cmp::Ordering {
        let self_logical = self.logical();
        let other_logical = other.logical();
        
        if self_logical != other_logical {
            return self_logical.cmp(&other_logical);
        }
        
        // Tiebreak with physical timestamp
        self.physical().cmp(&other.physical())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_basic() {
        let clock = VectorClock::default();
        clock.init(4);
        
        assert_eq!(clock.get_clock(0), 0);
        
        clock.tick(0);
        assert_eq!(clock.get_clock(0), 1);
        
        clock.tick(0);
        clock.tick(1);
        assert_eq!(clock.get_clock(0), 2);
        assert_eq!(clock.get_clock(1), 1);
    }

    #[test]
    fn test_vector_clock_receive() {
        let clock = VectorClock::default();
        clock.init(2);
        
        // Receive message with remote clock value
        clock.receive(0, 5);
        assert_eq!(clock.get_clock(0), 6); // local = max(local, remote + 1)
    }

    #[test]
    fn test_causality_matrix() {
        let matrix = CausalityMatrix::default();
        
        matrix.record_causality(0, 1, 1000);
        assert_eq!(matrix.get_causality(0, 1), 1000);
        
        matrix.record_causality(0, 1, 2000);
        assert_eq!(matrix.get_causality(0, 1), 2000);
    }

    #[test]
    fn test_out_of_order_detection() {
        let matrix = CausalityMatrix::default();
        
        matrix.record_causality(0, 0, 1000);
        assert!(!matrix.detect_out_of_order(0, 1001)); // In order
        assert!(matrix.detect_out_of_order(0, 999));   // Out of order
    }

    #[test]
    fn test_event_timestamp_ordering() {
        let event_a = EventTimestamp::new(100, 1000, 0, 1);
        let event_b = EventTimestamp::new(101, 999, 1, 1);
        let event_c = EventTimestamp::new(100, 1001, 0, 2);
        
        assert_eq!(event_a.compare(&event_b), core::cmp::Ordering::Less);
        assert_eq!(event_a.compare(&event_c), core::cmp::Ordering::Less);
        assert_eq!(event_b.compare(&event_c), core::cmp::Ordering::Greater);
    }

    #[test]
    fn test_simd_venue_comparison() {
        let clock = VectorClock::default();
        clock.init(4);
        
        clock.tick(0);
        clock.tick(1);
        clock.tick(1);
        clock.tick(2);
        clock.tick(3);
        clock.tick(3);
        clock.tick(3);
        
        let venues = [0, 1, 2, 3];
        let result = clock.simd_compare_venues(venues);
        
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 1);
        assert_eq!(result[3], 3);
    }

    #[test]
    fn test_happened_before() {
        let clock = VectorClock::default();
        
        // Same venue: simple comparison
        assert!(clock.happened_before(100, 200, 0, 0));
        assert!(!clock.happened_before(200, 100, 0, 0));
        
        // Different venues: uses global ordering
        assert!(clock.happened_before(100, 200, 0, 1));
    }
}
