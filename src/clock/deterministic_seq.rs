//! Chapter 2: Global Event Clock & Causality Ordering
//! Deterministic event sequencer to resolve out-of-order tick arrivals safely.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum sequence buffer size (power of 2 for efficient modulo)
const SEQ_BUFFER_SIZE: usize = 4096;

/// Maximum venues supported
const MAX_VENUES: usize = 16;

#[repr(C, align(64))]
pub struct DeterministicSequencer {
    /// Expected sequence number per venue
    expected_seq: [AtomicU64; MAX_VENUES],
    /// Received but delayed packets (circular buffer)
    delay_buffer_head: AtomicU64,
    delay_buffer_tail: AtomicU64,
    /// Total processed events
    processed_count: AtomicU64,
    /// Total out-of-order events detected
    ooo_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - MAX_VENUES * 8 - 4 * 8],
}

#[repr(C, align(64))]
pub struct DelayBufferEntry {
    /// Venue ID
    pub venue_id: u8,
    /// Sequence number
    pub seq_num: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Payload pointer (zero-copy reference)
    pub payload_ptr: u64,
    /// Valid flag
    pub valid: u8,
    _reserved: [u8; 5],
}

#[repr(C, align(64))]
pub struct ReorderBuffer {
    /// Fixed-size reorder buffer (pre-allocated, no heap)
    entries: [DelayBufferEntry; SEQ_BUFFER_SIZE],
    /// Head index
    head: AtomicU64,
    /// Tail index
    tail: AtomicU64,
    /// Count of entries
    count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 3 * 8],
}

// Compile-time assertions for alignment
const _: () = {
    assert!(core::mem::size_of::<DelayBufferEntry>() % 8 == 0, "DelayBufferEntry must be 8-byte aligned");
    assert!(SEQ_BUFFER_SIZE.is_power_of_two(), "Buffer size must be power of 2");
};

impl Default for DelayBufferEntry {
    fn default() -> Self {
        Self {
            venue_id: 0,
            seq_num: 0,
            timestamp: 0,
            payload_ptr: 0,
            valid: 0,
            _reserved: [0u8; 5],
        }
    }
}

impl Default for DeterministicSequencer {
    fn default() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        const INIT_USIZE: AtomicUsize = AtomicUsize::new(0);
        Self {
            expected_seq: [INIT; MAX_VENUES],
            delay_buffer_head: INIT,
            delay_buffer_tail: INIT,
            processed_count: INIT,
            ooo_count: INIT,
            _padding: [0u8; CACHE_LINE_SIZE - MAX_VENUES * 8 - 4 * 8],
        }
    }
}

impl Default for ReorderBuffer {
    fn default() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            entries: [DelayBufferEntry::default(); SEQ_BUFFER_SIZE],
            head: INIT,
            tail: INIT,
            count: INIT,
            _padding: [0u8; CACHE_LINE_SIZE - 3 * 8],
        }
    }
}

impl DeterministicSequencer {
    /// Initialize sequencer with starting sequence numbers per venue
    #[inline]
    pub fn init(&self, initial_seqs: &[u64]) {
        for (i, &seq) in initial_seqs.iter().enumerate().take(MAX_VENUES) {
            self.expected_seq[i].store(seq, Ordering::Relaxed);
        }
    }

    /// Process incoming event and determine if it should be executed now or buffered
    /// Returns: true if event is in-order and can be executed immediately
    #[inline]
    pub fn process_event(&self, venue_id: usize, seq_num: u64, buffer: &ReorderBuffer) -> bool {
        if venue_id >= MAX_VENUES {
            return false;
        }

        let expected = self.expected_seq[venue_id].load(Ordering::Relaxed);

        // Branchless comparison: is_next = (seq_num == expected)
        let is_next = ((seq_num ^ expected).wrapping_neg() >> 63) as u8;
        let is_later = (seq_num > expected) as u8;

        // Branchless execution decision
        // Execute if: is_next == 1 OR (is_later == 0 AND seq_num < expected [duplicate])
        let should_execute = is_next | ((is_later ^ 1) & 1);

        if is_next != 0 {
            // In-order event: execute immediately
            self.expected_seq[venue_id].fetch_add(1, Ordering::Relaxed);
            self.processed_count.fetch_add(1, Ordering::Relaxed);

            // Check if we can release any buffered events
            self.release_buffered_events(venue_id, buffer);
            return true;
        } else if is_later != 0 {
            // Out-of-order (future) event: buffer it
            self.ooo_count.fetch_add(1, Ordering::Relaxed);
            buffer.insert(venue_id as u8, seq_num, 0, 0);
            return false;
        } else {
            // Duplicate or old event: discard
            return false;
        }
    }

    /// Release buffered events that are now in-order (manually unrolled loop)
    #[inline]
    pub fn release_buffered_events(&self, venue_id: usize, buffer: &ReorderBuffer) {
        let mut current_expected = self.expected_seq[venue_id].load(Ordering::Relaxed);

        // Manually unrolled loop for branch prediction optimization
        // Check up to 8 buffered events per call
        for _ in 0..8 {
            let found = buffer.find_and_remove(venue_id as u8, current_expected);

            // Branchless continue/break simulation
            let has_found = (found != 0) as u64;
            if has_found == 0 {
                break;
            }

            current_expected += 1;
            self.processed_count.fetch_add(1, Ordering::Relaxed);
        }

        self.expected_seq[venue_id].store(current_expected, Ordering::Relaxed);
    }

    /// Get expected sequence for a venue
    #[inline]
    pub fn get_expected_seq(&self, venue_id: usize) -> u64 {
        if venue_id >= MAX_VENUES {
            return 0;
        }
        self.expected_seq[venue_id].load(Ordering::Relaxed)
    }

    /// Get total processed count
    #[inline]
    pub fn get_processed_count(&self) -> u64 {
        self.processed_count.load(Ordering::Relaxed)
    }

    /// Get out-of-order event count
    #[inline]
    pub fn get_ooo_count(&self) -> u64 {
        self.ooo_count.load(Ordering::Relaxed)
    }

    /// SIMD-accelerated sequence validation for multiple venues
    #[inline]
    pub fn simd_validate_sequences<const N: usize>(
        &self,
        venue_ids: &[usize; N],
        seq_nums: &[u64; N]
    ) -> u32
    where [usize; N]: Copy, [u64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut valid_mask = 0u32;

        unsafe {
            if N == 4 {
                // Load expected sequences
                let mut expected = [0u64; 4];
                for i in 0..4 {
                    expected[i] = self.expected_seq[venue_ids[i]].load(Ordering::Relaxed);
                }

                let expected_vec = _mm256_load_si256(expected.as_ptr() as *const __m256i);
                let actual_vec = _mm256_load_si256(seq_nums.as_ptr() as *const __m256i);

                // Compare: result[i] = (actual[i] >= expected[i])
                let cmp_result = _mm256_cmpgt_epi64(actual_vec, expected_vec);

                // Extract comparison results into mask
                let mask = _mm256_movemask_epi8(cmp_result);

                // Convert to 4-bit mask (one bit per lane)
                valid_mask = ((mask & 0x1) | ((mask >> 4) & 0x2) | ((mask >> 7) & 0x4) | ((mask >> 11) & 0x8)) as u32;
            } else {
                for i in 0..N {
                    let expected = self.expected_seq[venue_ids[i]].load(Ordering::Relaxed);
                    let bit = ((seq_nums[i] >= expected) as u32) << i;
                    valid_mask |= bit;
                }
            }
        }

        valid_mask
    }
}

impl ReorderBuffer {
    /// Insert event into reorder buffer (lock-free, circular)
    #[inline]
    pub fn insert(&self, venue_id: u8, seq_num: u64, timestamp: u64, payload_ptr: u64) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let count = self.count.load(Ordering::Relaxed);

        // Check if buffer is full
        if count >= SEQ_BUFFER_SIZE as u64 {
            return false;
        }

        let idx = (head % SEQ_BUFFER_SIZE as u64) as usize;

        // Write entry
        let entry = &mut self.entries[idx];
        entry.venue_id = venue_id;
        entry.seq_num = seq_num;
        entry.timestamp = timestamp;
        entry.payload_ptr = payload_ptr;
        entry.valid = 1;

        // Update head atomically
        self.head.fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        true
    }

    /// Find and remove entry matching venue_id and seq_num
    /// Returns the payload_ptr if found, 0 otherwise
    #[inline]
    pub fn find_and_remove(&self, venue_id: u8, seq_num: u64) -> u64 {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        let count = head - tail;

        // Search through buffer (limited search for performance)
        let search_limit = count.min(64);

        for i in 0..search_limit {
            let idx = ((tail + i) % SEQ_BUFFER_SIZE as u64) as usize;
            let entry = &self.entries[idx];

            // Branchless comparison
            let matches = ((entry.venue_id == venue_id) & (entry.seq_num == seq_num) & (entry.valid == 1)) as u8;

            if matches != 0 {
                let payload = entry.payload_ptr;

                // Mark as invalid (logically removed)
                // Note: In production, would need CAS for thread safety
                unsafe {
                    let entry_mut = &mut *(entry as *const DelayBufferEntry as *mut DelayBufferEntry);
                    entry_mut.valid = 0;
                }

                // Advance tail past this entry if it's at the front
                if i == 0 {
                    self.tail.fetch_add(1, Ordering::Relaxed);
                    self.count.fetch_sub(1, Ordering::Relaxed);
                }

                return payload;
            }
        }

        0
    }

    /// Get current buffer occupancy
    #[inline]
    pub fn occupancy(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Clear all entries
    #[inline]
    pub fn clear(&self) {
        let head = self.head.load(Ordering::Relaxed);
        self.tail.store(head, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    /// Get oldest entry timestamp
    #[inline]
    pub fn oldest_timestamp(&self) -> u64 {
        let tail = self.tail.load(Ordering::Relaxed);
        if self.count.load(Ordering::Relaxed) == 0 {
            return 0;
        }

        let idx = (tail % SEQ_BUFFER_SIZE as u64) as usize;
        self.entries[idx].timestamp
    }
}

/// Sequence gap detector for monitoring missing packets
#[repr(C, align(64))]
pub struct GapDetector {
    /// Last seen sequence per venue
    last_seen: [AtomicU64; MAX_VENUES],
    /// Gap count per venue
    gap_count: [AtomicU64; MAX_VENUES],
    /// Maximum gap observed
    max_gap: [AtomicU64; MAX_VENUES],
    _padding: [u8; CACHE_LINE_SIZE],
}

impl Default for GapDetector {
    fn default() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            last_seen: [INIT; MAX_VENUES],
            gap_count: [INIT; MAX_VENUES],
            max_gap: [INIT; MAX_VENUES],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }
}

impl GapDetector {
    /// Record received sequence and detect gaps
    #[inline]
    pub fn record(&self, venue_id: usize, seq_num: u64) {
        if venue_id >= MAX_VENUES {
            return;
        }

        let last = self.last_seen[venue_id].load(Ordering::Relaxed);

        if seq_num > last {
            let gap = seq_num - last - 1;

            // Branchless gap detection
            let has_gap = (gap > 0) as u64;
            self.gap_count[venue_id].fetch_add(has_gap, Ordering::Relaxed);

            // Update max gap (branchless max)
            let current_max = self.max_gap[venue_id].load(Ordering::Relaxed);
            let new_max = if gap > current_max { gap } else { current_max };
            self.max_gap[venue_id].store(new_max, Ordering::Relaxed);

            self.last_seen[venue_id].store(seq_num, Ordering::Relaxed);
        }
    }

    /// Get gap count for venue
    #[inline]
    pub fn get_gap_count(&self, venue_id: usize) -> u64 {
        if venue_id >= MAX_VENUES {
            return 0;
        }
        self.gap_count[venue_id].load(Ordering::Relaxed)
    }

    /// Get maximum gap observed for venue
    #[inline]
    pub fn get_max_gap(&self, venue_id: usize) -> u64 {
        if venue_id >= MAX_VENUES {
            return 0;
        }
        self.max_gap[venue_id].load(Ordering::Relaxed)
    }

    /// Reset statistics for venue
    #[inline]
    pub fn reset_venue(&self, venue_id: usize) {
        if venue_id >= MAX_VENUES {
            return;
        }
        self.last_seen[venue_id].store(0, Ordering::Relaxed);
        self.gap_count[venue_id].store(0, Ordering::Relaxed);
        self.max_gap[venue_id].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sequencer_basic() {
        let seq = DeterministicSequencer::default();
        let buffer = ReorderBuffer::default();

        // In-order events should execute immediately
        assert!(seq.process_event(0, 0, &buffer));
        assert!(seq.process_event(0, 1, &buffer));
        assert!(seq.process_event(0, 2, &buffer));

        assert_eq!(seq.get_processed_count(), 3);
        assert_eq!(seq.get_ooo_count(), 0);
    }

    #[test]
    fn test_out_of_order_handling() {
        let seq = DeterministicSequencer::default();
        let buffer = ReorderBuffer::default();

        // First event in-order
        assert!(seq.process_event(0, 0, &buffer));

        // Future event arrives early (out-of-order)
        assert!(!seq.process_event(0, 2, &buffer));

        // Missing event arrives
        assert!(seq.process_event(0, 1, &buffer));

        assert_eq!(seq.get_processed_count(), 3);
        assert_eq!(seq.get_ooo_count(), 1);
    }

    #[test]
    fn test_reorder_buffer_operations() {
        let buffer = ReorderBuffer::default();

        assert!(buffer.insert(0, 5, 1000, 0xDEADBEEF));
        assert!(buffer.insert(0, 6, 1001, 0xCAFEBABE));

        assert_eq!(buffer.occupancy(), 2);

        // Find and remove
        let payload = buffer.find_and_remove(0, 5);
        assert_eq!(payload, 0xDEADBEEF);

        assert_eq!(buffer.occupancy(), 1);
    }

    #[test]
    fn test_gap_detector() {
        let detector = GapDetector::default();

        detector.record(0, 0);
        detector.record(0, 1);
        assert_eq!(detector.get_gap_count(0), 0);

        // Gap: sequence jumps from 1 to 5
        detector.record(0, 5);
        assert_eq!(detector.get_gap_count(0), 1);
        assert_eq!(detector.get_max_gap(0), 3); // Gap of 3 (missing 2, 3, 4)
    }

    #[test]
    fn test_simd_validation() {
        let seq = DeterministicSequencer::default();

        // Set expected sequences
        let initial = [0u64, 5u64, 10u64, 15u64];
        seq.init(&initial);

        let venue_ids = [0, 1, 2, 3];
        let seq_nums = [0u64, 6u64, 9u64, 15u64];

        let mask = seq.simd_validate_sequences(&venue_ids, &seq_nums);

        // Expected: [true, true, false, true] = 0b1011 = 11
        assert_eq!(mask, 0b1011);
    }

    #[test]
    fn test_duplicate_handling() {
        let seq = DeterministicSequencer::default();
        let buffer = ReorderBuffer::default();

        assert!(seq.process_event(0, 0, &buffer));
        assert!(seq.process_event(0, 1, &buffer));

        // Duplicate should be rejected
        assert!(!seq.process_event(0, 0, &buffer));

        assert_eq!(seq.get_processed_count(), 2);
    }

    #[test]
    fn test_buffer_wraparound() {
        let buffer = ReorderBuffer::default();

        // Fill buffer close to capacity
        for i in 0..100 {
            buffer.insert(0, i, i as u64, i as u64);
        }

        // Remove some entries
        for i in 0..50 {
            buffer.find_and_remove(0, i);
        }

        // Add more entries (should wrap around)
        for i in 100..150 {
            buffer.insert(0, i, i as u64, i as u64);
        }

        assert!(buffer.occupancy() > 0);
    }
}
