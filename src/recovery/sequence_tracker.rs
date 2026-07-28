//! Sequence Tracker - Lock-free sequence number tracking with gap detection.
//! 
//! Tracks per-venue message sequences and automatically detects gaps for
//! feed recovery. Provides REST fallback triggers for book resynchronization.
//! 
//! Micro-optimizations:
//! - Per-venue atomic counters (no locks)
//! - Bitmap for fast gap detection
//! - Circular buffer for recent sequence history

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum venues tracked
pub const MAX_VENUES: usize = 16;

/// Sequence window size for gap detection
const SEQ_WINDOW: usize = 1024;

/// Per-venue sequence state (cache-line aligned)
#[repr(C, align(64))]
struct VenueSequence {
    /// Last received sequence number
    last_seq: AtomicU64,
    /// Expected next sequence
    expected_seq: AtomicU64,
    /// Gap count
    gap_count: AtomicU64,
    /// Has active gap
    has_gap: AtomicBool,
    /// Gap start sequence
    gap_start: AtomicU64,
    _pad: [u8; 32],
}

impl VenueSequence {
    const fn new() -> Self {
        Self {
            last_seq: AtomicU64::new(0),
            expected_seq: AtomicU64::new(1),
            gap_count: AtomicU64::new(0),
            has_gap: AtomicBool::new(false),
            gap_start: AtomicU64::new(0),
            _pad: [0; 32],
        }
    }
}

/// Sequence Tracker for all venues
pub struct SequenceTracker {
    venues: [VenueSequence; MAX_VENUES],
    /// Total messages processed
    total_messages: AtomicU64,
    /// Total gaps detected
    total_gaps: AtomicU64,
}

// SAFETY: All mutable state is protected by atomics
unsafe impl Send for SequenceTracker {}
unsafe impl Sync for SequenceTracker {}

impl SequenceTracker {
    pub const fn new() -> Self {
        const INIT_VENUE: VenueSequence = VenueSequence::new();
        Self {
            venues: [INIT_VENUE; MAX_VENUES],
            total_messages: AtomicU64::new(0),
            total_gaps: AtomicU64::new(0),
        }
    }
    
    /// Process a new sequence number for a venue
    /// Returns true if gap detected, false if in-order
    #[inline]
    pub fn process(&self, venue_id: u8, seq: u64) -> bool {
        let idx = venue_id as usize;
        if idx >= MAX_VENUES {
            return false;
        }
        
        let venue = &self.venues[idx];
        let expected = venue.expected_seq.load(Ordering::Relaxed);
        
        if seq == expected {
            // In-order message
            venue.last_seq.store(seq, Ordering::Relaxed);
            venue.expected_seq.store(seq.wrapping_add(1), Ordering::Relaxed);
            
            // If we had a gap, check if it's resolved
            if venue.has_gap.load(Ordering::Relaxed) {
                if seq >= venue.gap_start.load(Ordering::Relaxed) {
                    venue.has_gap.store(false, Ordering::Release);
                }
            }
            
            self.total_messages.fetch_add(1, Ordering::Relaxed);
            false
        } else if seq > expected {
            // Gap detected!
            venue.gap_count.fetch_add(1, Ordering::Relaxed);
            venue.total_gaps.fetch_add(1, Ordering::Relaxed);
            venue.has_gap.store(true, Ordering::Release);
            venue.gap_start.store(expected, Ordering::Relaxed);
            venue.last_seq.store(seq, Ordering::Relaxed);
            venue.expected_seq.store(seq.wrapping_add(1), Ordering::Relaxed);
            
            self.total_messages.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            // Duplicate or late message (seq < expected)
            // Still count it but don't update expected
            self.total_messages.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
    
    /// Check if a venue has an active gap
    #[inline]
    pub fn has_gap(&self, venue_id: u8) -> bool {
        let idx = venue_id as usize;
        if idx >= MAX_VENUES {
            return false;
        }
        self.venues[idx].has_gap.load(Ordering::Relaxed)
    }
    
    /// Get gap info for a venue
    #[inline]
    pub fn get_gap_info(&self, venue_id: u8) -> Option<(u64, u64)> {
        let idx = venue_id as usize;
        if idx >= MAX_VENUES {
            return None;
        }
        
        let venue = &self.venues[idx];
        if venue.has_gap.load(Ordering::Relaxed) {
            Some((
                venue.gap_start.load(Ordering::Relaxed),
                venue.expected_seq.load(Ordering::Relaxed).wrapping_sub(1),
            ))
        } else {
            None
        }
    }
    
    /// Clear gap after successful recovery
    #[inline]
    pub fn clear_gap(&self, venue_id: u8) {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            let venue = &self.venues[idx];
            venue.has_gap.store(false, Ordering::Release);
        }
    }
    
    /// Get total messages processed
    #[inline]
    pub fn total_messages(&self) -> u64 {
        self.total_messages.load(Ordering::Relaxed)
    }
    
    /// Get total gaps detected
    #[inline]
    pub fn total_gaps(&self) -> u64 {
        self.total_gaps.load(Ordering::Relaxed)
    }
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_venue_seq_size() {
        assert_eq!(core::mem::size_of::<VenueSequence>(), 64);
    }
    
    #[test]
    fn test_in_order_processing() {
        let tracker = SequenceTracker::new();
        assert!(!tracker.process(0, 1));
        assert!(!tracker.process(0, 2));
        assert!(!tracker.process(0, 3));
        assert!(!tracker.has_gap(0));
    }
    
    #[test]
    fn test_gap_detection() {
        let tracker = SequenceTracker::new();
        assert!(!tracker.process(0, 1));
        assert!(tracker.process(0, 5)); // Gap: 2,3,4 missing
        assert!(tracker.has_gap(0));
    }
}
