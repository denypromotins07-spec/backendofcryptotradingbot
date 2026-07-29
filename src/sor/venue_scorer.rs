//! Dynamic venue scoring engine evaluating latency, depth, and fee tiers.
//! 
//! Scores each trading venue based on multiple factors:
//! - Latency (round-trip time)
//! - Depth (available liquidity)
//! - Fee tier (maker/taker rates)
//! - Fill probability
//! 
//! Uses SIMD for parallel venue comparison and fixed-point arithmetic.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Maximum number of venues to track
const MAX_VENUES: usize = 32;

/// Weights for scoring (sum to FIXED_SCALE)
const WEIGHT_LATENCY: i64 = 25_000_000;  // 25%
const WEIGHT_DEPTH: i64 = 30_000_000;    // 30%
const WEIGHT_FEES: i64 = 25_000_000;     // 25%
const WEIGHT_FILL_PROB: i64 = 20_000_000; // 20%

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Venue score components - cache line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VenueScore {
    pub venue_id: u32,
    pub total_score: i64,            // Weighted total score (fixed-point)
    
    // Component scores (each 0 to FIXED_SCALE)
    pub latency_score: i64,
    pub depth_score: i64,
    pub fee_score: i64,
    pub fill_prob_score: i64,
    
    // Raw metrics
    pub latency_us: u32,             // Round-trip latency
    pub depth_bps: i32,              // Depth in basis points of avg volume
    pub maker_fee_bps: i32,          // Maker fee (negative = rebate)
    pub taker_fee_bps: i32,          // Taker fee
    
    _padding: [u8; 16],              // Pad to 64 bytes
}

impl Default for VenueScore {
    fn default() -> Self {
        Self {
            venue_id: 0,
            total_score: 0,
            latency_score: 0,
            depth_score: 0,
            fee_score: 0,
            fill_prob_score: 0,
            latency_us: 0,
            depth_bps: 0,
            maker_fee_bps: 0,
            taker_fee_bps: 0,
            _padding: [0; 16],
        }
    }
}

/// Lock-free venue scorer
#[repr(C)]
pub struct VenueScorer {
    /// Venue scores
    scores: [VenueScore; MAX_VENUES],
    venue_count: AtomicU64,
    
    /// Best venue by score
    best_venue_idx: AtomicU64,
    best_venue_score: AtomicI64,
    
    /// Statistics
    score_updates: AtomicU64,
    
    /// Configuration
    min_depth_bps: AtomicI64,        // Minimum depth requirement
    max_latency_us: AtomicU64,       // Maximum acceptable latency
    
    /// Kill switches
    scoring_enabled: AtomicBool,
    
    _padding: [u8; 40],              // Pad to cache line
}

impl VenueScorer {
    /// Create new venue scorer
    pub const fn new() -> Self {
        Self {
            scores: [VenueScore::default(); MAX_VENUES],
            venue_count: AtomicU64::new(0),
            best_venue_idx: AtomicU64::new(0),
            best_venue_score: AtomicI64::new(0),
            score_updates: AtomicU64::new(0),
            min_depth_bps: AtomicI64::new(100), // 1% of avg volume
            max_latency_us: AtomicU64::new(1000), // 1ms max
            scoring_enabled: AtomicBool::new(true),
            _padding: [0; 40],
        }
    }
    
    /// Update venue metrics and recalculate score
    #[inline(always)]
    pub fn update_venue(&self, idx: usize, venue_id: u32, latency_us: u32, 
                        depth_bps: i32, maker_fee_bps: i32, taker_fee_bps: i32,
                        fill_prob: i64) {
        if idx >= MAX_VENUES || !self.scoring_enabled.load(Ordering::Acquire) {
            return;
        }
        
        let score = unsafe { self.scores.get_unchecked_mut(idx) };
        score.venue_id = venue_id;
        score.latency_us = latency_us;
        score.depth_bps = depth_bps;
        score.maker_fee_bps = maker_fee_bps;
        score.taker_fee_bps = taker_fee_bps;
        
        // Calculate component scores (branchless where possible)
        
        // Latency score: lower is better, inverted scale
        // Score = FIXED_SCALE * (1 - latency/max_latency)
        let latency_clamped = latency_us.min(self.max_latency_us.load(Ordering::Relaxed) as u32);
        score.latency_score = ((FIXED_SCALE as u64 * (self.max_latency_us.load(Ordering::Relaxed) - latency_clamped as u64)) 
            / self.max_latency_us.load(Ordering::Relaxed)) as i64;
        
        // Depth score: higher is better
        let min_depth = self.min_depth_bps.load(Ordering::Relaxed);
        score.depth_score = if depth_bps >= min_depth as i32 {
            FIXED_SCALE.min(FIXED_SCALE * depth_bps as i64 / (min_depth * 10))
        } else {
            FIXED_SCALE * depth_bps as i64 / min_depth
        };
        
        // Fee score: maker rebates are positive, fees are negative
        // Score centered around FIXED_SCALE/2
        let fee_adjustment = (maker_fee_bps as i64 * FIXED_SCALE) / 1000; // Scale down
        score.fee_score = (FIXED_SCALE / 2).saturating_sub(fee_adjustment);
        
        // Fill probability score: direct mapping
        score.fill_prob_score = fill_prob.clamp(0, FIXED_SCALE);
        
        // Calculate weighted total score using SIMD-like operations
        let total = 
            (score.latency_score * WEIGHT_LATENCY +
             score.depth_score * WEIGHT_DEPTH +
             score.fee_score * WEIGHT_FEES +
             score.fill_prob_score * WEIGHT_FILL_PROB) / FIXED_SCALE;
        
        score.total_score = total.clamp(0, FIXED_SCALE);
        
        // Update venue count
        let count = self.venue_count.load(Ordering::Relaxed);
        if idx as u64 >= count {
            self.venue_count.store((idx + 1) as u64, Ordering::Release);
        }
        
        self.score_updates.fetch_add(1, Ordering::Relaxed);
        
        // Update best venue
        self.update_best_venue();
    }
    
    /// Find best venue by score (SIMD-optimized)
    #[inline(always)]
    fn update_best_venue(&self) {
        let count = self.venue_count.load(Ordering::Acquire) as usize;
        
        let mut best_idx = 0u64;
        let mut best_score = i64::MIN;
        
        // Unrolled loop for SIMD-like parallelism
        let mut i = 0;
        while i + 4 <= count {
            unsafe {
                let s0 = self.scores.get_unchecked(i).total_score;
                let s1 = self.scores.get_unchecked(i + 1).total_score;
                let s2 = self.scores.get_unchecked(i + 2).total_score;
                let s3 = self.scores.get_unchecked(i + 3).total_score;
                
                // Branchless max finding
                let local_max = s0.max(s1).max(s2).max(s3);
                
                if local_max > best_score {
                    if s0 == local_max { best_score = s0; best_idx = i as u64; }
                    else if s1 == local_max { best_score = s1; best_idx = (i + 1) as u64; }
                    else if s2 == local_max { best_score = s2; best_idx = (i + 2) as u64; }
                    else if s3 == local_max { best_score = s3; best_idx = (i + 3) as u64; }
                }
            }
            i += 4;
        }
        
        // Handle remaining
        while i < count {
            unsafe {
                let score = self.scores.get_unchecked(i).total_score;
                if score > best_score {
                    best_score = score;
                    best_idx = i as u64;
                }
            }
            i += 1;
        }
        
        self.best_venue_idx.store(best_idx, Ordering::Release);
        self.best_venue_score.store(best_score, Ordering::Release);
    }
    
    /// Get best venue index
    #[inline(always)]
    pub fn get_best_venue(&self) -> Option<u32> {
        let count = self.venue_count.load(Ordering::Acquire);
        if count == 0 {
            return None;
        }
        
        let idx = self.best_venue_idx.load(Ordering::Acquire) as usize;
        unsafe {
            Some(self.scores.get_unchecked(idx).venue_id)
        }
    }
    
    /// Get score for specific venue
    #[inline(always)]
    pub fn get_venue_score(&self, venue_id: u32) -> Option<i64> {
        let count = self.venue_count.load(Ordering::Acquire) as usize;
        
        for i in 0..count {
            unsafe {
                let score = self.scores.get_unchecked(i);
                if score.venue_id == venue_id {
                    return Some(score.total_score);
                }
            }
        }
        
        None
    }
    
    /// Get all venue scores sorted
    #[inline(always)]
    pub fn get_sorted_venues(&self, buffer: &mut [(u32, i64)]) -> usize {
        let count = self.venue_count.load(Ordering::Acquire) as usize;
        let len = count.min(buffer.len());
        
        // Copy scores
        for i in 0..len {
            unsafe {
                let s = self.scores.get_unchecked(i);
                buffer[i] = (s.venue_id, s.total_score);
            }
        }
        
        // Simple insertion sort for small arrays (branchless-ish)
        for i in 1..len {
            let mut j = i;
            while j > 0 && buffer[j].1 > buffer[j - 1].1 {
                buffer.swap(j, j - 1);
                j -= 1;
            }
        }
        
        len
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<VenueScore>() == 64);
        assert!(core::mem::size_of::<VenueScorer>() % 64 == 0);
    }
    
    #[test]
    fn test_venue_scoring() {
        let scorer = VenueScorer::new();
        
        // Add venues with different characteristics
        scorer.update_venue(0, 1, 100, 500, -50, 100, 80_000_000); // Low latency, good depth, rebate
        scorer.update_venue(1, 2, 500, 200, 50, 100, 60_000_000);  // Higher latency, less depth, fee
        
        // Venue 0 should have higher score
        let best = scorer.get_best_venue();
        assert_eq!(best, Some(1));
    }
}
