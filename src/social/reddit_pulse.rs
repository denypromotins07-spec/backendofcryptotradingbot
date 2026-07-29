//! Chapter 4: Streaming Social Sentiment & Explainable AI (XAI)
//! Reddit API streaming aggregator for subreddit sentiment spikes using SIMD.

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum subreddits tracked
const MAX_SUBREDDITS: usize = 64;

/// Rolling window size for sentiment calculation
const SENTIMENT_WINDOW: usize = 256;

/// Post buffer capacity (pre-allocated)
const POST_BUFFER_SIZE: usize = 512;

#[repr(C, align(64))]
pub struct SubredditTracker {
    /// Subreddit name hashes
    subreddit_hashes: [AtomicU64; MAX_SUBREDDITS],
    /// Post count in current window
    post_count: [AtomicU64; MAX_SUBREDDITS],
    /// Upvote ratio accumulator (scaled by 1e9)
    upvote_ratio_sum: [AtomicI64; MAX_SUBREDDITS],
    /// Comment velocity (posts per minute)
    comment_velocity: [AtomicU64; MAX_SUBREDDITS],
    /// Sentiment spike flag
    spike_active: [AtomicBool; MAX_SUBREDDITS],
    _padding: [u8; CACHE_LINE_SIZE - MAX_SUBREDDITS * (8 + 8 + 8 + 8 + 1)],
}

#[repr(C, align(64))]
pub struct SentimentRollingBuffer {
    /// Circular buffer for sentiment values
    sentiments: [i64; SENTIMENT_WINDOW],
    /// Head index
    head: AtomicU64,
    /// Running sum for O(1) average
    running_sum: AtomicI64,
    /// Running squared sum for variance
    running_sq_sum: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - SENTIMENT_WINDOW * 8 - 3 * 8],
}

#[repr(C, align(64))]
pub struct PostAggregator {
    /// Pre-allocated post storage
    post_hashes: [AtomicU64; POST_BUFFER_SIZE],
    post_scores: [AtomicI64; POST_BUFFER_SIZE],
    post_timestamps: [AtomicU64; POST_BUFFER_SIZE],
    post_subreddit_idx: [AtomicU64; POST_BUFFER_SIZE],
    /// Head/Tail for circular buffer
    head: AtomicU64,
    tail: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

// Compile-time assertions
const _: () = {
    assert!(MAX_SUBREDDITS <= 64, "MAX_SUBREDDITS exceeds limit");
    assert!(SENTIMENT_WINDOW.is_power_of_two(), "Window size must be power of 2");
};

impl Default for SubredditTracker {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        const INIT_BOOL: AtomicBool = AtomicBool::new(false);
        
        Self {
            subreddit_hashes: [INIT_U64; MAX_SUBREDDITS],
            post_count: [INIT_U64; MAX_SUBREDDITS],
            upvote_ratio_sum: [INIT_I64; MAX_SUBREDDITS],
            comment_velocity: [INIT_U64; MAX_SUBREDDITS],
            spike_active: [INIT_BOOL; MAX_SUBREDDITS],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_SUBREDDITS * (8 + 8 + 8 + 8 + 1)],
        }
    }
}

impl Default for SentimentRollingBuffer {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            sentiments: [0i64; SENTIMENT_WINDOW],
            head: INIT_U64,
            running_sum: INIT_I64,
            running_sq_sum: INIT_I64,
            _padding: [0u8; CACHE_LINE_SIZE - SENTIMENT_WINDOW * 8 - 3 * 8],
        }
    }
}

impl Default for PostAggregator {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            post_hashes: [INIT_U64; POST_BUFFER_SIZE],
            post_scores: [INIT_I64; POST_BUFFER_SIZE],
            post_timestamps: [INIT_U64; POST_BUFFER_SIZE],
            post_subreddit_idx: [INIT_U64; POST_BUFFER_SIZE],
            head: INIT_U64,
            tail: INIT_U64,
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl SubredditTracker {
    /// Register a subreddit
    #[inline]
    pub fn register_subreddit(&self, idx: usize, hash: u64) {
        if idx >= MAX_SUBREDDITS {
            return;
        }
        
        self.subreddit_hashes[idx].store(hash, Ordering::Relaxed);
        self.post_count[idx].store(0, Ordering::Relaxed);
        self.upvote_ratio_sum[idx].store(0, Ordering::Relaxed);
        self.comment_velocity[idx].store(0, Ordering::Relaxed);
        self.spike_active[idx].store(false, Ordering::Relaxed);
    }

    /// Record a post (zero-allocation)
    #[inline]
    pub fn record_post(&self, subreddit_idx: usize, upvote_ratio: i64) {
        if subreddit_idx >= MAX_SUBREDDITS {
            return;
        }
        
        self.post_count[subreddit_idx].fetch_add(1, Ordering::Relaxed);
        self.upvote_ratio_sum[subreddit_idx].fetch_add(upvote_ratio, Ordering::Relaxed);
        
        // Update velocity (simplified EMA)
        let current_vel = self.comment_velocity[subreddit_idx].load(Ordering::Relaxed);
        let new_vel = (current_vel * 7 + 1000) / 8; // Add ~1 post/min equivalent
        self.comment_velocity[subreddit_idx].store(new_vel, Ordering::Relaxed);
    }

    /// Check for sentiment spike (branchless)
    #[inline]
    pub fn check_spike(&self, subreddit_idx: usize, threshold_posts: u64) -> bool {
        if subreddit_idx >= MAX_SUBREDDITS {
            return false;
        }
        
        let count = self.post_count[subreddit_idx].load(Ordering::Relaxed);
        let is_spike = count >= threshold_posts;
        
        self.spike_active[subreddit_idx].store(is_spike, Ordering::Relaxed);
        is_spike
    }

    /// Get average upvote ratio
    #[inline]
    pub fn get_avg_upvote_ratio(&self, subreddit_idx: usize) -> i64 {
        if subreddit_idx >= MAX_SUBREDDITS {
            return 0;
        }
        
        let count = self.post_count[subreddit_idx].load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }
        
        let sum = self.upvote_ratio_sum[subreddit_idx].load(Ordering::Relaxed);
        sum / count as i64
    }

    /// Get comment velocity
    #[inline]
    pub fn get_comment_velocity(&self, subreddit_idx: usize) -> u64 {
        if subreddit_idx >= MAX_SUBREDDITS {
            return 0;
        }
        self.comment_velocity[subreddit_idx].load(Ordering::Relaxed)
    }

    /// Reset subreddit counters
    #[inline]
    pub fn reset_subreddit(&self, subreddit_idx: usize) {
        if subreddit_idx >= MAX_SUBREDDITS {
            return;
        }
        
        self.post_count[subreddit_idx].store(0, Ordering::Relaxed);
        self.upvote_ratio_sum[subreddit_idx].store(0, Ordering::Relaxed);
        self.comment_velocity[subreddit_idx].store(0, Ordering::Relaxed);
        self.spike_active[subreddit_idx].store(false, Ordering::Relaxed);
    }

    /// SIMD-accelerated multi-subreddit spike check
    #[inline]
    pub fn simd_check_spikes<const N: usize>(&self, indices: [usize; N], thresholds: [u64; N]) -> u32
    where [usize; N]: Copy, [u64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut spike_mask = 0u32;
        
        unsafe {
            if N == 4 {
                let mut counts = [0u64; 4];
                for i in 0..4 {
                    counts[i] = self.post_count[indices[i]].load(Ordering::Relaxed);
                }
                
                let count_vec = _mm256_load_si256(counts.as_ptr() as *const __m256i);
                let thresh_vec = _mm256_load_si256(thresholds.as_ptr() as *const __m256i);
                
                // Compare: count >= threshold
                let cmp_vec = _mm256_cmpgt_epi64(count_vec, _mm256_sub_epi64(thresh_vec, _mm256_set1_epi64x(1)));
                
                let mask = _mm256_movemask_epi8(cmp_vec);
                spike_mask = ((mask & 0x1) | ((mask >> 4) & 0x2) | ((mask >> 7) & 0x4) | ((mask >> 11) & 0x8)) as u32;
                
                _mm256_zeroupper();
            } else {
                for i in 0..N {
                    let count = self.post_count[indices[i]].load(Ordering::Relaxed);
                    let bit = ((count >= thresholds[i]) as u32) << i;
                    spike_mask |= bit;
                }
            }
        }
        
        spike_mask
    }
}

impl SentimentRollingBuffer {
    /// Push sentiment value with O(1) sum update
    #[inline]
    pub fn push(&self, sentiment: i64) {
        let head = self.head.load(Ordering::Relaxed);
        let idx = (head % SENTIMENT_WINDOW as u64) as usize;
        
        // Get old value for sum adjustment
        let old_val = unsafe { *self.sentiments.get_unchecked(idx) };
        
        // Update running sums
        let old_sum = self.running_sum.load(Ordering::Relaxed);
        self.running_sum.store(old_sum - old_val + sentiment, Ordering::Relaxed);
        
        let old_sq = old_val * old_val;
        let new_sq = sentiment * sentiment;
        let old_sq_sum = self.running_sq_sum.load(Ordering::Relaxed);
        self.running_sq_sum.store(old_sq_sum - old_sq + new_sq, Ordering::Relaxed);
        
        // Store new value
        unsafe {
            *self.sentiments.get_unchecked_mut(idx) = sentiment;
        }
        
        self.head.fetch_add(1, Ordering::Relaxed);
    }

    /// Get rolling average sentiment
    #[inline]
    pub fn rolling_average(&self, count: u64) -> i64 {
        let actual_count = count.min(SENTIMENT_WINDOW as u64);
        if actual_count == 0 {
            return 0;
        }
        
        self.running_sum.load(Ordering::Relaxed) / actual_count as i64
    }

    /// Get sentiment volatility (standard deviation approximation)
    #[inline]
    pub fn sentiment_volatility(&self, count: u64) -> u64 {
        let actual_count = count.min(SENTIMENT_WINDOW as u64);
        if actual_count < 2 {
            return 0;
        }
        
        let sum = self.running_sum.load(Ordering::Relaxed);
        let sq_sum = self.running_sq_sum.load(Ordering::Relaxed);
        
        // Variance = E[X^2] - E[X]^2
        let mean = sum / actual_count as i64;
        let mean_sq = mean * mean;
        let variance = (sq_sum / actual_count as i64 - mean_sq).max(0) as u64;
        
        // Approximate sqrt using Newton-Raphson
        if variance == 0 {
            return 0;
        }
        
        let mut guess = variance / 2;
        guess = (guess + variance / guess) / 2;
        guess = (guess + variance / guess) / 2;
        
        guess
    }

    /// Get sentiment trend (positive = improving, negative = worsening)
    #[inline]
    pub fn sentiment_trend(&self) -> i64 {
        let head = self.head.load(Ordering::Relaxed);
        if head < 2 {
            return 0;
        }
        
        let idx1 = ((head - 1) % SENTIMENT_WINDOW as u64) as usize;
        let idx2 = ((head - 2) % SENTIMENT_WINDOW as u64) as usize;
        
        unsafe {
            let current = *self.sentiments.get_unchecked(idx1);
            let previous = *self.sentiments.get_unchecked(idx2);
            current - previous
        }
    }
}

impl PostAggregator {
    /// Add post to aggregator
    #[inline]
    pub fn add_post(&self, hash: u64, score: i64, timestamp: u64, subreddit_idx: u64) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if head.wrapping_sub(tail) >= POST_BUFFER_SIZE as u64 {
            return false;
        }
        
        let idx = (head % POST_BUFFER_SIZE as u64) as usize;
        
        self.post_hashes[idx].store(hash, Ordering::Relaxed);
        self.post_scores[idx].store(score, Ordering::Relaxed);
        self.post_timestamps[idx].store(timestamp, Ordering::Relaxed);
        self.post_subreddit_idx[idx].store(subreddit_idx, Ordering::Relaxed);
        
        self.head.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Remove oldest post
    #[inline]
    pub fn remove_oldest(&self) -> Option<(u64, i64, u64, u64)> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        if tail >= head {
            return None;
        }
        
        let idx = (tail % POST_BUFFER_SIZE as u64) as usize;
        
        let hash = self.post_hashes[idx].load(Ordering::Relaxed);
        let score = self.post_scores[idx].load(Ordering::Relaxed);
        let ts = self.post_timestamps[idx].load(Ordering::Relaxed);
        let sr = self.post_subreddit_idx[idx].load(Ordering::Relaxed);
        
        self.tail.fetch_add(1, Ordering::Relaxed);
        
        Some((hash, score, ts, sr))
    }

    /// Get posts by subreddit
    #[inline]
    pub fn get_posts_by_subreddit(&self, subreddit_idx: u64, output: &mut [(u64, i64); 16]) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        let count = head.wrapping_sub(tail);
        
        let mut found = 0;
        for i in 0..count.min(POST_BUFFER_SIZE as u64) {
            let idx = ((tail + i) % POST_BUFFER_SIZE as u64) as usize;
            let sr = self.post_subreddit_idx[idx].load(Ordering::Relaxed);
            
            if sr == subreddit_idx && found < 16 {
                let hash = self.post_hashes[idx].load(Ordering::Relaxed);
                let score = self.post_scores[idx].load(Ordering::Relaxed);
                output[found] = (hash, score);
                found += 1;
            }
        }
        
        found
    }

    /// Get occupancy
    #[inline]
    pub fn occupancy(&self) -> u64 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subreddit_tracking() {
        let tracker = SubredditTracker::default();
        tracker.register_subreddit(0, 0xCRYPTO_HASH);
        
        // Record some posts
        tracker.record_post(0, 800_000_000); // 80% upvote ratio
        tracker.record_post(0, 900_000_000); // 90% upvote ratio
        
        assert_eq!(tracker.post_count[0].load(Ordering::Relaxed), 2);
        assert_eq!(tracker.get_avg_upvote_ratio(0), 850_000_000);
    }

    #[test]
    fn test_spike_detection() {
        let tracker = SubredditTracker::default();
        tracker.register_subreddit(0, 0xWALLSTREET_HASH);
        
        for _ in 0..99 {
            tracker.record_post(0, 700_000_000);
        }
        assert!(!tracker.check_spike(0, 100));
        
        tracker.record_post(0, 700_000_000);
        assert!(tracker.check_spike(0, 100));
    }

    #[test]
    fn test_sentiment_rolling_buffer() {
        let buffer = SentimentRollingBuffer::default();
        
        // Push consistent sentiment
        for _ in 0..100 {
            buffer.push(500_000_000);
        }
        
        assert_eq!(buffer.rolling_average(100), 500_000_000);
        assert_eq!(buffer.sentiment_volatility(100), 0); // No variance
    }

    #[test]
    fn test_sentiment_volatility() {
        let buffer = SentimentRollingBuffer::default();
        
        // Push varying sentiments
        buffer.push(100);
        buffer.push(200);
        buffer.push(300);
        buffer.push(400);
        
        let vol = buffer.sentiment_volatility(4);
        assert!(vol > 0);
    }

    #[test]
    fn test_simd_spike_check() {
        let tracker = SubredditTracker::default();
        
        for i in 0..4 {
            tracker.register_subreddit(i, i as u64);
            for _ in 0..(50 + i * 25) {
                tracker.record_post(i, 500_000_000);
            }
        }
        
        let indices = [0, 1, 2, 3];
        let thresholds = [100, 100, 100, 100];
        let mask = tracker.simd_check_spikes(indices, thresholds);
        
        // Expected: [false, true, true, true] = 0b1110 = 14
        assert_eq!(mask, 0b1110);
    }

    #[test]
    fn test_post_aggregator() {
        let agg = PostAggregator::default();
        
        agg.add_post(0x111, 100, 1000, 0);
        agg.add_post(0x222, 200, 1001, 0);
        agg.add_post(0x333, 150, 1002, 1);
        
        assert_eq!(agg.occupancy(), 3);
        
        let mut posts = [(0u64, 0i64); 16];
        let count = agg.get_posts_by_subreddit(0, &mut posts);
        assert_eq!(count, 2);
    }
}
