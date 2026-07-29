//! Chapter 4: Streaming Social Sentiment & Explainable AI (XAI)
//! Zero-allocation Twitter/X firehose parser tracking crypto influencer mentions.

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum influencers tracked
const MAX_INFLUENCERS: usize = 256;

/// Maximum token symbols tracked
const MAX_TOKENS: usize = 64;

/// Tweet buffer capacity (pre-allocated, zero heap)
const TWEET_BUFFER_SIZE: usize = 1024;

#[repr(C, align(64))]
pub struct InfluencerTracker {
    /// Active influencer IDs (stored as u64 hashes)
    influencer_ids: [AtomicU64; MAX_INFLUENCERS],
    /// Mention count per influencer
    mention_count: [AtomicU64; MAX_INFLUENCERS],
    /// Last tweet timestamp per influencer
    last_tweet_ts: [AtomicU64; MAX_INFLUENCERS],
    /// Influence score (scaled by 1e6)
    influence_score: [AtomicI64; MAX_INFLUENCERS],
    _padding: [u8; CACHE_LINE_SIZE],
}

#[repr(C, align(64))]
pub struct TokenMentionCounter {
    /// Token symbol hashes
    token_hashes: [AtomicU64; MAX_TOKENS],
    /// Mention counts
    mention_counts: [AtomicU64; MAX_TOKENS],
    /// Sentiment sum (for averaging)
    sentiment_sum: [AtomicI64; MAX_TOKENS],
    /// Spike detection threshold
    spike_threshold: [AtomicU64; MAX_TOKENS],
    _padding: [u8; CACHE_LINE_SIZE - MAX_TOKENS * 4 * 8],
}

#[repr(C, align(64))]
pub struct TweetBuffer {
    /// Pre-allocated fixed-size buffer entries
    /// Each entry stores: hash, timestamp, influencer_id, sentiment
    hashes: [AtomicU64; TWEET_BUFFER_SIZE],
    timestamps: [AtomicU64; TWEET_BUFFER_SIZE],
    influencer_ids: [AtomicU64; TWEET_BUFFER_SIZE],
    sentiments: [AtomicI64; TWEET_BUFFER_SIZE],
    /// Head index
    head: AtomicU64,
    /// Tail index
    tail: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 3 * 8],
}

// Compile-time assertions
const _: () = {
    assert!(MAX_INFLUENCERS <= 256, "MAX_INFLUENCERS exceeds limit");
    assert!(TWEET_BUFFER_SIZE.is_power_of_two(), "Buffer size must be power of 2");
};

impl Default for InfluencerTracker {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            influencer_ids: [INIT_U64; MAX_INFLUENCERS],
            mention_count: [INIT_U64; MAX_INFLUENCERS],
            last_tweet_ts: [INIT_U64; MAX_INFLUENCERS],
            influence_score: [INIT_I64; MAX_INFLUENCERS],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }
}

impl Default for TokenMentionCounter {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            token_hashes: [INIT_U64; MAX_TOKENS],
            mention_counts: [INIT_U64; MAX_TOKENS],
            sentiment_sum: [INIT_I64; MAX_TOKENS],
            spike_threshold: [INIT_U64; MAX_TOKENS],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_TOKENS * 4 * 8],
        }
    }
}

impl Default for TweetBuffer {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            hashes: [INIT_U64; TWEET_BUFFER_SIZE],
            timestamps: [INIT_U64; TWEET_BUFFER_SIZE],
            influencer_ids: [INIT_U64; TWEET_BUFFER_SIZE],
            sentiments: [INIT_I64; TWEET_BUFFER_SIZE],
            head: INIT_U64,
            tail: INIT_U64,
            _padding: [0u8; CACHE_LINE_SIZE - 3 * 8],
        }
    }
}

impl InfluencerTracker {
    /// Register an influencer by ID hash
    #[inline]
    pub fn register_influencer(&self, idx: usize, id_hash: u64, initial_score: i64) {
        if idx >= MAX_INFLUENCERS {
            return;
        }
        
        self.influencer_ids[idx].store(id_hash, Ordering::Relaxed);
        self.influence_score[idx].store(initial_score, Ordering::Relaxed);
        self.mention_count[idx].store(0, Ordering::Relaxed);
    }

    /// Record a tweet from an influencer (zero-allocation)
    #[inline]
    pub fn record_tweet(&self, influencer_idx: usize, tweet_hash: u64, timestamp: u64) {
        if influencer_idx >= MAX_INFLUENCERS {
            return;
        }
        
        self.mention_count[influencer_idx].fetch_add(1, Ordering::Relaxed);
        self.last_tweet_ts[influencer_idx].store(timestamp, Ordering::Relaxed);
        
        // Update influence score based on activity (branchless)
        let current = self.influence_score[influencer_idx].load(Ordering::Relaxed);
        let boost = ((current > 0) as i64) * 1000;
        self.influence_score[influencer_idx].store(current + boost, Ordering::Relaxed);
    }

    /// Get mention count for influencer
    #[inline]
    pub fn get_mention_count(&self, idx: usize) -> u64 {
        if idx >= MAX_INFLUENCERS {
            return 0;
        }
        self.mention_count[idx].load(Ordering::Relaxed)
    }

    /// Get influence score
    #[inline]
    pub fn get_influence_score(&self, idx: usize) -> i64 {
        if idx >= MAX_INFLUENCERS {
            return 0;
        }
        self.influence_score[idx].load(Ordering::Relaxed)
    }

    /// SIMD-accelerated influencer ID matching
    #[inline]
    pub fn simd_match_influencer<const N: usize>(&self, target_hash: u64) -> i32
    where [u64; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        
        unsafe {
            if N == 4 {
                let mut ids = [0u64; 4];
                for i in 0..4 {
                    ids[i] = self.influencer_ids[i].load(Ordering::Relaxed);
                }
                
                let ids_vec = _mm256_load_si256(ids.as_ptr() as *const __m256i);
                let target_vec = _mm256_set1_epi64x(target_hash as i64);
                
                // Compare for equality
                let cmp_vec = _mm256_cmpeq_epi64(ids_vec, target_vec);
                
                // Extract match position
                let mask = _mm256_movemask_epi8(cmp_vec);
                
                // Find first matching lane
                if mask & 0xFF != 0 {
                    if mask & 0xF != 0 { 0 }
                    else if mask & 0xF0 != 0 { 2 }
                    else { -1 }
                } else {
                    -1
                }
            } else {
                for i in 0..N {
                    if self.influencer_ids[i].load(Ordering::Relaxed) == target_hash {
                        return i as i32;
                    }
                }
                -1
            }
        }
    }

    /// Get top influencer by score
    #[inline]
    pub fn get_top_influencer(&self) -> Option<(usize, i64)> {
        let mut best_idx = None;
        let mut best_score = i64::MIN;
        
        for i in 0..MAX_INFLUENCERS {
            let score = self.influence_score[i].load(Ordering::Relaxed);
            if score > best_score && self.influencer_ids[i].load(Ordering::Relaxed) != 0 {
                best_score = score;
                best_idx = Some(i);
            }
        }
        
        best_idx.map(|idx| (idx, best_score))
    }
}

impl TokenMentionCounter {
    /// Register a token symbol
    #[inline]
    pub fn register_token(&self, idx: usize, symbol_hash: u64, threshold: u64) {
        if idx >= MAX_TOKENS {
            return;
        }
        
        self.token_hashes[idx].store(symbol_hash, Ordering::Relaxed);
        self.spike_threshold[idx].store(threshold, Ordering::Relaxed);
        self.mention_counts[idx].store(0, Ordering::Relaxed);
        self.sentiment_sum[idx].store(0, Ordering::Relaxed);
    }

    /// Record token mention with sentiment (branchless)
    #[inline]
    pub fn record_mention(&self, token_idx: usize, sentiment: i64) {
        if token_idx >= MAX_TOKENS {
            return;
        }
        
        self.mention_counts[token_idx].fetch_add(1, Ordering::Relaxed);
        self.sentiment_sum[token_idx].fetch_add(sentiment, Ordering::Relaxed);
    }

    /// Check if mention spike detected (branchless threshold crossing)
    #[inline]
    pub fn is_spike_detected(&self, token_idx: usize) -> bool {
        if token_idx >= MAX_TOKENS {
            return false;
        }
        
        let count = self.mention_counts[token_idx].load(Ordering::Relaxed);
        let threshold = self.spike_threshold[token_idx].load(Ordering::Relaxed);
        
        count >= threshold
    }

    /// Get average sentiment for token (fixed-point)
    #[inline]
    pub fn get_avg_sentiment(&self, token_idx: usize) -> i64 {
        if token_idx >= MAX_TOKENS {
            return 0;
        }
        
        let count = self.mention_counts[token_idx].load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }
        
        let sum = self.sentiment_sum[token_idx].load(Ordering::Relaxed);
        sum / count as i64
    }

    /// Get mention count
    #[inline]
    pub fn get_mention_count(&self, token_idx: usize) -> u64 {
        if token_idx >= MAX_TOKENS {
            return 0;
        }
        self.mention_counts[token_idx].load(Ordering::Relaxed)
    }

    /// Reset counter for token
    #[inline]
    pub fn reset_token(&self, token_idx: usize) {
        if token_idx >= MAX_TOKENS {
            return;
        }
        
        self.mention_counts[token_idx].store(0, Ordering::Relaxed);
        self.sentiment_sum[token_idx].store(0, Ordering::Relaxed);
    }

    /// SIMD-accelerated token matching
    #[inline]
    pub fn simd_match_token<const N: usize>(&self, target_hash: u64) -> i32 {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        
        unsafe {
            if N == 4 {
                let mut hashes = [0u64; 4];
                for i in 0..4 {
                    hashes[i] = self.token_hashes[i].load(Ordering::Relaxed);
                }
                
                let hash_vec = _mm256_load_si256(hashes.as_ptr() as *const __m256i);
                let target_vec = _mm256_set1_epi64x(target_hash as i64);
                
                let cmp_vec = _mm256_cmpeq_epi64(hash_vec, target_vec);
                let mask = _mm256_movemask_epi8(cmp_vec);
                
                if mask & 0xFF != 0 {
                    if mask & 0xF != 0 { 0 }
                    else if mask & 0xF0 != 0 { 2 }
                    else { -1 }
                } else {
                    -1
                }
            } else {
                for i in 0..N {
                    if self.token_hashes[i].load(Ordering::Relaxed) == target_hash {
                        return i as i32;
                    }
                }
                -1
            }
        }
    }
}

impl TweetBuffer {
    /// Push tweet to buffer (lock-free circular)
    #[inline]
    pub fn push(&self, hash: u64, timestamp: u64, influencer_id: u64, sentiment: i64) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if head.wrapping_sub(tail) >= TWEET_BUFFER_SIZE as u64 {
            return false; // Buffer full
        }
        
        let idx = (head % TWEET_BUFFER_SIZE as u64) as usize;
        
        self.hashes[idx].store(hash, Ordering::Relaxed);
        self.timestamps[idx].store(timestamp, Ordering::Relaxed);
        self.influencer_ids[idx].store(influencer_id, Ordering::Relaxed);
        self.sentiments[idx].store(sentiment, Ordering::Relaxed);
        
        self.head.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Pop oldest tweet from buffer
    #[inline]
    pub fn pop(&self) -> Option<(u64, u64, u64, i64)> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        if tail >= head {
            return None;
        }
        
        let idx = (tail % TWEET_BUFFER_SIZE as u64) as usize;
        
        let hash = self.hashes[idx].load(Ordering::Relaxed);
        let ts = self.timestamps[idx].load(Ordering::Relaxed);
        let inf = self.influencer_ids[idx].load(Ordering::Relaxed);
        let sent = self.sentiments[idx].load(Ordering::Relaxed);
        
        self.tail.fetch_add(1, Ordering::Relaxed);
        
        Some((hash, ts, inf, sent))
    }

    /// Get buffer occupancy
    #[inline]
    pub fn occupancy(&self) -> u64 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// Clear buffer
    #[inline]
    pub fn clear(&self) {
        let head = self.head.load(Ordering::Relaxed);
        self.tail.store(head, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_influencer_tracking() {
        let tracker = InfluencerTracker::default();
        
        tracker.register_influencer(0, 0xDEADBEEF, 100_000_000);
        tracker.record_tweet(0, 0x12345678, 1000);
        tracker.record_tweet(0, 0x12345679, 1001);
        
        assert_eq!(tracker.get_mention_count(0), 2);
        assert!(tracker.get_influence_score(0) > 100_000_000);
    }

    #[test]
    fn test_token_mentions() {
        let counter = TokenMentionCounter::default();
        
        counter.register_token(0, 0xBTC_HASH, 100);
        counter.record_mention(0, 1_000_000_000); // Positive sentiment
        counter.record_mention(0, -500_000_000);  // Negative sentiment
        
        assert_eq!(counter.get_mention_count(0), 2);
        assert_eq!(counter.get_avg_sentiment(0), 250_000_000);
    }

    #[test]
    fn test_spike_detection() {
        let counter = TokenMentionCounter::default();
        counter.register_token(0, 0xETH_HASH, 10);
        
        // Below threshold
        for _ in 0..9 {
            counter.record_mention(0, 0);
        }
        assert!(!counter.is_spike_detected(0));
        
        // At threshold
        counter.record_mention(0, 0);
        assert!(counter.is_spike_detected(0));
    }

    #[test]
    fn test_tweet_buffer_operations() {
        let buffer = TweetBuffer::default();
        
        assert!(buffer.push(0x111, 1000, 0xAAA, 100));
        assert!(buffer.push(0x222, 1001, 0xBBB, -50));
        
        assert_eq!(buffer.occupancy(), 2);
        
        let popped = buffer.pop();
        assert_eq!(popped, Some((0x111, 1000, 0xAAA, 100)));
        assert_eq!(buffer.occupancy(), 1);
    }

    #[test]
    fn test_simd_influencer_match() {
        let tracker = InfluencerTracker::default();
        tracker.register_influencer(0, 0x1111, 100);
        tracker.register_influencer(1, 0x2222, 200);
        tracker.register_influencer(2, 0x3333, 300);
        tracker.register_influencer(3, 0x4444, 400);
        
        let idx = tracker.simd_match_influencer::<4>(0x3333);
        assert_eq!(idx, 2);
        
        let not_found = tracker.simd_match_influencer::<4>(0xFFFF);
        assert_eq!(not_found, -1);
    }

    #[test]
    fn test_buffer_wraparound() {
        let buffer = TweetBuffer::default();
        
        // Fill buffer
        for i in 0..TWEET_BUFFER_SIZE {
            buffer.push(i as u64, i as u64, i as u64, i as i64);
        }
        
        // Remove half
        for _ in 0..TWEET_BUFFER_SIZE / 2 {
            buffer.pop();
        }
        
        // Add more (should wrap around)
        for i in 0..100 {
            assert!(buffer.push((1000 + i) as u64, (1000 + i) as u64, 0, 0));
        }
    }
}
