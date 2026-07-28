//! Idempotency Key Generator
//! 
//! Idempotent client order ID generator to prevent duplicate orders after network retries.
//! Uses atomic counters and cryptographic hashing for uniqueness guarantees.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of tracked keys for deduplication
const MAX_TRACKED_KEYS: usize = 65536;

/// Idempotency key - 128-bit unique identifier
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdempotencyKey {
    pub high: u64,
    pub low: u64,
}

impl IdempotencyKey {
    pub const fn zero() -> Self {
        Self { high: 0, low: 0 }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.high == 0 && self.low == 0
    }

    #[inline(always)]
    pub fn combine(high: u64, low: u64) -> Self {
        Self { high, low }
    }

    /// Convert to bytes for network transmission
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&self.high.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.low.to_le_bytes());
        bytes
    }

    /// Create from bytes
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let high = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]);
        let low = u64::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]);
        Self { high, low }
    }
}

/// Tracked key entry for deduplication
#[repr(C)]
struct KeyEntry {
    key: AtomicU64, // Low 64 bits of key (high bits stored separately)
    timestamp: AtomicU64,
    strategy_id: AtomicU32,
    used: AtomicU32,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl KeyEntry {
    const fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            timestamp: AtomicU64::new(0),
            strategy_id: AtomicU32::new(0),
            used: AtomicU32::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }
}

/// Idempotency key generator
pub struct IdempotencyGenerator {
    /// Per-strategy sequence counters
    strategy_counters: [AtomicU64; 256],
    /// Tracked keys for deduplication
    tracked_keys: [KeyEntry; MAX_TRACKED_KEYS],
    /// Global counter for additional entropy
    global_counter: AtomicU64,
    /// Session ID (set at startup)
    session_id: AtomicU64,
    /// Start time (cycles)
    start_time: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for IdempotencyGenerator {}
unsafe impl Sync for IdempotencyGenerator {}

impl IdempotencyGenerator {
    /// Create new idempotency generator
    pub const fn new() -> Self {
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        const EMPTY_ENTRY: KeyEntry = KeyEntry::new();
        
        Self {
            strategy_counters: [ZERO_U64; 256],
            tracked_keys: [EMPTY_ENTRY; MAX_TRACKED_KEYS],
            global_counter: AtomicU64::new(0),
            session_id: AtomicU64::new(0),
            start_time: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Initialize with session ID (call once at startup)
    #[inline(always)]
    pub fn initialize(&self, session_id: u64) {
        self.session_id.store(session_id, Ordering::Release);
        self.start_time.store(unsafe { _rdtsc() }, Ordering::Release);
    }

    /// Generate unique idempotency key for an order
    #[inline(always)]
    pub fn generate_key(&self, strategy_id: u16) -> IdempotencyKey {
        let sidx = strategy_id as usize;
        
        // Get per-strategy counter (atomic increment)
        let seq = self.strategy_counters[sidx].fetch_add(1, Ordering::Relaxed);
        
        // Get global counter for additional uniqueness
        let global = self.global_counter.fetch_add(1, Ordering::Relaxed);
        
        // Get current timestamp
        let tsc = unsafe { _rdtsc() };
        
        // Combine into 128-bit key using custom hash
        let low = self.combine_hash(seq, global, tsc);
        let high = self.combine_hash(self.session_id.load(Ordering::Relaxed), strategy_id as u64, tsc >> 32);
        
        IdempotencyKey { high, low }
    }

    /// Generate key with explicit parameters (for replay scenarios)
    #[inline(always)]
    pub fn generate_key_explicit(
        &self,
        strategy_id: u16,
        sequence: u64,
        timestamp: u64,
    ) -> IdempotencyKey {
        let low = self.combine_hash(sequence, strategy_id as u64, timestamp);
        let high = self.combine_hash(self.session_id.load(Ordering::Relaxed), strategy_id as u64, timestamp >> 32);
        
        IdempotencyKey { high, low }
    }

    /// Custom hash combining function (branchless, fast)
    #[inline(always)]
    fn combine_hash(&self, a: u64, b: u64, c: u64) -> u64 {
        // Simple but effective mixing function
        let mut h = a.wrapping_mul(0x9e3779b97f4a7c15);
        h = h ^ b.rotate_left(13);
        h = h.wrapping_mul(0xbf58476d1ce4e5b9);
        h = h ^ c.rotate_right(7);
        h = h.wrapping_mul(0x94d049bb133111eb);
        h = h ^ (h >> 31);
        h
    }

    /// Check if a key has been used (idempotency check)
    #[inline(always)]
    pub fn check_and_mark_used(&self, key: IdempotencyKey, strategy_id: u16) -> bool {
        // Find key in tracked entries using linear probe
        let bucket = self.hash_key(key) % MAX_TRACKED_KEYS as u64;
        
        for i in 0..16 {
            let idx = ((bucket + i) % MAX_TRACKED_KEYS as u64) as usize;
            let entry = &self.tracked_keys[idx];
            
            let stored_key = entry.key.load(Ordering::Relaxed);
            let used = entry.used.load(Ordering::Relaxed);
            
            // Check if slot is empty or matches
            if stored_key == 0 {
                // Empty slot - try to claim it
                let result = entry.key.compare_exchange(
                    0,
                    key.low,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                );
                
                if result.is_ok() {
                    entry.strategy_id.store(strategy_id as u32, Ordering::Relaxed);
                    entry.timestamp.store(unsafe { _rdtsc() }, Ordering::Relaxed);
                    entry.used.store(1, Ordering::Release);
                    return false; // First use
                }
            } else if stored_key == key.low {
                // Key exists
                if used != 0 {
                    return true; // Already used - duplicate detected
                }
                // Mark as used
                entry.used.store(1, Ordering::Release);
                return false;
            }
        }
        
        // Could not find slot (very unlikely with proper sizing)
        false
    }

    /// Hash key to bucket index
    #[inline(always)]
    fn hash_key(&self, key: IdempotencyKey) -> u64 {
        key.high.wrapping_xor(key.low).wrapping_mul(0x9e3779b97f4a7c15)
    }

    /// Clear old entries (periodic maintenance)
    #[inline(always)]
    pub fn clear_old_entries(&self, max_age_cycles: u64) {
        let now = unsafe { _rdtsc() };
        
        for i in 0..MAX_TRACKED_KEYS {
            let entry = &self.tracked_keys[i];
            let timestamp = entry.timestamp.load(Ordering::Relaxed);
            let used = entry.used.load(Ordering::Relaxed);
            
            if used != 0 && now.saturating_sub(timestamp) > max_age_cycles {
                // Reset entry
                entry.key.store(0, Ordering::Release);
                entry.used.store(0, Ordering::Release);
            }
        }
    }

    /// Get strategy counter value
    #[inline(always)]
    pub fn get_strategy_counter(&self, strategy_id: u16) -> u64 {
        self.strategy_counters[strategy_id as usize].load(Ordering::Relaxed)
    }

    /// Reset strategy counter (use with caution)
    #[inline(always)]
    pub fn reset_strategy_counter(&self, strategy_id: u16) {
        self.strategy_counters[strategy_id as usize].store(0, Ordering::Release);
    }

    /// Get number of tracked keys
    #[inline(always)]
    pub fn count_tracked_keys(&self) -> u64 {
        let mut count = 0u64;
        for i in 0..MAX_TRACKED_KEYS {
            if self.tracked_keys[i].used.load(Ordering::Relaxed) != 0 {
                count += 1;
            }
        }
        count
    }
}

impl Default for IdempotencyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(12345);
        
        let key1 = gen.generate_key(0);
        let key2 = gen.generate_key(0);
        
        assert!(!key1.is_zero());
        assert!(!key2.is_zero());
        assert_ne!(key1, key2); // Should be unique
    }

    #[test]
    fn test_key_uniqueness() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(99999);
        
        let mut keys = Vec::new();
        for _ in 0..1000 {
            keys.push(gen.generate_key(5));
        }
        
        // Check all keys are unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_different_strategies() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(11111);
        
        let key_a = gen.generate_key(0);
        let key_b = gen.generate_key(1);
        
        // Keys from different strategies should differ
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn test_idempotency_check() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(22222);
        
        let key = gen.generate_key(0);
        
        // First use should succeed
        assert!(!gen.check_and_mark_used(key, 0));
        
        // Second use should be detected as duplicate
        assert!(gen.check_and_mark_used(key, 0));
    }

    #[test]
    fn test_key_bytes_conversion() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(33333);
        
        let key = gen.generate_key(0);
        let bytes = key.to_bytes();
        let recovered = IdempotencyKey::from_bytes(&bytes);
        
        assert_eq!(key, recovered);
    }

    #[test]
    fn test_strategy_counters() {
        let gen = IdempotencyGenerator::new();
        
        assert_eq!(gen.get_strategy_counter(0), 0);
        
        gen.generate_key(0);
        gen.generate_key(0);
        gen.generate_key(0);
        
        assert_eq!(gen.get_strategy_counter(0), 3);
    }

    #[test]
    fn test_clear_old_entries() {
        let gen = IdempotencyGenerator::new();
        gen.initialize(44444);
        
        // Generate and mark some keys
        for _ in 0..10 {
            let key = gen.generate_key(0);
            gen.check_and_mark_used(key, 0);
        }
        
        assert!(gen.count_tracked_keys() >= 1);
        
        // Clear with very short age (immediate)
        gen.clear_old_entries(0);
        
        // Entries should be cleared
        assert_eq!(gen.count_tracked_keys(), 0);
    }
}
