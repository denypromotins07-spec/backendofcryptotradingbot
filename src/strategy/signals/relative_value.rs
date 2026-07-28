//! ETH/BTC and SOL/BTC Relative-Value Spread Tracking
//! 
//! Implements z-score calculations for relative value trading pairs.
//! Uses circular buffers for O(1) rolling statistics and branchless
//! threshold detection for signal generation.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache-line padded atomic boolean
#[repr(C, align(64))]
pub struct PaddedAtomicBool {
    value: AtomicBool,
    _padding: [u8; 63],
}

impl PaddedAtomicBool {
    #[inline(always)]
    pub const fn new(val: bool) -> Self {
        Self {
            value: AtomicBool::new(val),
            _padding: [0u8; 63],
        }
    }
    
    #[inline(always)]
    pub fn set(&self, val: bool) {
        self.value.store(val, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn get(&self) -> bool {
        self.value.load(Ordering::Relaxed)
    }
}

/// Cache-line padded atomic u64
#[repr(C, align(64))]
pub struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 56],
}

impl PaddedAtomicU64 {
    #[inline(always)]
    pub const fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; 56],
        }
    }
    
    #[inline(always)]
    pub fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline(always)]
    pub fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed);
    }
}

/// Spread state - fits in single cache line
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct SpreadState {
    /// Current spread value (Q32.32 fixed-point)
    pub spread_i64: i64,
    /// Z-score of spread (Q16.16 fixed-point)
    pub z_score_i32: i32,
    /// Rolling mean (Q32.32)
    pub mean_i64: i64,
    /// Rolling std dev (Q32.32)
    pub std_dev_i64: i64,
    /// Signal: 0=neutral, 1=long spread, 2=short spread
    pub signal: u8,
    /// Timestamp (TSC cycles)
    pub timestamp_tsc: u64,
    _padding: [u8; 35],
}

const _: () = assert!(core::mem::size_of::<SpreadState>() == 64);

impl SpreadState {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            spread_i64: 0,
            z_score_i32: 0,
            mean_i64: 0,
            std_dev_i64: 0,
            signal: 0,
            timestamp_tsc: 0,
            _padding: [0u8; 35],
        }
    }
    
    #[inline(always)]
    pub fn spread(&self) -> f64 {
        self.spread_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn z_score(&self) -> f64 {
        self.z_score_i32 as f64 / 65536.0
    }
    
    #[inline(always)]
    pub fn mean(&self) -> f64 {
        self.mean_i64 as f64 / 4294967296.0
    }
    
    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.std_dev_i64 as f64 / 4294967296.0
    }
}

/// Rolling statistics calculator for spread z-scores
#[repr(C, align(64))]
pub struct RollingStats<const N: usize> {
    /// Circular buffer of values (Q32.32)
    buffer: [i64; N],
    /// Head position
    head: u64,
    /// Count of valid entries
    count: u64,
    /// Running sum (Q32.32 extended to i128)
    sum: i128,
    /// Running sum of squares (Q64.64)
    sum_sq: i128,
    /// Cached mean (Q32.32)
    cached_mean: i64,
    /// Cached variance (Q64.64)
    cached_variance: i128,
    /// Dirty flag for cache invalidation
    dirty: bool,
    _padding: [u8; 27],
}

impl<const N: usize> RollingStats<N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            buffer: [0; N],
            head: 0,
            count: 0,
            sum: 0,
            sum_sq: 0,
            cached_mean: 0,
            cached_variance: 0,
            dirty: false,
            _padding: [0u8; 27],
        }
    }
    
    /// Push new value and update statistics - O(1)
    #[inline(always)]
    pub fn push(&mut self, value: i64) {
        let idx = (self.head % N as u64) as usize;
        let old_value = self.buffer[idx];
        
        // Update running sums
        self.sum = self.sum - old_value as i128 + value as i128;
        self.sum_sq = self.sum_sq - (old_value * old_value) as i128 + (value * value) as i128;
        
        self.buffer[idx] = value;
        self.head += 1;
        
        if self.count < N as u64 {
            self.count += 1;
        }
        
        self.dirty = true;
    }
    
    /// Get mean (cached if not dirty) - O(1)
    #[inline(always)]
    pub fn mean(&mut self) -> i64 {
        if self.dirty || self.cached_mean == 0 {
            if self.count == 0 {
                return 0;
            }
            self.cached_mean = (self.sum / self.count as i128) as i64;
            self.dirty = false;
        }
        self.cached_mean
    }
    
    /// Get variance (cached if not dirty) - O(1)
    #[inline(always)]
    pub fn variance(&mut self) -> i128 {
        if self.dirty || self.cached_variance == 0 {
            if self.count < 2 {
                return 0;
            }
            let mean = self.sum / self.count as i128;
            self.cached_variance = (self.sum_sq / self.count as i128) - (mean * mean);
        }
        self.cached_variance
    }
    
    /// Calculate z-score for a value - O(1)
    #[inline(always)]
    pub fn z_score(&mut self, value: i64) -> i32 {
        let mean = self.mean();
        let var = self.variance();
        
        if var <= 0 {
            return 0;
        }
        
        // Fixed-point sqrt approximation using Newton-Raphson
        let std = Self::isqrt(var as u128) as i64;
        
        if std == 0 {
            return 0;
        }
        
        // Z-score in Q16.16 format
        let diff = value - mean;
        ((diff << 16) / std) as i32
    }
    
    /// Fast inverse square root approximation for std dev
    #[inline(always)]
    fn isqrt(x: u128) -> u64 {
        if x == 0 {
            return 0;
        }
        
        // Newton-Raphson iteration for integer sqrt
        let mut guess = (x >> 64) as u64;
        if guess == 0 {
            guess = 1;
        }
        
        // Two iterations for sufficient precision
        guess = ((guess as u128) + (x / guess as u128)) as u64 >> 1;
        guess = ((guess as u128) + (x / guess as u128)) as u64 >> 1;
        
        guess
    }
}

/// Relative value tracker for a single pair (e.g., ETH/BTC)
#[repr(C, align(64))]
pub struct RelativeValueTracker<const WINDOW: usize> {
    /// Asset A identifier (e.g., ETH)
    pub asset_a: u64,
    /// Asset B identifier (e.g., BTC)
    pub asset_b: u64,
    /// Rolling stats for spread
    spread_stats: RollingStats<WINDOW>,
    /// Current spread state
    pub state: SpreadState,
    /// Entry threshold (z-score, Q16.16)
    entry_threshold: i32,
    /// Exit threshold (z-score, Q16.16)
    exit_threshold: i32,
    /// Circuit breaker active
    pub circuit_breaker: PaddedAtomicBool,
    /// Last signal timestamp
    pub last_signal_tsc: PaddedAtomicU64,
    _padding: [u8; 16],
}

impl<const WINDOW: usize> RelativeValueTracker<WINDOW> {
    #[inline(always)]
    pub const fn new(asset_a: u64, asset_b: u64, entry_thresh: f64, exit_thresh: f64) -> Self {
        Self {
            asset_a,
            asset_b,
            spread_stats: RollingStats::new(),
            state: SpreadState::new(),
            entry_threshold: (entry_thresh * 65536.0) as i32,
            exit_threshold: (exit_thresh * 65536.0) as i32,
            circuit_breaker: PaddedAtomicBool::new(false),
            last_signal_tsc: PaddedAtomicU64::new(0),
            _padding: [0u8; 16],
        }
    }
    
    /// Update with new prices and compute spread z-score
    /// Prices are in Q32.32 fixed-point format
    #[inline(always)]
    pub fn update(&mut self, price_a: i64, price_b: i64, timestamp_tsc: u64) -> u8 {
        if self.circuit_breaker.get() {
            return 0;
        }
        
        // Calculate spread: log(price_a) - log(price_b) approximated as ratio
        // For fixed-point: spread = (price_a - price_b) / price_b
        // Simplified: spread = price_a - price_b (for similar magnitude assets)
        let spread = price_a - price_b;
        
        // Update rolling statistics
        self.spread_stats.push(spread);
        
        // Calculate z-score
        let z_score = self.spread_stats.z_score(spread);
        
        // Update state
        self.state.spread_i64 = spread;
        self.state.z_score_i32 = z_score;
        self.state.mean_i64 = self.spread_stats.mean();
        // Variance to std_dev conversion
        let var = self.spread_stats.variance();
        self.state.std_dev_i64 = RollingStats::<WINDOW>::isqrt(var as u128) as i64;
        self.state.timestamp_tsc = timestamp_tsc;
        
        // Branchless signal generation
        let abs_z = z_score.abs();
        let above_entry = (abs_z >= self.entry_threshold.abs()) as u8;
        let positive_z = (z_score > 0) as u8;
        
        // Signal: 0=neutral, 1=long spread (buy A, sell B), 2=short spread
        // Long spread when z < -entry (A is cheap relative to B)
        // Short spread when z > entry (A is expensive relative to B)
        let long_signal = ((z_score < -self.entry_threshold) as u8) * above_entry;
        let short_signal = ((z_score > self.entry_threshold) as u8) * above_entry;
        
        self.state.signal = long_signal * 1 + short_signal * 2;
        
        // Record signal timestamp if signal changed
        if self.state.signal != 0 {
            self.last_signal_tsc.store(timestamp_tsc);
        }
        
        self.state.signal
    }
    
    /// Check if position should be closed based on exit threshold
    #[inline(always)]
    pub fn should_exit(&self, current_z: i32, position_type: u8) -> bool {
        let abs_z = current_z.abs();
        let below_exit = (abs_z < self.exit_threshold.abs()) as u8;
        
        // Branchless exit logic
        ((below_exit == 1) & (position_type != 0)) != 0
    }
    
    /// Trigger circuit breaker
    #[inline(always)]
    pub fn halt(&mut self) {
        self.circuit_breaker.set(true);
    }
    
    /// Reset circuit breaker
    #[inline(always)]
    pub fn resume(&mut self) {
        self.circuit_breaker.set(false);
    }
    
    /// Get current z-score as f64
    #[inline(always)]
    pub fn z_score_f64(&self) -> f64 {
        self.state.z_score_i32 as f64 / 65536.0
    }
}

/// Multi-pair relative value engine for ETH/BTC and SOL/BTC
#[repr(C, align(64))]
pub struct MultiPairRelativeValue<const WINDOW: usize> {
    /// ETH/BTC tracker
    eth_btc: RelativeValueTracker<WINDOW>,
    /// SOL/BTC tracker
    sol_btc: RelativeValueTracker<WINDOW>,
    /// Cross-pair correlation (Q16.16)
    pub pair_correlation: i32,
    /// Active pair selector (bitmask)
    pub active_pairs: u8,
    _padding: [u8; 51],
}

impl<const WINDOW: usize> MultiPairRelativeValue<WINDOW> {
    #[inline(always)]
    pub const fn new(entry_thresh: f64, exit_thresh: f64) -> Self {
        Self {
            eth_btc: RelativeValueTracker::new(
                0x4554480000000000, // ETH
                0x4254430000000000, // BTC
                entry_thresh,
                exit_thresh,
            ),
            sol_btc: RelativeValueTracker::new(
                0x534F4C0000000000, // SOL
                0x4254430000000000, // BTC
                entry_thresh,
                exit_thresh,
            ),
            pair_correlation: 0,
            active_pairs: 0b11, // Both pairs active
            _padding: [0u8; 51],
        }
    }
    
    /// Update both pairs with new prices
    #[inline(always)]
    pub fn update(
        &mut self,
        btc_price: i64,
        eth_price: i64,
        sol_price: i64,
        timestamp_tsc: u64,
    ) -> (u8, u8) {
        let eth_btc_signal = if (self.active_pairs & 0b01) != 0 {
            self.eth_btc.update(eth_price, btc_price, timestamp_tsc)
        } else {
            0
        };
        
        let sol_btc_signal = if (self.active_pairs & 0b10) != 0 {
            self.sol_btc.update(sol_price, btc_price, timestamp_tsc)
        } else {
            0
        };
        
        // Update cross-pair correlation (simplified)
        let eth_z = self.eth_btc.z_score_f64();
        let sol_z = self.sol_btc.z_score_f64();
        self.pair_correlation = ((eth_z * sol_z).signum() * 32768.0) as i32;
        
        (eth_btc_signal, sol_btc_signal)
    }
    
    /// Get ETH/BTC z-score
    #[inline(always)]
    pub fn eth_btc_z_score(&self) -> f64 {
        self.eth_btc.z_score_f64()
    }
    
    /// Get SOL/BTC z-score
    #[inline(always)]
    pub fn sol_btc_z_score(&self) -> f64 {
        self.sol_btc.z_score_f64()
    }
    
    /// Disable a specific pair
    #[inline(always)]
    pub fn disable_pair(&mut self, pair_mask: u8) {
        self.active_pairs &= !pair_mask;
    }
    
    /// Enable a specific pair
    #[inline(always)]
    pub fn enable_pair(&mut self, pair_mask: u8) {
        self.active_pairs |= pair_mask;
    }
}

// Compile-time assertions
const _: () = assert!(core::mem::size_of::<RollingStats<64>>() % 64 == 0);
const _: () = assert!(core::mem::size_of::<RelativeValueTracker<64>>() == 64);
const _: () = assert!(core::mem::size_of::<MultiPairRelativeValue<64>>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spread_state_size() {
        assert_eq!(core::mem::size_of::<SpreadState>(), 64);
    }
    
    #[test]
    fn test_rolling_stats_z_score() {
        let mut stats: RollingStats<16> = RollingStats::new();
        let base = 4294967296i64; // 1.0 in Q32.32
        
        // Push consistent values
        for i in 0..10 {
            stats.push(base + (i as i64 * 100000));
        }
        
        // New value far from mean should have high z-score
        let extreme = base + 10000000i64;
        let z = stats.z_score(extreme);
        assert!(z.abs() > 10000); // Z-score > 0.15 in Q16.16
    }
    
    #[test]
    fn test_relative_value_tracker() {
        let mut tracker: RelativeValueTracker<32> = RelativeValueTracker::new(
            0x4554480000000000,
            0x4254430000000000,
            2.0,
            0.5,
        );
        
        let base = 4294967296i64;
        
        // Feed normal prices
        for i in 0..20 {
            tracker.update(base + (i as i64 * 1000000), base + (i as i64 * 1000000), i);
        }
        
        // Verify state is updated
        assert!(tracker.state.timestamp_tsc > 0);
    }
}
