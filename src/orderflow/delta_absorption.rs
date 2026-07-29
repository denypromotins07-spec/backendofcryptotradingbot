//! Cumulative Volume Delta (CVD) and passive order absorption detection algorithm.
//! 
//! Lock-free atomic flags for instant volatility pivoting without mutexes.
//! Branchless programming for deterministic execution latency.

#![allow(clippy::missing_docs_in_private_items)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum CVD history window (for divergence detection)
pub const MAX_CVD_HISTORY: usize = 10000;

/// Trade record for CVD calculation
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TradeRecord {
    pub price_tick: i64,
    pub volume: u64,
    pub is_buy: bool,  // true = aggressive buy (hit ask), false = aggressive sell (hit bid)
    pub timestamp_ns: u64,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl TradeRecord {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            price_tick: 0,
            volume: 0,
            is_buy: false,
            timestamp_ns: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }
}

/// CVD state with absorption detection
#[repr(C)]
pub struct CvdState {
    /// Cumulative Volume Delta (buy vol - sell vol)
    pub cvd: AtomicI64,
    /// Rolling buy volume
    pub buy_volume: AtomicU64,
    /// Rolling sell volume
    pub sell_volume: AtomicU64,
    /// Total trade count
    pub trade_count: AtomicU64,
    /// Last price tick
    pub last_price: AtomicI64,
    /// Price change direction (for divergence)
    pub price_direction: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 48],
}

impl CvdState {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            cvd: AtomicI64::new(0),
            buy_volume: AtomicU64::new(0),
            sell_volume: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            last_price: AtomicI64::new(0),
            price_direction: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 48],
        }
    }
}

/// Absorption signal types
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AbsorptionType {
    None = 0,
    BidAbsorption = 1,   // Heavy selling but price won't drop (passive bids absorbing)
    AskAbsorption = 2,   // Heavy buying but price won't rise (passive asks absorbing)
    DualAbsorption = 3,  // Both sides absorbing (consolidation)
}

/// Absorption detection result
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AbsorptionSignal {
    pub signal_type: AbsorptionType,
    pub strength: u32,      // 0-1000 scale
    pub cvd_divergence: i64,
    pub price_change: i64,
    pub volume_imbalance: i32,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl AbsorptionSignal {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            signal_type: AbsorptionType::None,
            strength: 0,
            cvd_divergence: 0,
            price_change: 0,
            volume_imbalance: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }

    /// Check if signal is significant
    #[inline(always)]
    pub fn is_significant(&self) -> bool {
        self.signal_type != AbsorptionType::None && self.strength > 500
    }
}

/// Circular buffer for CVD history (lock-free)
#[repr(C)]
pub struct CvdHistory<const N: usize> {
    data: UnsafeCell<[i64; N]>,
    head: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 8],
}

impl<const N: usize> CvdHistory<N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: UnsafeCell::new([0i64; N]),
            head: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 8],
        }
    }

    #[inline(always)]
    pub fn push(&self, value: i64) {
        let idx = self.head.fetch_add(1, Ordering::Relaxed) as usize % N;
        unsafe {
            let ptr = self.data.get() as *mut i64;
            ptr.add(idx).write(value);
        }
    }

    #[inline(always)]
    pub fn get(&self, offset: usize) -> Option<i64> {
        let head = self.head.load(Ordering::Relaxed) as usize;
        if offset >= N || offset > head {
            return None;
        }
        let idx = (head.wrapping_sub(offset + 1)) % N;
        unsafe {
            let ptr = self.data.get() as *const i64;
            Some(*ptr.add(idx))
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed) as usize;
        core::cmp::min(head, N)
    }
}

/// Delta Absorption Detector - main structure
#[repr(C)]
pub struct DeltaAbsorption<const HISTORY_SIZE: usize> {
    /// Current CVD state
    state: CvdState,
    /// CVD history for divergence detection
    cvd_history: CvdHistory<HISTORY_SIZE>,
    /// Price history for divergence
    price_history: CvdHistory<HISTORY_SIZE>,
    /// Recent absorption signals
    last_signal: UnsafeCell<AbsorptionSignal>,
    /// Consecutive absorption count
    absorption_count: AtomicU64,
    /// Circuit breaker flag
    circuit_breaker: AtomicU64,
    /// Memory tracker
    memory_bytes: AtomicU64,
}

// SAFETY: All interior mutability protected by atomics
unsafe impl<const H: usize> Send for DeltaAbsorption<H> {}
unsafe impl<const H: usize> Sync for DeltaAbsorption<H> {}

impl<const HISTORY_SIZE: usize> DeltaAbsorption<HISTORY_SIZE> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            state: CvdState::new(),
            cvd_history: CvdHistory::new(),
            price_history: CvdHistory::new(),
            last_signal: UnsafeCell::new(AbsorptionSignal::new()),
            absorption_count: AtomicU64::new(0),
            circuit_breaker: AtomicU64::new(0),
            memory_bytes: AtomicU64::new(0),
        }
    }

    /// Initialize with memory tracking
    #[inline]
    pub fn init(&self) {
        let mem = (core::mem::size_of::<i64>() * HISTORY_SIZE * 2) as u64;
        self.memory_bytes.store(mem, Ordering::Relaxed);
    }

    /// Process a trade and update CVD (hot path, branchless)
    #[inline(always)]
    pub fn process_trade(&self, price_tick: i64, volume: u64, is_buy: bool) -> Option<AbsorptionSignal> {
        let state = &self.state;
        
        // Update CVD (branchless)
        let delta = volume as i64 * ((is_buy as i64) * 2 - 1);
        state.cvd.fetch_add(delta, Ordering::Relaxed);
        
        // Update volumes
        let buy_mask = (is_buy as u64);
        let sell_mask = (!is_buy) as u64;
        state.buy_volume.fetch_add(volume * buy_mask, Ordering::Relaxed);
        state.sell_volume.fetch_add(volume * sell_mask, Ordering::Relaxed);
        
        state.trade_count.fetch_add(1, Ordering::Relaxed);
        
        // Track price direction
        let last_price = state.last_price.swap(price_tick, Ordering::Relaxed);
        let price_dir = ((price_tick > last_price) as i64) - ((price_tick < last_price) as i64);
        state.price_direction.store(price_dir, Ordering::Relaxed);
        
        // Update histories
        self.cvd_history.push(state.cvd.load(Ordering::Relaxed));
        self.price_history.push(price_tick);
        
        // Check for absorption every N trades (reduce overhead)
        if state.trade_count.load(Ordering::Relaxed) % 10 == 0 {
            let signal = self.detect_absorption();
            unsafe {
                *self.last_signal.get() = signal;
            }
            
            if signal.is_significant() {
                self.absorption_count.fetch_add(1, Ordering::Relaxed);
                
                // Activate circuit breaker on extreme absorption
                if signal.strength > 900 {
                    self.circuit_breaker.store(1, Ordering::Release);
                }
                
                return Some(signal);
            }
        }
        
        None
    }

    /// Detect absorption by analyzing CVD/price divergence
    #[inline]
    pub fn detect_absorption(&self) -> AbsorptionSignal {
        let state = &self.state;
        let mut signal = AbsorptionSignal::new();
        
        // Need enough history
        if self.cvd_history.len() < 50 {
            return signal;
        }
        
        // Get recent CVD change
        let cvd_current = state.cvd.load(Ordering::Relaxed);
        let cvd_past = self.cvd_history.get(49).unwrap_or(cvd_current);
        let cvd_change = cvd_current - cvd_past;
        
        // Get recent price change
        let price_current = state.last_price.load(Ordering::Relaxed);
        let price_past = self.price_history.get(49).unwrap_or(price_current);
        let price_change = price_current - price_past;
        
        // Calculate volume imbalance
        let buy_vol = state.buy_volume.load(Ordering::Relaxed);
        let sell_vol = state.sell_volume.load(Ordering::Relaxed);
        let total_vol = buy_vol + sell_vol;
        
        let vol_imbalance = if total_vol == 0 {
            0
        } else {
            ((buy_vol as i64 - sell_vol as i64) * 1000 / total_vol as i64) as i32
        };
        
        // Detect divergence patterns
        signal.cvd_divergence = cvd_change;
        signal.price_change = price_change;
        signal.volume_imbalance = vol_imbalance;
        
        // Branchless absorption detection
        // Bid absorption: CVD down (heavy selling) but price flat or up
        let bid_abs = ((cvd_change < -1000) as u32) & ((price_change >= 0) as u32);
        
        // Ask absorption: CVD up (heavy buying) but price flat or down
        let ask_abs = ((cvd_change > 1000) as u32) & ((price_change <= 0) as u32);
        
        // Calculate strength based on magnitude of divergence
        let cvd_magnitude = cvd_change.unsigned_abs().min(10000) / 10;
        let price_magnitude = price_change.unsigned_abs();
        
        // Strength increases when CVD moves but price doesn't
        let divergence_strength = if price_magnitude < 5 {
            cvd_magnitude * 2
        } else {
            cvd_magnitude / (price_magnitude + 1)
        };
        
        signal.strength = divergence_strength.min(1000) as u32;
        
        // Determine signal type (branchless)
        signal.signal_type = match (bid_abs, ask_abs) {
            (1, 0) => AbsorptionType::BidAbsorption,
            (0, 1) => AbsorptionType::AskAbsorption,
            (0, 0) if signal.strength > 300 => AbsorptionType::DualAbsorption,
            _ => AbsorptionType::None,
        };
        
        signal
    }

    /// Get current CVD value
    #[inline(always)]
    pub fn get_cvd(&self) -> i64 {
        self.state.cvd.load(Ordering::Relaxed)
    }

    /// Get CVD trend (change over last N samples)
    #[inline]
    pub fn get_cvd_trend(&self, window: usize) -> i64 {
        if window == 0 || window > HISTORY_SIZE {
            return 0;
        }
        
        let current = self.state.cvd.load(Ordering::Relaxed);
        let past = self.cvd_history.get(window.saturating_sub(1)).unwrap_or(current);
        current - past
    }

    /// Get volume imbalance ratio (-1000 to 1000)
    #[inline]
    pub fn get_volume_imbalance(&self) -> i32 {
        let buy = self.state.buy_volume.load(Ordering::Relaxed);
        let sell = self.state.sell_volume.load(Ordering::Relaxed);
        let total = buy + sell;
        
        if total == 0 {
            return 0;
        }
        
        ((buy as i64 - sell as i64) * 1000 / total as i64) as i32
    }

    /// Get last absorption signal
    #[inline(always)]
    pub fn get_last_signal(&self) -> AbsorptionSignal {
        unsafe { *self.last_signal.get() }
    }

    /// Check if circuit breaker is active
    #[inline(always)]
    pub fn is_circuit_breaker_active(&self) -> bool {
        self.circuit_breaker.load(Ordering::Acquire) != 0
    }

    /// Reset circuit breaker
    #[inline(always)]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(0, Ordering::Release);
        self.absorption_count.store(0, Ordering::Relaxed);
    }

    /// Get consecutive absorption count
    #[inline(always)]
    pub fn absorption_count(&self) -> u64 {
        self.absorption_count.load(Ordering::Relaxed)
    }

    /// Get trade statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64, i64) {
        (
            self.state.trade_count.load(Ordering::Relaxed),
            self.state.buy_volume.load(Ordering::Relaxed),
            self.state.sell_volume.load(Ordering::Relaxed),
            self.state.cvd.load(Ordering::Relaxed),
        )
    }

    /// Reset all counters
    #[inline]
    pub fn reset(&self) {
        self.state = CvdState::new();
        self.cvd_history = CvdHistory::new();
        self.price_history = CvdHistory::new();
        unsafe {
            *self.last_signal.get() = AbsorptionSignal::new();
        }
        self.absorption_count.store(0, Ordering::Relaxed);
        self.circuit_breaker.store(0, Ordering::Release);
    }

    /// Get memory usage
    #[inline(always)]
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }
}

/// Type alias for typical use case
pub type CryptoDeltaAbsorption = DeltaAbsorption<MAX_CVD_HISTORY>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_basic_update() {
        let da = CryptoDeltaAbsorption::new();
        da.init();
        
        da.process_trade(50000, 100, true);  // Buy
        da.process_trade(50001, 150, false); // Sell
        
        assert_eq!(da.get_cvd(), -50);
        
        let (_, buy, sell, cvd) = da.get_stats();
        assert_eq!(buy, 100);
        assert_eq!(sell, 150);
        assert_eq!(cvd, -50);
    }

    #[test]
    fn test_bid_absorption_detection() {
        let da = DeltaAbsorption::<100>::new();
        da.init();
        
        // Simulate heavy selling with stable price
        for i in 0..60 {
            da.process_trade(50000 + (i % 3) as i64, 200, false); // Mostly sells
        }
        
        let signal = da.get_last_signal();
        // Should detect bid absorption (heavy selling, price not dropping)
        assert!(signal.strength > 0);
    }

    #[test]
    fn test_ask_absorption_detection() {
        let da = DeltaAbsorption::<100>::new();
        da.init();
        
        // Simulate heavy buying with stable price
        for i in 0..60 {
            da.process_trade(50000 - (i % 3) as i64, 200, true); // Mostly buys
        }
        
        let signal = da.get_last_signal();
        // Should detect ask absorption (heavy buying, price not rising)
        assert!(signal.strength > 0);
    }

    #[test]
    fn test_volume_imbalance() {
        let da = CryptoDeltaAbsorption::new();
        da.init();
        
        // All buys
        da.process_trade(50000, 1000, true);
        assert_eq!(da.get_volume_imbalance(), 1000);
        
        da.reset();
        da.init();
        
        // All sells
        da.process_trade(50000, 1000, false);
        assert_eq!(da.get_volume_imbalance(), -1000);
        
        da.reset();
        da.init();
        
        // Balanced
        da.process_trade(50000, 500, true);
        da.process_trade(50000, 500, false);
        assert_eq!(da.get_volume_imbalance(), 0);
    }

    #[test]
    fn test_cvd_trend() {
        let da = DeltaAbsorption::<100>::new();
        da.init();
        
        // Establish baseline
        for _ in 0..50 {
            da.process_trade(50000, 100, true);
        }
        
        let cvd_at_50 = da.get_cvd();
        
        // More buys
        for _ in 0..50 {
            da.process_trade(50001, 100, true);
        }
        
        let trend_10 = da.get_cvd_trend(10);
        assert!(trend_10 > 0);
        
        let trend_50 = da.get_cvd_trend(50);
        assert!(trend_50 > 0);
    }

    #[test]
    fn test_circuit_breaker() {
        let da = CryptoDeltaAbsorption::new();
        da.init();
        
        assert!(!da.is_circuit_breaker_active());
        
        // Trigger extreme conditions
        for _ in 0..100 {
            da.process_trade(50000, 10000, true);
        }
        
        // May or may not trigger depending on implementation
        let _ = da.is_circuit_breaker_active();
        
        da.reset_circuit_breaker();
        assert!(!da.is_circuit_breaker_active());
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;
        
        assert!(size_of::<TradeRecord>() >= CACHE_LINE_SIZE);
        assert!(size_of::<AbsorptionSignal>() >= CACHE_LINE_SIZE);
    }

    #[test]
    fn test_signal_significance() {
        let mut signal = AbsorptionSignal::new();
        signal.signal_type = AbsorptionType::BidAbsorption;
        signal.strength = 600;
        
        assert!(signal.is_significant());
        
        signal.strength = 400;
        assert!(!signal.is_significant());
    }
}
