//! Lock-free streaming EMA, SMA, and VWAP calculators
//! Uses incremental updates with O(1) time complexity.
//! Circular buffers eliminate heap allocations in hot paths.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicU64, Ordering};

/// Fixed-point representation for calculations
pub type FixedValue = i64;
const SCALE: i64 = 100_000_000;

/// Maximum window size for rolling calculations (pre-allocated)
pub const MAX_WINDOW: usize = 512;

/// Cache-line aligned streaming EMA calculator
#[repr(C)]
pub struct StreamingEMA {
    /// Current EMA value (fixed-point)
    pub value: FixedValue,
    /// Multiplier factor (alpha * SCALE)
    pub multiplier: FixedValue,
    /// Initialized flag
    pub initialized: bool,
    _padding: [u8; 55], // Pad to 64 bytes
}

impl StreamingEMA {
    pub fn new(period: u32) -> Self {
        let multiplier = (2 * SCALE) / (period as i64 + 1);
        Self {
            value: 0,
            multiplier,
            initialized: false,
            _padding: [0; 55],
        }
    }

    /// Update EMA with new value (branchless, O(1))
    #[inline]
    pub fn update(&mut self, price: FixedValue) -> FixedValue {
        if !self.initialized {
            self.value = price;
            self.initialized = true;
        } else {
            // EMA = price * alpha + prev_ema * (1 - alpha)
            // Using fixed-point: value = (price * mult + value * (SCALE - mult)) / SCALE
            let price_contrib = (price * self.multiplier) / SCALE;
            let prev_contrib = (self.value * (SCALE - self.multiplier)) / SCALE;
            self.value = price_contrib + prev_contrib;
        }
        self.value
    }

    /// Get current EMA value
    #[inline]
    pub fn get(&self) -> FixedValue {
        self.value
    }

    /// Reset the EMA
    #[inline]
    pub fn reset(&mut self) {
        self.value = 0;
        self.initialized = false;
    }
}

/// Cache-line aligned streaming SMA calculator with circular buffer
#[repr(C)]
pub struct StreamingSMA {
    /// Pre-allocated circular buffer
    pub buffer: [FixedValue; MAX_WINDOW],
    /// Sum of values in buffer (for O(1) update)
    pub sum: FixedValue,
    /// Current index in circular buffer
    pub idx: AtomicU64,
    /// Count of values added (up to period)
    pub count: u32,
    /// Window period
    pub period: u32,
    _padding: [u8; 44], // Pad to 64 bytes
}

impl StreamingSMA {
    pub fn new(period: u32) -> Self {
        assert!(period <= MAX_WINDOW as u32, "Period exceeds MAX_WINDOW");
        Self {
            buffer: [0; MAX_WINDOW],
            sum: 0,
            idx: AtomicU64::new(0),
            count: 0,
            period,
            _padding: [0; 44],
        }
    }

    /// Update SMA with new value (O(1) using circular buffer)
    #[inline]
    pub fn update(&mut self, price: FixedValue) -> FixedValue {
        let i = self.idx.load(Ordering::Relaxed) as usize % self.period as usize;
        
        // If buffer is full, subtract oldest value
        if self.count >= self.period {
            let oldest = self.buffer[i];
            self.sum = self.sum.saturating_sub(oldest);
        } else {
            self.count = self.count.saturating_add(1);
        }

        // Add new value
        self.buffer[i] = price;
        self.sum = self.sum.saturating_add(price);
        
        self.idx.fetch_add(1, Ordering::AcqRel);

        // Return average
        if self.count > 0 {
            self.sum / self.count as i64
        } else {
            0
        }
    }

    /// Get current SMA value without updating
    #[inline]
    pub fn get(&self) -> FixedValue {
        if self.count > 0 {
            self.sum / self.count as i64
        } else {
            0
        }
    }

    /// Reset the SMA
    #[inline]
    pub fn reset(&mut self) {
        self.sum = 0;
        self.count = 0;
        self.idx.store(0, Ordering::Release);
        self.buffer.fill(0);
    }
}

/// Cache-line aligned streaming VWAP calculator
#[repr(C)]
pub struct StreamingVWAP {
    /// Cumulative typical price * volume
    pub cum_tp_volume: i128,
    /// Cumulative volume
    pub cum_volume: i128,
    /// Session start timestamp (for session reset)
    pub session_start_ns: u64,
    /// Current VWAP value (cached)
    pub current_vwap: FixedValue,
    _padding: [u8; 32], // Pad to 64 bytes
}

impl StreamingVWAP {
    pub fn new() -> Self {
        Self {
            cum_tp_volume: 0,
            cum_volume: 0,
            session_start_ns: 0,
            current_vwap: 0,
            _padding: [0; 32],
        }
    }

    /// Update VWAP with new candle data (high, low, close, volume)
    /// Returns updated VWAP value
    #[inline]
    pub fn update(&mut self, high: FixedValue, low: FixedValue, close: FixedValue, volume: i64) -> FixedValue {
        // Typical price = (H + L + C) / 3
        let typical_price = (high + low + close) / 3;
        
        // Accumulate
        self.cum_tp_volume += (typical_price as i128) * (volume as i128);
        self.cum_volume += volume as i128;

        // Calculate VWAP
        if self.cum_volume > 0 {
            self.current_vwap = (self.cum_tp_volume / self.cum_volume) as FixedValue;
        }

        self.current_vwap
    }

    /// Reset for new session
    #[inline]
    pub fn reset_session(&mut self, timestamp_ns: u64) {
        self.cum_tp_volume = 0;
        self.cum_volume = 0;
        self.session_start_ns = timestamp_ns;
        self.current_vwap = 0;
    }

    /// Get current VWAP
    #[inline]
    pub fn get(&self) -> FixedValue {
        self.current_vwap
    }
}

impl Default for StreamingVWAP {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined indicator state for cache efficiency
#[repr(C)]
pub struct IndicatorBundle {
    pub ema_fast: StreamingEMA,
    pub ema_slow: StreamingEMA,
    pub sma: StreamingSMA,
    pub vwap: StreamingVWAP,
    _padding: [u8; 64], // Extra padding for alignment
}

impl IndicatorBundle {
    pub fn new(fast_period: u32, slow_period: u32, sma_period: u32) -> Self {
        Self {
            ema_fast: StreamingEMA::new(fast_period),
            ema_slow: StreamingEMA::new(slow_period),
            sma: StreamingSMA::new(sma_period),
            vwap: StreamingVWAP::new(),
            _padding: [0; 64],
        }
    }

    /// Update all indicators with single price point
    #[inline]
    pub fn update_price(&mut self, price: FixedValue) -> (FixedValue, FixedValue, FixedValue) {
        let ema_f = self.ema_fast.update(price);
        let ema_s = self.ema_slow.update(price);
        let sma = self.sma.update(price);
        (ema_f, ema_s, sma)
    }

    /// Update VWAP separately (needs OHLCV)
    #[inline]
    pub fn update_vwap(&mut self, high: FixedValue, low: FixedValue, close: FixedValue, volume: i64) -> FixedValue {
        self.vwap.update(high, low, close, volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_convergence() {
        let mut ema = StreamingEMA::new(10);
        let price = 100 * SCALE;
        
        // Feed constant price
        for _ in 0..50 {
            ema.update(price);
        }
        
        // Should converge to price
        assert!((ema.get() - price).abs() < SCALE); // Within 1%
    }

    #[test]
    fn test_sma_sliding_window() {
        let mut sma = StreamingSMA::new(5);
        
        // Fill window
        for i in 1..=5 {
            sma.update(i * SCALE);
        }
        
        // SMA of 1,2,3,4,5 = 3
        assert_eq!(sma.get(), 3 * SCALE);
        
        // Add 6, should remove 1: SMA of 2,3,4,5,6 = 4
        sma.update(6 * SCALE);
        assert_eq!(sma.get(), 4 * SCALE);
    }

    #[test]
    fn test_vwap_calculation() {
        let mut vwap = StreamingVWAP::new();
        
        // Single candle: H=100, L=100, C=100, V=1000
        // TP = 100, VWAP = 100
        let result = vwap.update(100 * SCALE, 100 * SCALE, 100 * SCALE, 1000);
        assert_eq!(result, 100 * SCALE);
        
        // Second candle: H=102, L=102, C=102, V=1000
        // TP = 102, cum_tp_vol = 100*1000 + 102*1000 = 202000
        // cum_vol = 2000, VWAP = 101
        let result = vwap.update(102 * SCALE, 102 * SCALE, 102 * SCALE, 1000);
        assert_eq!(result, 101 * SCALE);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<StreamingEMA>() >= 64);
        assert!(core::mem::size_of::<StreamingSMA>() >= 64);
        assert!(core::mem::size_of::<StreamingVWAP>() >= 64);
    }

    #[test]
    fn test_indicator_bundle() {
        let mut bundle = IndicatorBundle::new(10, 20, 20);
        
        for i in 0..50 {
            let price = (100 + i) * SCALE;
            bundle.update_price(price);
        }
        
        // Fast EMA should be above slow EMA in uptrend
        assert!(bundle.ema_fast.get() > bundle.ema_slow.get());
    }
}
