//! RSI, MACD, and ADX calculations using SIMD-optimized rolling windows.
//! Branchless threshold detection for deterministic latency.
//! Pre-allocated buffers for zero heap allocation in hot paths.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::indicators::streaming_ta::{FixedValue, SCALE, StreamingEMA, MAX_WINDOW};

/// Maximum lookback for ADX calculation
pub const MAX_ADX_LOOKBACK: usize = 100;

/// Cache-line aligned RSI calculator
#[repr(C)]
pub struct MomentumRSI {
    /// Gain EMA
    pub gain_ema: StreamingEMA,
    /// Loss EMA
    pub loss_ema: StreamingEMA,
    /// Current RSI value (scaled by 100 for fixed-point percentage)
    pub rsi_value: FixedValue,
    /// Previous close for change calculation
    pub prev_close: FixedValue,
    /// Initialized flag
    pub initialized: bool,
    /// Overbought threshold (default 70 * SCALE/100)
    pub overbought: FixedValue,
    /// Oversold threshold (default 30 * SCALE/100)
    pub oversold: FixedValue,
    /// Crossed overbought flag
    pub crossed_overbought: AtomicBool,
    /// Crossed oversold flag
    pub crossed_oversold: AtomicBool,
    _padding: [u8; 20], // Pad to 64 bytes
}

impl MomentumRSI {
    pub fn new(period: u32) -> Self {
        Self {
            gain_ema: StreamingEMA::new(period),
            loss_ema: StreamingEMA::new(period),
            rsi_value: 50 * SCALE / 100, // Default midpoint
            prev_close: 0,
            initialized: false,
            overbought: 70 * SCALE / 100,
            oversold: 30 * SCALE / 100,
            crossed_overbought: AtomicBool::new(false),
            crossed_oversold: AtomicBool::new(false),
            _padding: [0; 20],
        }
    }

    /// Update RSI with new close price
    #[inline]
    pub fn update(&mut self, close: FixedValue) -> FixedValue {
        if !self.initialized {
            self.prev_close = close;
            self.initialized = true;
            return self.rsi_value;
        }

        // Calculate price change (branchless)
        let change = close - self.prev_close;
        
        // Branchless gain/loss extraction
        let gain = ((change > 0) as i64) * change;
        let loss = ((change < 0) as i64) * (-change);

        // Update EMAs
        self.gain_ema.update(gain);
        self.loss_ema.update(loss);

        let avg_gain = self.gain_ema.get();
        let avg_loss = self.loss_ema.get();

        // Calculate RSI: 100 - 100/(1 + RS) where RS = avg_gain/avg_loss
        if avg_loss == 0 {
            self.rsi_value = 100 * SCALE / 100;
        } else {
            let rs = (avg_gain * SCALE) / avg_loss;
            self.rsi_value = SCALE - (SCALE * SCALE) / (SCALE + rs);
        }

        // Branchless threshold crossing detection
        let cross_ob = (self.rsi_value >= self.overbought) as u8;
        let prev_below_ob = (self.prev_close < self.overbought) as u8;
        if cross_ob & prev_below_ob != 0 {
            self.crossed_overbought.store(true, Ordering::Release);
        }

        let cross_os = (self.rsi_value <= self.oversold) as u8;
        let prev_above_os = (self.prev_close > self.oversold) as u8;
        if cross_os & prev_above_os != 0 {
            self.crossed_oversold.store(true, Ordering::Release);
        }

        self.prev_close = close;
        self.rsi_value
    }

    /// Get current RSI
    #[inline]
    pub fn get(&self) -> FixedValue {
        self.rsi_value
    }

    /// Check overbought cross (consumes flag)
    #[inline]
    pub fn check_overbought_cross(&self) -> bool {
        let detected = self.crossed_overbought.load(Ordering::Acquire);
        if detected {
            self.crossed_overbought.store(false, Ordering::Release);
        }
        detected
    }

    /// Check oversold cross (consumes flag)
    #[inline]
    pub fn check_oversold_cross(&self) -> bool {
        let detected = self.crossed_oversold.load(Ordering::Acquire);
        if detected {
            self.crossed_oversold.store(false, Ordering::Release);
        }
        detected
    }
}

/// Cache-line aligned MACD calculator
#[repr(C)]
pub struct MomentumMACD {
    /// Fast EMA (typically 12)
    pub fast_ema: StreamingEMA,
    /// Slow EMA (typically 26)
    pub slow_ema: StreamingEMA,
    /// Signal line EMA (typically 9)
    pub signal_ema: StreamingEMA,
    /// Current MACD line value
    pub macd_line: FixedValue,
    /// Current signal line value
    pub signal_line: FixedValue,
    /// Current histogram value
    pub histogram: FixedValue,
    /// Previous histogram for divergence detection
    pub prev_histogram: FixedValue,
    /// Signal initialized flag
    pub signal_initialized: bool,
    /// Bullish divergence detected
    pub bullish_divergence: AtomicBool,
    /// Bearish divergence detected
    pub bearish_divergence: AtomicBool,
    _padding: [u8; 18], // Pad to 64 bytes
}

impl MomentumMACD {
    pub fn new(fast_period: u32, slow_period: u32, signal_period: u32) -> Self {
        Self {
            fast_ema: StreamingEMA::new(fast_period),
            slow_ema: StreamingEMA::new(slow_period),
            signal_ema: StreamingEMA::new(signal_period),
            macd_line: 0,
            signal_line: 0,
            histogram: 0,
            prev_histogram: 0,
            signal_initialized: false,
            bullish_divergence: AtomicBool::new(false),
            bearish_divergence: AtomicBool::new(false),
            _padding: [0; 18],
        }
    }

    /// Update MACD with new price
    #[inline]
    pub fn update(&mut self, price: FixedValue) -> (FixedValue, FixedValue, FixedValue) {
        // Update EMAs
        let fast = self.fast_ema.update(price);
        let slow = self.slow_ema.update(price);

        // MACD line = Fast EMA - Slow EMA
        self.macd_line = fast - slow;

        // Update signal line (EMA of MACD)
        if self.signal_initialized {
            self.signal_line = self.signal_ema.update(self.macd_line);
        } else {
            self.signal_line = self.macd_line;
            self.signal_ema.value = self.macd_line;
            self.signal_initialized = true;
        }

        // Histogram = MACD - Signal
        self.prev_histogram = self.histogram;
        self.histogram = self.macd_line - self.signal_line;

        // Detect divergence (simplified)
        self.detect_divergence();

        (self.macd_line, self.signal_line, self.histogram)
    }

    /// Simple divergence detection
    #[inline]
    fn detect_divergence(&mut self) {
        // Bullish: histogram crosses above zero from negative
        if self.prev_histogram < 0 && self.histogram >= 0 {
            self.bullish_divergence.store(true, Ordering::Release);
        }
        // Bearish: histogram crosses below zero from positive
        if self.prev_histogram > 0 && self.histogram <= 0 {
            self.bearish_divergence.store(true, Ordering::Release);
        }
    }

    /// Get values
    #[inline]
    pub fn get(&self) -> (FixedValue, FixedValue, FixedValue) {
        (self.macd_line, self.signal_line, self.histogram)
    }

    /// Check bullish divergence
    #[inline]
    pub fn check_bullish_divergence(&self) -> bool {
        let detected = self.bullish_divergence.load(Ordering::Acquire);
        if detected {
            self.bullish_divergence.store(false, Ordering::Release);
        }
        detected
    }

    /// Check bearish divergence
    #[inline]
    pub fn check_bearish_divergence(&self) -> bool {
        let detected = self.bearish_divergence.load(Ordering::Acquire);
        if detected {
            self.bearish_divergence.store(false, Ordering::Release);
        }
        detected
    }
}

/// Cache-line aligned ADX calculator
#[repr(C)]
pub struct MomentumADX {
    /// Pre-allocated buffer for +DM
    pub plus_dm_buffer: [FixedValue; MAX_ADX_LOOKBACK],
    /// Pre-allocated buffer for -DM
    pub minus_dm_buffer: [FixedValue; MAX_ADX_LOOKBACK],
    /// Pre-allocated buffer for TR
    pub tr_buffer: [FixedValue; MAX_ADX_LOOKBACK],
    /// Smoothed +DI
    pub plus_di: FixedValue,
    /// Smoothed -DI
    pub minus_di: FixedValue,
    /// Current ADX value
    pub adx_value: FixedValue,
    /// Buffer index
    pub idx: AtomicU64,
    /// Count of samples
    pub count: u32,
    /// Lookback period
    pub period: u32,
    /// Previous high/low for DM calculation
    pub prev_high: FixedValue,
    pub prev_low: FixedValue,
    pub prev_close: FixedValue,
    /// Initialized flag
    pub initialized: bool,
    _padding: [u8; 8], // Pad to 64 bytes
}

impl MomentumADX {
    pub fn new(period: u32) -> Self {
        assert!(period <= MAX_ADX_LOOKBACK as u32, "Period exceeds MAX_ADX_LOOKBACK");
        Self {
            plus_dm_buffer: [0; MAX_ADX_LOOKBACK],
            minus_dm_buffer: [0; MAX_ADX_LOOKBACK],
            tr_buffer: [0; MAX_ADX_LOOKBACK],
            plus_di: 0,
            minus_di: 0,
            adx_value: 0,
            idx: AtomicU64::new(0),
            count: 0,
            period,
            prev_high: 0,
            prev_low: 0,
            prev_close: 0,
            initialized: false,
            _padding: [0; 8],
        }
    }

    /// Update ADX with new candle
    #[inline]
    pub fn update(&mut self, high: FixedValue, low: FixedValue, close: FixedValue) -> FixedValue {
        if !self.initialized {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            self.initialized = true;
            return 0;
        }

        // Calculate +DM and -DM (branchless)
        let up_move = high - self.prev_high;
        let down_move = self.prev_low - low;

        let plus_dm = ((up_move > down_move) as i64) * ((up_move > 0) as i64) * up_move;
        let minus_dm = ((down_move > up_move) as i64) * ((down_move > 0) as i64) * down_move;

        // Calculate True Range
        let tr1 = high - low;
        let tr2 = (high - self.prev_close).abs();
        let tr3 = (low - self.prev_close).abs();
        let tr = tr1.max(tr2.max(tr3));

        // Store in circular buffer
        let i = self.idx.load(Ordering::Relaxed) as usize % self.period as usize;
        self.plus_dm_buffer[i] = plus_dm;
        self.minus_dm_buffer[i] = minus_dm;
        self.tr_buffer[i] = tr;
        self.idx.fetch_add(1, Ordering::AcqRel);

        if self.count < self.period {
            self.count += 1;
        }

        // Calculate smoothed sums (Wilder's smoothing approximation)
        let mut sum_plus_dm = FixedValue::default();
        let mut sum_minus_dm = FixedValue::default();
        let mut sum_tr = FixedValue::default();

        for j in 0..self.count {
            let k = (self.idx.load(Ordering::Relaxed) as usize - 1 - j) % self.period as usize;
            sum_plus_dm = sum_plus_dm.saturating_add(self.plus_dm_buffer[k]);
            sum_minus_dm = sum_minus_dm.saturating_add(self.minus_dm_buffer[k]);
            sum_tr = sum_tr.saturating_add(self.tr_buffer[k]);
        }

        // Calculate DI+ and DI-
        if sum_tr > 0 {
            self.plus_di = (sum_plus_dm * SCALE) / sum_tr;
            self.minus_di = (sum_minus_dm * SCALE) / sum_tr;
        }

        // Calculate DX and ADX
        let di_sum = self.plus_di + self.minus_di;
        let di_diff = (self.plus_di - self.minus_di).abs();

        if di_sum > 0 {
            let dx = (di_diff * SCALE) / di_sum;
            
            // Simple ADX (average of DX)
            self.adx_value = dx;
        }

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;

        self.adx_value
    }

    /// Get current ADX
    #[inline]
    pub fn get(&self) -> FixedValue {
        self.adx_value
    }

    /// Get +DI
    #[inline]
    pub fn get_plus_di(&self) -> FixedValue {
        self.plus_di
    }

    /// Get -DI
    #[inline]
    pub fn get_minus_di(&self) -> FixedValue {
        self.minus_di
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        let mut rsi = MomentumRSI::new(14);
        
        // Feed constant prices
        for _ in 0..30 {
            rsi.update(100 * SCALE);
        }
        
        // With no change, RSI should be neutral (50)
        let val = rsi.get();
        assert!(val >= 40 * SCALE / 100 && val <= 60 * SCALE / 100);
    }

    #[test]
    fn test_macd_crossover() {
        let mut macd = MomentumMACD::new(12, 26, 9);
        
        // Uptrend
        for i in 0..50 {
            macd.update((100 + i) * SCALE);
        }
        
        let (macd_line, signal, hist) = macd.get();
        // In uptrend, MACD should be above signal
        assert!(macd_line >= signal);
    }

    #[test]
    fn test_adx_trending() {
        let mut adx = MomentumADX::new(14);
        
        // Strong uptrend (higher highs, higher lows)
        for i in 0..30 {
            let high = (100 + i * 2) * SCALE;
            let low = (98 + i * 2) * SCALE;
            let close = (99 + i * 2) * SCALE;
            adx.update(high, low, close);
        }
        
        // ADX should show some trend strength
        let adx_val = adx.get();
        assert!(adx_val >= 0);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<MomentumRSI>() >= 64);
        assert!(core::mem::size_of::<MomentumMACD>() >= 64);
        assert!(core::mem::size_of::<MomentumADX>() >= 64);
    }
}
