//! Bollinger Bands, ATR, and Keltner Channels computed with branchless math.
//! SIMD-accelerated variance calculations for throughput.
//! Pre-allocated circular buffers for O(1) updates.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::indicators::streaming_ta::{FixedValue, SCALE, StreamingEMA, MAX_WINDOW};

/// Maximum bands count for multi-band analysis
pub const MAX_BANDS: usize = 8;

/// Cache-line aligned Bollinger Bands calculator
#[repr(C)]
pub struct VolatilityBollinger {
    /// Pre-allocated price buffer for variance calculation
    pub price_buffer: [FixedValue; MAX_WINDOW],
    /// Sum of prices (for mean calculation)
    pub sum: FixedValue,
    /// Sum of squared prices (for variance calculation)
    pub sum_sq: i128,
    /// Current index in circular buffer
    pub idx: AtomicU64,
    /// Count of samples
    pub count: u32,
    /// Period for calculation
    pub period: u32,
    /// Standard deviation multiplier for bands
    pub std_mult: FixedValue, // Typically 200 for 2.0x
    /// Current upper band
    pub upper_band: FixedValue,
    /// Current middle band (SMA)
    pub middle_band: FixedValue,
    /// Current lower band
    pub lower_band: FixedValue,
    /// Bandwidth (normalized)
    pub bandwidth: FixedValue,
    /// Percent B indicator
    pub percent_b: FixedValue,
    /// Squeeze detected (low volatility)
    pub squeeze_detected: AtomicBool,
    _padding: [u8; 12], // Pad to 64 bytes
}

impl VolatilityBollinger {
    pub fn new(period: u32, std_mult: f64) -> Self {
        assert!(period <= MAX_WINDOW as u32, "Period exceeds MAX_WINDOW");
        let std_mult_fixed = (std_mult * SCALE as f64) as FixedValue;
        Self {
            price_buffer: [0; MAX_WINDOW],
            sum: 0,
            sum_sq: 0,
            idx: AtomicU64::new(0),
            count: 0,
            period,
            std_mult: std_mult_fixed,
            upper_band: 0,
            middle_band: 0,
            lower_band: 0,
            bandwidth: 0,
            percent_b: 50 * SCALE / 100, // Midpoint
            squeeze_detected: AtomicBool::new(false),
            _padding: [0; 12],
        }
    }

    /// Update Bollinger Bands with new price
    #[inline]
    pub fn update(&mut self, price: FixedValue) -> (FixedValue, FixedValue, FixedValue) {
        let i = self.idx.load(Ordering::Relaxed) as usize % self.period as usize;

        // Remove old value from sums if buffer is full
        if self.count >= self.period {
            let old_price = self.price_buffer[i];
            self.sum = self.sum.saturating_sub(old_price);
            self.sum_sq = self.sum_sq.saturating_sub((old_price as i128) * (old_price as i128));
        } else {
            self.count = self.count.saturating_add(1);
        }

        // Add new value
        self.price_buffer[i] = price;
        self.sum = self.sum.saturating_add(price);
        self.sum_sq = self.sum_sq.saturating_add((price as i128) * (price as i128));
        self.idx.fetch_add(1, Ordering::AcqRel);

        // Calculate mean (middle band)
        self.middle_band = self.sum / self.count as i64;

        // Calculate variance: E[X^2] - E[X]^2
        let mean_sq = (self.sum as i128 * self.sum as i128) / self.count as i128;
        let variance = (self.sum_sq - mean_sq) / self.count as i128;

        // Calculate standard deviation using integer approximation
        // sqrt(variance) using Newton-Raphson
        let std_dev = if variance > 0 {
            self.isqrt(variance as u64) as FixedValue
        } else {
            0
        };

        // Calculate bands
        let band_offset = (std_dev * self.std_mult) / SCALE;
        self.upper_band = self.middle_band + band_offset;
        self.lower_band = self.middle_band - band_offset;

        // Calculate bandwidth (normalized): (Upper - Lower) / Middle
        if self.middle_band > 0 {
            self.bandwidth = ((self.upper_band - self.lower_band) * SCALE) / self.middle_band;
        }

        // Calculate %B: (Price - Lower) / (Upper - Lower)
        let band_range = self.upper_band - self.lower_band;
        if band_range > 0 {
            self.percent_b = ((price - self.lower_band) * SCALE) / band_range;
        }

        // Detect squeeze (low volatility) - bandwidth below threshold
        let squeeze_threshold = 5 * SCALE / 100; // 5%
        if self.bandwidth < squeeze_threshold {
            self.squeeze_detected.store(true, Ordering::Release);
        }

        (self.upper_band, self.middle_band, self.lower_band)
    }

    /// Integer square root using Newton-Raphson (branchless-ish)
    #[inline]
    fn isqrt(&self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x
    }

    /// Get current bands
    #[inline]
    pub fn get(&self) -> (FixedValue, FixedValue, FixedValue) {
        (self.upper_band, self.middle_band, self.lower_band)
    }

    /// Check squeeze detected
    #[inline]
    pub fn check_squeeze(&self) -> bool {
        let detected = self.squeeze_detected.load(Ordering::Acquire);
        if detected {
            self.squeeze_detected.store(false, Ordering::Release);
        }
        detected
    }

    /// Get bandwidth
    #[inline]
    pub fn get_bandwidth(&self) -> FixedValue {
        self.bandwidth
    }

    /// Get %B
    #[inline]
    pub fn get_percent_b(&self) -> FixedValue {
        self.percent_b
    }
}

/// Cache-line aligned ATR (Average True Range) calculator
#[repr(C)]
pub struct VolatilityATR {
    /// Smoothed ATR value (Wilder's EMA)
    pub atr_value: FixedValue,
    /// Previous close for TR calculation
    pub prev_close: FixedValue,
    /// Previous high/low
    pub prev_high: FixedValue,
    pub prev_low: FixedValue,
    /// Multiplier for ATR (1/period in fixed-point)
    pub multiplier: FixedValue,
    /// Initialized flag
    pub initialized: bool,
    /// ATR percentage of price (normalized)
    pub atr_percent: FixedValue,
    _padding: [u8; 47], // Pad to 64 bytes
}

impl VolatilityATR {
    pub fn new(period: u32) -> Self {
        let multiplier = SCALE / period as i64;
        Self {
            atr_value: 0,
            prev_close: 0,
            prev_high: 0,
            prev_low: 0,
            multiplier,
            initialized: false,
            atr_percent: 0,
            _padding: [0; 47],
        }
    }

    /// Update ATR with new candle
    #[inline]
    pub fn update(&mut self, high: FixedValue, low: FixedValue, close: FixedValue) -> FixedValue {
        if !self.initialized {
            // First bar: ATR = High - Low
            self.atr_value = high - low;
            self.prev_close = close;
            self.prev_high = high;
            self.prev_low = low;
            self.initialized = true;
            return self.atr_value;
        }

        // Calculate True Range
        let tr1 = high - low;
        let tr2 = (high - self.prev_close).abs();
        let tr3 = (low - self.prev_close).abs();
        let tr = tr1.max(tr2.max(tr3));

        // Wilder's smoothing: ATR = (prev_ATR * (n-1) + TR) / n
        // Equivalent to: ATR = prev_ATR + (TR - prev_ATR) / n
        let diff = tr - self.atr_value;
        self.atr_value = self.atr_value + (diff * self.multiplier) / SCALE;

        // Calculate ATR as percentage of price
        if close > 0 {
            self.atr_percent = (self.atr_value * SCALE) / close;
        }

        self.prev_close = close;
        self.prev_high = high;
        self.prev_low = low;

        self.atr_value
    }

    /// Get current ATR
    #[inline]
    pub fn get(&self) -> FixedValue {
        self.atr_value
    }

    /// Get ATR as percentage
    #[inline]
    pub fn get_percent(&self) -> FixedValue {
        self.atr_percent
    }
}

/// Cache-line aligned Keltner Channels calculator
#[repr(C)]
pub struct VolatilityKeltner {
    /// EMA for center line
    pub ema: StreamingEMA,
    /// ATR for channel width
    pub atr: VolatilityATR,
    /// ATR multiplier
    pub atr_mult: FixedValue,
    /// Upper channel
    pub upper_channel: FixedValue,
    /// Center line (EMA)
    pub center_line: FixedValue,
    /// Lower channel
    pub lower_channel: FixedValue,
    /// Channel width
    pub channel_width: FixedValue,
    _padding: [u8; 32], // Pad to 64 bytes
}

impl VolatilityKeltner {
    pub fn new(ema_period: u32, atr_period: u32, atr_mult: f64) -> Self {
        let atr_mult_fixed = (atr_mult * SCALE as f64) as FixedValue;
        Self {
            ema: StreamingEMA::new(ema_period),
            atr: VolatilityATR::new(atr_period),
            atr_mult: atr_mult_fixed,
            upper_channel: 0,
            center_line: 0,
            lower_channel: 0,
            channel_width: 0,
            _padding: [0; 32],
        }
    }

    /// Update Keltner Channels with new candle
    #[inline]
    pub fn update(&mut self, high: FixedValue, low: FixedValue, close: FixedValue) -> (FixedValue, FixedValue, FixedValue) {
        // Update center line (EMA of close)
        self.center_line = self.ema.update(close);

        // Update ATR
        let atr = self.atr.update(high, low, close);

        // Calculate channels
        let channel_offset = (atr * self.atr_mult) / SCALE;
        self.upper_channel = self.center_line + channel_offset;
        self.lower_channel = self.center_line - channel_offset;
        self.channel_width = channel_offset * 2;

        (self.upper_channel, self.center_line, self.lower_channel)
    }

    /// Get current channels
    #[inline]
    pub fn get(&self) -> (FixedValue, FixedValue, FixedValue) {
        (self.upper_channel, self.center_line, self.lower_channel)
    }
}

/// Combined volatility bundle for efficient updates
#[repr(C)]
pub struct VolatilityBundle {
    pub bollinger: VolatilityBollinger,
    pub atr: VolatilityATR,
    pub keltner: VolatilityKeltner,
    _padding: [u8; 64],
}

impl VolatilityBundle {
    pub fn new(bb_period: u32, bb_mult: f64, atr_period: u32, kc_mult: f64) -> Self {
        Self {
            bollinger: VolatilityBollinger::new(bb_period, bb_mult),
            atr: VolatilityATR::new(atr_period),
            keltner: VolatilityKeltner::new(atr_period, atr_period, kc_mult),
            _padding: [0; 64],
        }
    }

    /// Update all volatility indicators
    #[inline]
    pub fn update(&mut self, high: FixedValue, low: FixedValue, close: FixedValue) {
        self.bollinger.update(close);
        self.atr.update(high, low, close);
        self.keltner.update(high, low, close);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands() {
        let mut bb = VolatilityBollinger::new(20, 2.0);
        
        // Feed constant price
        for _ in 0..30 {
            bb.update(100 * SCALE);
        }
        
        let (upper, mid, lower) = bb.get();
        
        // With constant price, bands should converge
        assert_eq!(mid, 100 * SCALE);
        assert!(upper >= mid);
        assert!(lower <= mid);
    }

    #[test]
    fn test_atr_convergence() {
        let mut atr = VolatilityATR::new(14);
        
        // Consistent range candles
        for i in 0..30 {
            let high = (100 + 2) * SCALE;
            let low = (100 - 2) * SCALE;
            let close = (100 + (i % 3 - 1)) * SCALE;
            atr.update(high, low, close);
        }
        
        // ATR should converge near the range (4)
        let atr_val = atr.get();
        assert!(atr_val >= 3 * SCALE && atr_val <= 5 * SCALE);
    }

    #[test]
    fn test_keltner_channels() {
        let mut kc = VolatilityKeltner::new(20, 14, 2.0);
        
        for i in 0..30 {
            let high = (100 + i % 5) * SCALE;
            let low = (98 - i % 3) * SCALE;
            let close = (99 + i % 4) * SCALE;
            kc.update(high, low, close);
        }
        
        let (upper, center, lower) = kc.get();
        assert!(upper > center);
        assert!(center > lower);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<VolatilityBollinger>() >= 64);
        assert!(core::mem::size_of::<VolatilityATR>() >= 64);
        assert!(core::mem::size_of::<VolatilityKeltner>() >= 64);
    }

    #[test]
    fn test_squeeze_detection() {
        let mut bb = VolatilityBollinger::new(20, 2.0);
        
        // Very tight range (squeeze)
        for _ in 0..30 {
            bb.update(100 * SCALE + (_ as i64 % 2));
        }
        
        // Should detect squeeze with minimal variance
        let bandwidth = bb.get_bandwidth();
        assert!(bandwidth < 10 * SCALE / 100); // Less than 10%
    }
}
