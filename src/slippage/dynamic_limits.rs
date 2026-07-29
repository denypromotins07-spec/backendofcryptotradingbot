//! Dynamic Slippage Limits
//! 
//! Dynamic slippage tolerance bounds based on real-time volatility.
//! Adjusts execution limits adaptively based on market conditions,
//! order size, and toxicity metrics to minimize adverse selection.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

use crate::common::circular_buffer::CircularBuffer;
use crate::common::fixed_point::FixedPoint;

/// Slippage limit configuration
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlippageConfig {
    pub base_slippage_bps: FixedPoint,    // Base slippage tolerance
    pub vol_multiplier: FixedPoint,       // Multiplier for volatility adjustment
    pub size_impact_factor: FixedPoint,   // Factor for order size impact
    pub toxicity_threshold: FixedPoint,   // VPIN threshold for tightening
    pub max_slippage_bps: FixedPoint,     // Absolute maximum slippage
    pub min_slippage_bps: FixedPoint,     // Absolute minimum slippage
    _padding: [u8; 32],                   // Pad to 64 bytes
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            base_slippage_bps: FixedPoint::from_raw(10),   // 0.10% base
            vol_multiplier: FixedPoint::from_raw(200),     // 2x vol adjustment
            size_impact_factor: FixedPoint::from_raw(50),  // 0.5% per unit
            toxicity_threshold: FixedPoint::from_raw(7000), // 70% VPIN
            max_slippage_bps: FixedPoint::from_raw(100),   // 1% max
            min_slippage_bps: FixedPoint::from_raw(1),     // 0.01% min
            _padding: [0u8; 32],
        }
    }
}

/// Current slippage limit state
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlippageState {
    pub current_limit_bps: FixedPoint,
    pub volatility_adjusted_bps: FixedPoint,
    pub size_adjusted_bps: FixedPoint,
    pub toxicity_adjusted_bps: FixedPoint,
    pub last_update_cycle: u64,
    pub regime: u8,                        // 0=normal, 1=stressed, 2=crisis
    _padding: [u8; 39],                    // Pad to 64 bytes
}

impl Default for SlippageState {
    fn default() -> Self {
        Self {
            current_limit_bps: FixedPoint::ZERO,
            volatility_adjusted_bps: FixedPoint::ZERO,
            size_adjusted_bps: FixedPoint::ZERO,
            toxicity_adjusted_bps: FixedPoint::ZERO,
            last_update_cycle: 0,
            regime: 0,
            _padding: [0u8; 39],
        }
    }
}

/// Historical slippage sample for analysis
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlippageSample {
    pub timestamp_cycles: u64,
    pub requested_limit_bps: FixedPoint,
    pub actual_slippage_bps: FixedPoint,
    pub fill_rate_bps: FixedPoint,         // Percentage filled
    pub volatility: FixedPoint,
    pub toxicity: FixedPoint,
    _padding: [u8; 24],                    // Pad to 64 bytes
}

impl Default for SlippageSample {
    fn default() -> Self {
        Self {
            timestamp_cycles: 0,
            requested_limit_bps: FixedPoint::ZERO,
            actual_slippage_bps: FixedPoint::ZERO,
            fill_rate_bps: FixedPoint::ZERO,
            volatility: FixedPoint::ZERO,
            toxicity: FixedPoint::ZERO,
            _padding: [0u8; 24],
        }
    }
}

/// Shadow log entry for slippage decisions
#[repr(C)]
pub struct SlippageShadowLog {
    pub timestamp_cycles: u64,
    pub order_id: u64,
    pub computed_limit_bps: FixedPoint,
    pub applied_limit_bps: FixedPoint,
    pub override_reason: u8,               // 0=none, 1=min, 2=max, 3=halt
    _padding: [u8; 39],                    // Pad to 64 bytes
}

/// Dynamic slippage limits engine
#[repr(C)]
pub struct DynamicSlippageLimits {
    /// Configuration
    config: SlippageConfig,
    
    /// Current state
    state: SlippageState,
    
    /// Historical samples (pre-allocated circular buffer)
    history: CircularBuffer<SlippageSample, 4096>,
    
    /// Rolling statistics
    avg_slippage_bps: AtomicU64,           // EMA of actual slippage
    avg_fill_rate_bps: AtomicU64,          // EMA of fill rate
    vol_estimate: AtomicU64,               // Current vol estimate
    
    /// Shadow mode logging
    shadow_log: CircularBuffer<SlippageShadowLog, 2048>,
    shadow_enabled: AtomicU64,
    
    /// Circuit breaker
    halted: AtomicU64,
    stress_flag: AtomicU64,                // Set when in stressed regime
    
    _padding: [u8; 32],                    // Pad to cache line
}

impl DynamicSlippageLimits {
    pub const fn new() -> Self {
        Self {
            config: SlippageConfig {
                base_slippage_bps: FixedPoint::ZERO,
                vol_multiplier: FixedPoint::ZERO,
                size_impact_factor: FixedPoint::ZERO,
                toxicity_threshold: FixedPoint::ZERO,
                max_slippage_bps: FixedPoint::ZERO,
                min_slippage_bps: FixedPoint::ZERO,
                _padding: [0u8; 32],
            },
            state: SlippageState {
                current_limit_bps: FixedPoint::ZERO,
                volatility_adjusted_bps: FixedPoint::ZERO,
                size_adjusted_bps: FixedPoint::ZERO,
                toxicity_adjusted_bps: FixedPoint::ZERO,
                last_update_cycle: 0,
                regime: 0,
                _padding: [0u8; 39],
            },
            history: CircularBuffer::new(),
            avg_slippage_bps: AtomicU64::new(0),
            avg_fill_rate_bps: AtomicU64::new(0),
            vol_estimate: AtomicU64::new(0),
            shadow_log: CircularBuffer::new(),
            shadow_enabled: AtomicU64::new(0),
            halted: AtomicU64::new(0),
            stress_flag: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }
    
    /// Initialize with configuration
    #[inline]
    pub fn init(&mut self, config: SlippageConfig) {
        self.config = config;
    }
    
    /// Compute dynamic slippage limit for an order
    #[inline]
    pub fn compute_limit(
        &self,
        order_size: FixedPoint,
        current_volatility: FixedPoint,
        toxicity: FixedPoint,
    ) -> FixedPoint {
        if self.halted.load(Ordering::Acquire) != 0 {
            return FixedPoint::ZERO;
        }
        
        let config = unsafe { core::ptr::read_volatile(&self.config) };
        
        // Base limit
        let mut limit = config.base_slippage_bps;
        
        // Volatility adjustment (branchless)
        let vol_adjustment = current_volatility * config.vol_multiplier / FixedPoint::from_raw(10000);
        let vol_adjusted = limit + vol_adjustment;
        
        // Size impact adjustment
        let size_factor = order_size.to_raw().min(100_000_000) as i64; // Cap at reasonable size
        let size_adjustment = FixedPoint::from_raw(size_factor) * config.size_impact_factor / FixedPoint::from_raw(100000000);
        let size_adjusted = vol_adjusted + size_adjustment;
        
        // Toxicity adjustment (tighten limits when toxicity is high)
        let toxicity_ratio = toxicity * FixedPoint::from_raw(10000) / config.toxicity_threshold;
        let toxicity_factor = if toxicity_ratio > FixedPoint::from_raw(10000) {
            // Toxicity above threshold - tighten limits
            FixedPoint::from_raw(5000) // 0.5x multiplier
        } else {
            FixedPoint::from_raw(10000) // 1.0x multiplier
        };
        let toxicity_adjusted = size_adjusted * toxicity_factor / FixedPoint::from_raw(10000);
        
        // Apply bounds (branchless clamping)
        let bounded = toxicity_adjusted
            .max(config.min_slippage_bps)
            .min(config.max_slippage_bps);
        
        // Determine regime
        let regime = if toxicity > config.toxicity_threshold {
            2u8 // Crisis
        } else if current_volatility > FixedPoint::from_raw(5000) {
            1u8 // Stressed
        } else {
            0u8 // Normal
        };
        
        // Update state atomically (assumed single writer)
        unsafe {
            let new_state = SlippageState {
                current_limit_bps: bounded,
                volatility_adjusted_bps: vol_adjusted,
                size_adjusted_bps: size_adjusted,
                toxicity_adjusted_bps: toxicity_adjusted,
                last_update_cycle: self.read_rdtsc(),
                regime,
                _padding: [0u8; 39],
            };
            core::ptr::write_volatile(&mut self.state as *mut SlippageState, new_state);
        }
        
        // Set stress flag if needed
        if regime >= 1 {
            self.stress_flag.store(1, Ordering::Relaxed);
        } else {
            self.stress_flag.store(0, Ordering::Relaxed);
        }
        
        bounded
    }
    
    /// Record actual slippage outcome for learning
    #[inline]
    pub fn record_outcome(&self, sample: SlippageSample) {
        if self.halted.load(Ordering::Acquire) != 0 {
            return;
        }
        
        self.history.push(sample);
        
        // Update EMA of slippage (alpha = 0.1)
        let alpha = FixedPoint::from_raw(1000); // 0.1
        let one_minus_alpha = FixedPoint::from_raw(9000); // 0.9
        
        let current_avg = FixedPoint::from_raw(self.avg_slippage_bps.load(Ordering::Relaxed) as i64);
        let new_avg = (current_avg * one_minus_alpha + sample.actual_slippage_bps * alpha) / FixedPoint::from_raw(10000);
        self.avg_slippage_bps.store(new_avg.to_raw() as u64, Ordering::Relaxed);
        
        // Update EMA of fill rate
        let current_fill = FixedPoint::from_raw(self.avg_fill_rate_bps.load(Ordering::Relaxed) as i64);
        let new_fill = (current_fill * one_minus_alpha + sample.fill_rate_bps * alpha) / FixedPoint::from_raw(10000);
        self.avg_fill_rate_bps.store(new_fill.to_raw() as u64, Ordering::Relaxed);
        
        // Update vol estimate using Welford's algorithm
        let current_vol = self.vol_estimate.load(Ordering::Relaxed);
        let vol_raw = sample.volatility.to_raw() as u64;
        let new_vol = ((current_vol * 9 + vol_raw) / 10).max(1);
        self.vol_estimate.store(new_vol, Ordering::Relaxed);
    }
    
    /// Log slippage decision (shadow mode)
    #[inline]
    pub fn log_decision(&self, log_entry: SlippageShadowLog) {
        if self.shadow_enabled.load(Ordering::Relaxed) == 0 {
            return;
        }
        
        self.shadow_log.push(log_entry);
    }
    
    /// Get current slippage state
    #[inline]
    pub fn get_state(&self) -> SlippageState {
        unsafe { core::ptr::read_volatile(&self.state) }
    }
    
    /// Get average slippage
    #[inline]
    pub fn get_avg_slippage(&self) -> FixedPoint {
        FixedPoint::from_raw(self.avg_slippage_bps.load(Ordering::Relaxed) as i64)
    }
    
    /// Get average fill rate
    #[inline]
    pub fn get_avg_fill_rate(&self) -> FixedPoint {
        FixedPoint::from_raw(self.avg_fill_rate_bps.load(Ordering::Relaxed) as i64)
    }
    
    /// Halt slippage calculation
    #[inline]
    pub fn halt(&self) {
        self.halted.store(1, Ordering::SeqCst);
    }
    
    /// Resume slippage calculation
    #[inline]
    pub fn resume(&self) {
        self.halted.store(0, Ordering::SeqCst);
    }
    
    /// Enable shadow mode
    #[inline]
    pub fn enable_shadow_mode(&self) {
        self.shadow_enabled.store(1, Ordering::Relaxed);
    }
    
    /// Disable shadow mode
    #[inline]
    pub fn disable_shadow_mode(&self) {
        self.shadow_enabled.store(0, Ordering::Relaxed);
    }
    
    /// Check if in stress regime
    #[inline]
    pub fn is_stressed(&self) -> bool {
        self.stress_flag.load(Ordering::Acquire) != 0
    }
    
    /// Read timestamp counter
    #[inline]
    fn read_rdtsc(&self) -> u64 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::x86_64::_rdtsc()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_limits_initialization() {
        let limits = DynamicSlippageLimits::new();
        assert_eq!(limits.halted.load(Ordering::Relaxed), 0);
        assert_eq!(limits.stress_flag.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_limit_computation_normal() {
        let mut limits = DynamicSlippageLimits::new();
        limits.init(SlippageConfig::default());
        
        let limit = limits.compute_limit(
            FixedPoint::from_raw(1000000),  // 1M order size
            FixedPoint::from_raw(2000),     // 20% vol
            FixedPoint::from_raw(3000),     // 30% toxicity (low)
        );
        
        // Should be within bounds
        assert!(limit >= FixedPoint::from_raw(1));
        assert!(limit <= FixedPoint::from_raw(100));
    }
    
    #[test]
    fn test_limit_computation_stressed() {
        let mut limits = DynamicSlippageLimits::new();
        limits.init(SlippageConfig::default());
        
        let limit = limits.compute_limit(
            FixedPoint::from_raw(1000000),
            FixedPoint::from_raw(8000),     // 80% vol (high)
            FixedPoint::from_raw(8000),     // 80% toxicity (high)
        );
        
        // Should be tightened due to high toxicity
        assert!(limits.is_stressed());
    }
    
    #[test]
    fn test_outcome_recording() {
        let limits = DynamicSlippageLimits::new();
        
        let sample = SlippageSample {
            timestamp_cycles: 12345,
            requested_limit_bps: FixedPoint::from_raw(50),
            actual_slippage_bps: FixedPoint::from_raw(45),
            fill_rate_bps: FixedPoint::from_raw(9500),
            volatility: FixedPoint::from_raw(2000),
            toxicity: FixedPoint::from_raw(3000),
            _padding: [0u8; 24],
        };
        
        limits.record_outcome(sample);
        assert_eq!(limits.history.len(), 1);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let limits = DynamicSlippageLimits::new();
        
        limits.halt();
        assert_eq!(limits.halted.load(Ordering::Acquire), 1);
        
        let limit = limits.compute_limit(
            FixedPoint::from_raw(1000000),
            FixedPoint::from_raw(2000),
            FixedPoint::from_raw(3000),
        );
        assert_eq!(limit, FixedPoint::ZERO);
        
        limits.resume();
        assert_eq!(limits.halted.load(Ordering::Acquire), 0);
    }
}
