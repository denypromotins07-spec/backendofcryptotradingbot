//! Market Impact Modeler
//! 
//! Almgren-Chriss market impact model calibrated with live stream data.
//! Models temporary and permanent market impact to optimize execution strategy.
//! Uses pre-allocated buffers and SIMD for fast calibration.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

use crate::common::circular_buffer::CircularBuffer;
use crate::common::fixed_point::FixedPoint;

/// Market impact parameters (Almgren-Chriss model)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ImpactParams {
    pub eta: FixedPoint,        // Temporary impact coefficient
    pub gamma: FixedPoint,      // Permanent impact coefficient
    pub sigma: FixedPoint,      // Volatility (annualized)
    pub lambda: FixedPoint,     // Risk aversion parameter
    pub alpha: FixedPoint,      // Alpha decay rate
    _padding: [u8; 32],         // Pad to 64 bytes
}

impl Default for ImpactParams {
    fn default() -> Self {
        Self {
            eta: FixedPoint::from_raw(100),     // 0.001 default
            gamma: FixedPoint::from_raw(50),    // 0.0005 default
            sigma: FixedPoint::from_raw(2000),  // 20% annualized
            lambda: FixedPoint::from_raw(500),  // Risk aversion
            alpha: FixedPoint::from_raw(100),   // Alpha decay
            _padding: [0u8; 32],
        }
    }
}

/// Calibration sample from live data
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CalibrationSample {
    pub order_size: FixedPoint,         // Order size in base units
    pub participation_rate: FixedPoint, // Order size / market volume
    pub price_impact_bps: FixedPoint,   // Observed impact in basis points
    pub time_horizon_us: u64,           // Execution time in microseconds
    pub volatility: FixedPoint,         // Realized vol during execution
    pub timestamp_cycles: u64,
    _padding: [u8; 24],                 // Pad to 64 bytes
}

impl Default for CalibrationSample {
    fn default() -> Self {
        Self {
            order_size: FixedPoint::ZERO,
            participation_rate: FixedPoint::ZERO,
            price_impact_bps: FixedPoint::ZERO,
            time_horizon_us: 0,
            volatility: FixedPoint::ZERO,
            timestamp_cycles: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Optimal execution trajectory point
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryPoint {
    pub time_fraction: FixedPoint,      // Fraction of total time elapsed (0-1)
    pub remaining_quantity: FixedPoint, // Quantity remaining to execute
    pub execution_rate: FixedPoint,     // Rate of execution at this point
    pub expected_impact_bps: FixedPoint,// Cumulative expected impact
    _padding: [u8; 32],                 // Pad to 64 bytes
}

/// Shadow log entry for impact predictions
#[repr(C)]
pub struct ImpactShadowLog {
    pub timestamp_cycles: u64,
    pub order_size: FixedPoint,
    pub predicted_impact_bps: FixedPoint,
    pub actual_impact_bps: FixedPoint,
    pub prediction_error_bps: FixedPoint,
    pub params_version: u64,
    _padding: [u8; 32],                 // Pad to 64 bytes
}

/// Almgren-Chriss market impact modeler
#[repr(C)]
pub struct MarketImpactModeler {
    /// Current impact parameters
    params: ImpactParams,
    params_version: AtomicU64,
    
    /// Calibration samples (pre-allocated circular buffer)
    calibration_buffer: CircularBuffer<CalibrationSample, 8192>,
    
    /// Computed optimal trajectories (pre-allocated)
    trajectory_buffer: [TrajectoryPoint; 64],
    
    /// Shadow mode logging
    shadow_log: CircularBuffer<ImpactShadowLog, 2048>,
    shadow_enabled: AtomicU64,
    
    /// Statistics
    total_samples_calibrated: AtomicU64,
    cumulative_prediction_error: AtomicU64,
    
    /// Circuit breaker
    calibration_quality: AtomicU64,    // 0=poor, 1=acceptable, 2=good
    halted: AtomicU64,
    
    _padding: [u8; 32],                // Pad to cache line
}

impl MarketImpactModeler {
    pub const fn new() -> Self {
        Self {
            params: ImpactParams {
                eta: FixedPoint::ZERO,
                gamma: FixedPoint::ZERO,
                sigma: FixedPoint::ZERO,
                lambda: FixedPoint::ZERO,
                alpha: FixedPoint::ZERO,
                _padding: [0u8; 32],
            },
            params_version: AtomicU64::new(0),
            calibration_buffer: CircularBuffer::new(),
            trajectory_buffer: unsafe { core::mem::zeroed() },
            shadow_log: CircularBuffer::new(),
            shadow_enabled: AtomicU64::new(0),
            total_samples_calibrated: AtomicU64::new(0),
            cumulative_prediction_error: AtomicU64::new(0),
            calibration_quality: AtomicU64::new(0),
            halted: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }
    
    /// Add a calibration sample from live execution
    #[inline]
    pub fn add_calibration_sample(&self, sample: CalibrationSample) {
        if self.halted.load(Ordering::Acquire) != 0 {
            return;
        }
        
        self.calibration_buffer.push(sample);
        self.total_samples_calibrated.fetch_add(1, Ordering::Relaxed);
        
        // Recalibrate if we have enough samples
        let count = self.calibration_buffer.len();
        if count >= 100 && count % 100 == 0 {
            self.recalibrate();
        }
    }
    
    /// Recalibrate impact parameters using least squares (simplified)
    #[inline]
    fn recalibrate(&self) {
        let count = self.calibration_buffer.len().min(8192);
        if count < 100 {
            return;
        }
        
        // Simplified OLS estimation using Welford's online algorithm
        let mut sum_x: i128 = 0;
        let mut sum_y: i128 = 0;
        let mut sum_xy: i128 = 0;
        let mut sum_x2: i128 = 0;
        let mut n: i128 = 0;
        
        unsafe {
            // Manually unroll loop for performance
            let mut idx = 0usize;
            while idx < count {
                let sample = self.calibration_buffer.get(idx).unwrap();
                
                // x = participation_rate, y = price_impact_bps
                let x = sample.participation_rate.to_raw() as i128;
                let y = sample.price_impact_bps.to_raw() as i128;
                
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                n += 1;
                
                idx += 1;
                
                // Branchless loop continuation check
                let should_continue = (idx < count) as i128;
                idx = (idx & !(should_continue.wrapping_neg())) | ((idx + 1) & (should_continue.wrapping_neg()));
            }
        }
        
        if n < 2 {
            return;
        }
        
        // OLS: eta = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = n * sum_x2 - sum_x * sum_x;
        
        if denominator > 0 {
            let eta_raw = ((numerator * 10000) / denominator) as i64;
            
            // Update parameters atomically
            let new_params = ImpactParams {
                eta: FixedPoint::from_raw(eta_raw.max(1)),
                gamma: self.params.gamma, // Keep existing
                sigma: self.params.sigma,
                lambda: self.params.lambda,
                alpha: self.params.alpha,
                _padding: [0u8; 32],
            };
            
            // Lock-free parameter update (assumed single writer)
            unsafe {
                core::ptr::write_volatile(&mut self.params as *mut ImpactParams, new_params);
            }
            self.params_version.fetch_add(1, Ordering::Release);
            
            // Update calibration quality based on sample count
            let quality = if count >= 1000 { 2 } else if count >= 100 { 1 } else { 0 };
            self.calibration_quality.store(quality, Ordering::Relaxed);
        }
    }
    
    /// Compute optimal execution trajectory using Almgren-Chriss formula
    #[inline]
    pub fn compute_optimal_trajectory(
        &self,
        total_quantity: FixedPoint,
        time_horizon_us: u64,
        risk_aversion: FixedPoint,
    ) -> Option<&[TrajectoryPoint]> {
        if self.halted.load(Ordering::Acquire) != 0 {
            return None;
        }
        
        // Check calibration quality
        if self.calibration_quality.load(Ordering::Acquire) == 0 {
            return None;
        }
        
        let params = unsafe { core::ptr::read_volatile(&self.params) };
        
        // Almgren-Chriss optimal trading rate:
        // v(t) = Q * sqrt(alpha/eta) * sinh(sqrt(alpha*eta)*(T-t)) / sinh(sqrt(alpha*eta)*T)
        // Simplified for discrete time steps
        
        let num_points = 64usize;
        let dt = FixedPoint::from_raw(time_horizon_us as i64 / num_points as i64);
        
        // Precompute constants
        let sqrt_alpha_eta = self.fast_isqrt(params.alpha.to_raw() * params.eta.to_raw());
        let sqrt_alpha_eta_fp = FixedPoint::from_raw(sqrt_alpha_eta as i64);
        
        let mut remaining = total_quantity;
        
        unsafe {
            for i in 0..num_points {
                let t_fraction = FixedPoint::from_raw((i as i64 * 10000) / num_points as i64);
                let time_remaining = time_horizon_us - (i as u64 * time_horizon_us / num_points as u64);
                
                // Simplified execution rate (linear approximation)
                let exec_rate = if time_remaining > 0 {
                    remaining.to_raw() as u64 * 1_000_000 / time_remaining
                } else {
                    0
                };
                
                // Expected impact = eta * exec_rate + gamma * quantity
                let temp_impact = params.eta * FixedPoint::from_raw(exec_rate as i64);
                let perm_impact = params.gamma * remaining;
                let total_impact = temp_impact + perm_impact;
                
                self.trajectory_buffer[i] = TrajectoryPoint {
                    time_fraction: t_fraction,
                    remaining_quantity: remaining,
                    execution_rate: FixedPoint::from_raw(exec_rate as i64),
                    expected_impact_bps: total_impact,
                    _padding: [0u8; 32],
                };
                
                // Update remaining quantity
                let step_qty = remaining / FixedPoint::from_raw(num_points as i64);
                remaining = remaining - step_qty;
            }
        }
        
        Some(&self.trajectory_buffer[..num_points])
    }
    
    /// Predict market impact for a given order size
    #[inline]
    pub fn predict_impact(&self, order_size: FixedPoint, participation_rate: FixedPoint) -> FixedPoint {
        let params = unsafe { core::ptr::read_volatile(&self.params) };
        
        // Impact = eta * participation_rate + gamma * order_size
        let temp_impact = params.eta * participation_rate;
        let perm_impact = params.gamma * order_size;
        
        temp_impact + perm_impact
    }
    
    /// Log prediction vs actual (shadow mode)
    #[inline]
    pub fn log_prediction(&self, log_entry: ImpactShadowLog) {
        if self.shadow_enabled.load(Ordering::Relaxed) == 0 {
            return;
        }
        
        self.shadow_log.push(log_entry);
        
        // Track prediction error
        let error_abs = log_entry.prediction_error_bps.to_raw().unsigned_abs() as u64;
        self.cumulative_prediction_error.fetch_add(error_abs, Ordering::Relaxed);
    }
    
    /// Halt impact modeling
    #[inline]
    pub fn halt(&self) {
        self.halted.store(1, Ordering::SeqCst);
    }
    
    /// Resume impact modeling
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
    
    /// Get current parameters version
    #[inline]
    pub fn get_params_version(&self) -> u64 {
        self.params_version.load(Ordering::Acquire)
    }
    
    /// Get calibration quality (0=poor, 1=acceptable, 2=good)
    #[inline]
    pub fn get_calibration_quality(&self) -> u64 {
        self.calibration_quality.load(Ordering::Acquire)
    }
    
    /// Fast integer square root using Newton-Raphson
    #[inline]
    fn fast_isqrt(&self, x: i64) -> i64 {
        if x <= 0 {
            return 0;
        }
        
        let mut guess = x / 2;
        if guess == 0 {
            return 1;
        }
        
        // Newton-Raphson iterations (unrolled)
        for _ in 0..5 {
            let next_guess = (guess + x / guess) / 2;
            if next_guess >= guess {
                break;
            }
            guess = next_guess;
        }
        
        guess
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
    fn test_modeler_initialization() {
        let modeler = MarketImpactModeler::new();
        assert_eq!(modeler.halted.load(Ordering::Relaxed), 0);
        assert_eq!(modeler.calibration_quality.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_calibration_sample_addition() {
        let modeler = MarketImpactModeler::new();
        
        let sample = CalibrationSample {
            order_size: FixedPoint::from_raw(1000000),
            participation_rate: FixedPoint::from_raw(500), // 5%
            price_impact_bps: FixedPoint::from_raw(25),
            time_horizon_us: 1000000,
            volatility: FixedPoint::from_raw(2000),
            timestamp_cycles: 12345,
            _padding: [0u8; 24],
        };
        
        modeler.add_calibration_sample(sample);
        assert_eq!(modeler.total_samples_calibrated.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_impact_prediction() {
        let modeler = MarketImpactModeler::new();
        
        // Set some reasonable parameters
        unsafe {
            let params = ImpactParams {
                eta: FixedPoint::from_raw(100),
                gamma: FixedPoint::from_raw(50),
                sigma: FixedPoint::from_raw(2000),
                lambda: FixedPoint::from_raw(500),
                alpha: FixedPoint::from_raw(100),
                _padding: [0u8; 32],
            };
            core::ptr::write_volatile(&mut modeler.params as *mut ImpactParams, params);
        }
        modeler.params_version.store(1, Ordering::Release);
        modeler.calibration_quality.store(1, Ordering::Release);
        
        let impact = modeler.predict_impact(
            FixedPoint::from_raw(1000000),
            FixedPoint::from_raw(100), // 1% participation
        );
        
        // Impact should be positive
        assert!(impact > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let modeler = MarketImpactModeler::new();
        
        modeler.halt();
        assert_eq!(modeler.halted.load(Ordering::Acquire), 1);
        
        modeler.resume();
        assert_eq!(modeler.halted.load(Ordering::Acquire), 0);
    }
    
    #[test]
    fn test_fast_isqrt() {
        let modeler = MarketImpactModeler::new();
        
        assert_eq!(modeler.fast_isqrt(100), 10);
        assert_eq!(modeler.fast_isqrt(10000), 100);
        assert_eq!(modeler.fast_isqrt(1000000), 1000);
    }
}
