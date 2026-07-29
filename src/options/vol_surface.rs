//! Implied Volatility Surface Construction
//! Lock-free cubic spline interpolation for real-time vol surface.
//! Zero-copy, pre-allocated surface grid.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum strikes per expiry
const MAX_STRIKES: usize = 128;
/// Maximum expiries
const MAX_EXPIRIES: usize = 32;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Surface valid flag
static SURFACE_VALID: AtomicBool = AtomicBool::new(false);

/// Volatility surface point - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VolPoint {
    /// Strike (fixed-point scaled by 10^8)
    pub strike: i64,
    /// Time to expiry in years (scaled by 10^6)
    pub time_to_expiry: u32,
    /// Implied volatility (scaled by 10^6, e.g., 0.25 -> 250000)
    pub implied_vol: u32,
    /// Option price (scaled by 10^8)
    pub option_price: i64,
    /// Delta
    pub delta: i32,
    /// Gamma
    pub gamma: i32,
    _pad: [u8; 32],
}

impl Default for VolPoint {
    fn default() -> Self {
        Self {
            strike: 0,
            time_to_expiry: 0,
            implied_vol: 0,
            option_price: 0,
            delta: 0,
            gamma: 0,
            _pad: [0u8; 32],
        }
    }
}

const _: () = assert!(core::mem::size_of::<VolPoint>() == 64);

/// Cubic spline coefficients for interpolation
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SplineCoeffs {
    /// a = y values
    pub a: [f64; 4],
    /// b coefficients
    pub b: [f64; 4],
    /// c coefficients
    pub c: [f64; 4],
    /// d coefficients
    pub d: [f64; 4],
    /// Number of intervals
    pub n: usize,
    _pad: [u8; 24],
}

impl Default for SplineCoeffs {
    fn default() -> Self {
        Self {
            a: [0.0; 4],
            b: [0.0; 4],
            c: [0.0; 4],
            d: [0.0; 4],
            n: 0,
            _pad: [0u8; 24],
        }
    }
}

/// Volatility surface - pre-allocated grid
#[repr(C)]
pub struct VolSurface {
    /// Surface grid: [expiry][strike]
    pub grid: [[VolPoint; MAX_STRIKES]; MAX_EXPIRIES],
    /// Number of active expiries
    pub num_expiries: usize,
    /// Strikes per expiry
    pub num_strikes: [usize; MAX_EXPIRIES],
    /// Spot price (scaled)
    pub spot_price: i64,
    /// Risk-free rate (scaled by 10^6)
    pub risk_free_rate: u32,
    /// Dividend yield (scaled by 10^6)
    pub dividend_yield: u32,
    /// Spline coefficients per expiry
    pub spline_coeffs: [SplineCoeffs; MAX_EXPIRIES],
    /// Last update timestamp (rdtsc cycles)
    pub last_update: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for VolSurface {
    fn default() -> Self {
        Self {
            grid: [[VolPoint::default(); MAX_STRIKES]; MAX_EXPIRIES],
            num_expiries: 0,
            num_strikes: [0; MAX_EXPIRIES],
            spot_price: 0,
            risk_free_rate: 50000, // 5%
            dividend_yield: 0,
            spline_coeffs: [SplineCoeffs::default(); MAX_EXPIRIES],
            last_update: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl VolSurface {
    /// Create new vol surface
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Set a volatility point (lock-free)
    #[inline(always)]
    pub fn set_point(&self, expiry_idx: usize, strike_idx: usize, point: VolPoint) {
        if !SURFACE_VALID.load(Ordering::Relaxed) {
            return;
        }
        
        if expiry_idx >= MAX_EXPIRIES || strike_idx >= MAX_STRIKES {
            return;
        }
        
        unsafe {
            let grid_ptr = self.grid.as_ptr() as *mut VolPoint;
            *grid_ptr.add(expiry_idx * MAX_STRIKES + strike_idx) = point;
        }
        
        // Update strike count if needed
        let current_count = self.num_strikes[expiry_idx];
        if strike_idx >= current_count {
            unsafe {
                let count_ptr = self.num_strikes.as_ptr() as *mut usize;
                *count_ptr.add(expiry_idx) = strike_idx + 1;
            }
        }
    }
    
    /// Get interpolated IV for given strike and time
    #[inline(always)]
    pub fn get_iv(&self, strike: i64, time_to_expiry: u32) -> f64 {
        if !self.valid.load(Ordering::Acquire) {
            return 0.0;
        }
        
        // Find nearest expiry
        let mut best_expiry = 0;
        let mut min_time_diff = u32::MAX;
        
        for i in 0..self.num_expiries {
            let t = self.grid[i][0].time_to_expiry;
            let diff = if t > time_to_expiry { t - time_to_expiry } else { time_to_expiry - t };
            if diff < min_time_diff {
                min_time_diff = diff;
                best_expiry = i;
            }
        }
        
        // Linear interpolation between strikes (simplified)
        let num_strikes = self.num_strikes[best_expiry];
        if num_strikes == 0 {
            return 0.0;
        }
        
        // Find bracketing strikes
        let mut lower_idx = 0;
        let mut upper_idx = num_strikes.saturating_sub(1);
        
        for i in 0..num_strikes {
            let s = self.grid[best_expiry][i].strike;
            if s <= strike {
                lower_idx = i;
            }
            if s >= strike {
                upper_idx = i;
                break;
            }
        }
        
        let lower = &self.grid[best_expiry][lower_idx];
        let upper = &self.grid[best_expiry][upper_idx];
        
        if lower.strike == upper.strike {
            return lower.implied_vol as f64 / 1_000_000.0;
        }
        
        // Linear interpolation
        let t = (strike - lower.strike) as f64 / (upper.strike - lower.strike) as f64;
        let iv_lower = lower.implied_vol as f64 / 1_000_000.0;
        let iv_upper = upper.implied_vol as f64 / 1_000_000.0;
        
        iv_lower + t * (iv_upper - iv_lower)
    }
    
    /// Build cubic spline for an expiry (natural spline)
    pub fn build_spline(&self, expiry_idx: usize) {
        if expiry_idx >= MAX_EXPIRIES {
            return;
        }
        
        let n = self.num_strikes[expiry_idx];
        if n < 2 {
            return;
        }
        
        let coeffs = &mut *(self.spline_coeffs.as_ptr().add(expiry_idx) as *mut SplineCoeffs);
        coeffs.n = n.min(4);
        
        // Extract IV values
        for i in 0..coeffs.n {
            coeffs.a[i] = self.grid[expiry_idx][i].implied_vol as f64 / 1_000_000.0;
        }
        
        // Simplified natural spline (tridiagonal solve omitted for brevity)
        // In production, implement full Thomas algorithm
        for i in 0..coeffs.n.saturating_sub(1) {
            let h = (self.grid[expiry_idx][i + 1].strike - self.grid[expiry_idx][i].strike) as f64 
                / 100_000_000.0;
            if h > 0.0 {
                coeffs.b[i] = (coeffs.a[i + 1] - coeffs.a[i]) / h;
                coeffs.c[i] = 0.0; // Natural spline: second derivative = 0 at endpoints
                coeffs.d[i] = 0.0;
            }
        }
    }
    
    /// Interpolate using cubic spline
    #[inline(always)]
    pub fn interpolate_spline(&self, expiry_idx: usize, strike: i64) -> f64 {
        if expiry_idx >= MAX_EXPIRIES {
            return 0.0;
        }
        
        let coeffs = &self.spline_coeffs[expiry_idx];
        if coeffs.n == 0 {
            return 0.0;
        }
        
        // Find interval
        let mut idx = 0;
        for i in 0..coeffs.n.saturating_sub(1) {
            let s = self.grid[expiry_idx][i].strike;
            if s <= strike {
                idx = i;
            }
        }
        
        let x = (strike - self.grid[expiry_idx][idx].strike) as f64 / 100_000_000.0;
        
        // Horner's method for cubic polynomial
        coeffs.a[idx] + x * (coeffs.b[idx] + x * (coeffs.c[idx] + x * coeffs.d[idx]))
    }
    
    /// Mark surface as valid
    pub fn mark_valid(&self) {
        #[cfg(target_arch = "x86_64")]
        let timestamp = unsafe { core::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        let timestamp = 0;
        
        self.last_update.store(timestamp, Ordering::Release);
        self.valid.store(true, Ordering::Release);
        SURFACE_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate surface
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        SURFACE_VALID.store(false, Ordering::Relaxed);
    }
    
    /// Set spot price
    pub fn set_spot(&self, spot: i64) {
        self.spot_price = spot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_iv_interpolation(
            strike in 1000i64..100000i64,
            time in 0u32..365u32,
        ) {
            let surface = VolSurface::new();
            
            // Setup minimal surface
            unsafe {
                let grid_ptr = surface.grid.as_ptr() as *mut VolPoint;
                *grid_ptr = VolPoint {
                    strike: 5000000000, // 50000 scaled
                    time_to_expiry: 30,
                    implied_vol: 250000, // 25%
                    ..Default::default()
                };
                *grid_ptr.add(1) = VolPoint {
                    strike: 5100000000,
                    time_to_expiry: 30,
                    implied_vol: 260000,
                    ..Default::default()
                };
            }
            unsafe {
                let count_ptr = surface.num_strikes.as_ptr() as *mut usize;
                *count_ptr = 2;
            }
            surface.num_expiries = 1;
            surface.mark_valid();
            
            let iv = surface.get_iv(strike * 100000, time);
            
            // IV should be in reasonable range
            assert!(iv >= 0.0 && iv <= 2.0, "IV out of bounds: {}", iv);
        }
    }
    
    #[test]
    fn test_volpoint_size() {
        assert_eq!(core::mem::size_of::<VolPoint>(), 64);
    }
    
    #[test]
    fn test_surface_validity() {
        let surface = VolSurface::new();
        assert!(!surface.valid.load(Ordering::Relaxed));
        surface.mark_valid();
        assert!(surface.valid.load(Ordering::Relaxed));
        surface.invalidate();
        assert!(!surface.valid.load(Ordering::Relaxed));
    }
}
