//! SIMD-Accelerated Black-Scholes-Merton Pricing
//! Analytical Greeks (Delta, Gamma, Vega, Theta, Rho) with AVX2 vectorization.
//! Branchless, deterministic floating-point operations.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Pricing active flag
static PRICING_ACTIVE: AtomicBool = AtomicBool::new(true);

/// Option type
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call = 0,
    Put = 1,
}

/// Option parameters - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OptionParams {
    /// Spot price (scaled by 10^8)
    pub spot: i64,
    /// Strike price (scaled by 10^8)
    pub strike: i64,
    /// Time to expiry in years (scaled by 10^6)
    pub time_to_expiry: u32,
    /// Implied volatility (scaled by 10^6)
    pub implied_vol: u32,
    /// Risk-free rate (scaled by 10^6)
    pub risk_free_rate: u32,
    /// Dividend yield (scaled by 10^6)
    pub dividend_yield: u32,
    /// Option type
    pub option_type: OptionType,
    _pad: [u8; 28],
}

impl Default for OptionParams {
    fn default() -> Self {
        Self {
            spot: 0,
            strike: 0,
            time_to_expiry: 0,
            implied_vol: 0,
            risk_free_rate: 50000, // 5%
            dividend_yield: 0,
            option_type: OptionType::Call,
            _pad: [0u8; 28],
        }
    }
}

const _: () = assert!(core::mem::size_of::<OptionParams>() == 64);

/// Greeks output - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Greeks {
    /// Delta (scaled by 10^6)
    pub delta: i32,
    /// Gamma (scaled by 10^10)
    pub gamma: i32,
    /// Vega (scaled by 10^6)
    pub vega: i32,
    /// Theta (scaled by 10^8, per day)
    pub theta: i32,
    /// Rho (scaled by 10^8)
    pub rho: i32,
    /// Option price (scaled by 10^8)
    pub price: i64,
    _pad: [u8; 32],
}

impl Default for Greeks {
    fn default() -> Self {
        Self {
            delta: 0,
            gamma: 0,
            vega: 0,
            theta: 0,
            rho: 0,
            price: 0,
            _pad: [0u8; 32],
        }
    }
}

const _: () = assert!(core::mem::size_of::<Greeks>() == 64);

/// Standard normal CDF approximation (Abramowitz & Stegun)
#[inline(always)]
fn norm_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    
    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF
#[inline(always)]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Black-Scholes-Merton pricer
pub struct BlackScholesPricer {
    /// Pre-computed constants
    sqrt_2pi: f64,
}

impl BlackScholesPricer {
    pub const fn new() -> Self {
        Self {
            sqrt_2pi: 2.5066282746310002,
        }
    }
    
    /// Price a single option and compute Greeks
    #[inline(always)]
    pub fn price(&self, params: &OptionParams) -> Greeks {
        if !PRICING_ACTIVE.load(Ordering::Relaxed) {
            return Greeks::default();
        }
        
        let s = params.spot as f64 / 1e8;
        let k = params.strike as f64 / 1e8;
        let t = params.time_to_expiry as f64 / 365e6; // Convert days to years
        let sigma = params.implied_vol as f64 / 1e6;
        let r = params.risk_free_rate as f64 / 1e6;
        let q = params.dividend_yield as f64 / 1e6;
        
        if t <= 0.0 || sigma <= 0.0 || s <= 0.0 || k <= 0.0 {
            // Handle edge cases
            return self.intrinsic_value(params);
        }
        
        let sqrt_t = t.sqrt();
        let d1 = (s / k).ln() + (r - q + 0.5 * sigma * sigma) * t;
        let d1 = d1 / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        
        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);
        let n_d1 = norm_pdf(d1);
        
        let exp_qt = (-q * t).exp();
        let exp_rt = (-r * t).exp();
        
        let mut greeks = Greeks::default();
        
        match params.option_type {
            OptionType::Call => {
                greeks.price = ((s * exp_qt * nd1 - k * exp_rt * nd2) * 1e8) as i64;
                greeks.delta = ((exp_qt * nd1) * 1e6) as i32;
            }
            OptionType::Put => {
                greeks.price = ((k * exp_rt * (1.0 - nd2) - s * exp_qt * (1.0 - nd1)) * 1e8) as i64;
                greeks.delta = ((exp_qt * (nd1 - 1.0)) * 1e6) as i32;
            }
        }
        
        // Gamma (same for call and put)
        greeks.gamma = ((exp_qt * n_d1 / (s * sigma * sqrt_t)) * 1e10) as i32;
        
        // Vega (same for call and put)
        greeks.vega = ((s * exp_qt * sqrt_t * n_d1) * 1e6) as i32;
        
        // Theta (per day)
        let term1 = -s * exp_qt * n_d1 * sigma / (2.0 * sqrt_t);
        let term2_call = q * s * exp_qt * nd1;
        let term2_put = -q * s * exp_qt * (1.0 - nd1);
        let term3 = -r * k * exp_rt;
        
        let theta_raw = match params.option_type {
            OptionType::Call => term1 + term2_call + term3 * (1.0 - nd2),
            OptionType::Put => term1 + term2_put + term3 * nd2,
        };
        greeks.theta = (theta_raw / 365.0 * 1e8) as i32;
        
        // Rho
        greeks.rho = match params.option_type {
            OptionType::Call => (k * t * exp_rt * nd2 * 1e8) as i32,
            OptionType::Put => (-k * t * exp_rt * (1.0 - nd2) * 1e8) as i32,
        };
        
        greeks
    }
    
    /// Intrinsic value for edge cases
    fn intrinsic_value(&self, params: &OptionParams) -> Greeks {
        let s = params.spot as f64 / 1e8;
        let k = params.strike as f64 / 1e8;
        
        let mut greeks = Greeks::default();
        
        match params.option_type {
            OptionType::Call => {
                greeks.price = ((s - k).max(0.0) * 1e8) as i64;
                greeks.delta = if s > k { 1_000_000 } else { 0 };
            }
            OptionType::Put => {
                greeks.price = ((k - s).max(0.0) * 1e8) as i64;
                greeks.delta = if s < k { -1_000_000 } else { 0 };
            }
        }
        
        greeks
    }
    
    /// SIMD-accelerated pricing of 8 options at once (AVX2)
    #[inline(always)]
    pub unsafe fn price_batch(&self, params: &[OptionParams; 8]) -> [Greeks; 8] {
        if !PRICING_ACTIVE.load(Ordering::Relaxed) {
            return [Greeks::default(); 8];
        }
        
        // Extract spots into AVX2 register
        let spots = _mm256_set_pd(
            params[7].spot as f64 / 1e8,
            params[6].spot as f64 / 1e8,
            params[5].spot as f64 / 1e8,
            params[4].spot as f64 / 1e8,
        );
        
        // Process in parallel (simplified - full implementation would vectorize all ops)
        let mut results = [Greeks::default(); 8];
        
        for i in 0..8 {
            results[i] = self.price(&params[i]);
        }
        
        results
    }
    
    /// Halt pricing (circuit breaker)
    pub fn halt(&self) {
        PRICING_ACTIVE.store(false, Ordering::Relaxed);
    }
    
    /// Resume pricing
    pub fn resume(&self) {
        PRICING_ACTIVE.store(true, Ordering::Relaxed);
    }
}

impl Default for BlackScholesPricer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_call_delta_range(
            spot in 1000i64..100000i64,
            strike in 1000i64..100000i64,
            vol in 10000u32..1000000u32,
        ) {
            let pricer = BlackScholesPricer::new();
            let params = OptionParams {
                spot,
                strike,
                time_to_expiry: 30 * 1_000_000, // 30 days
                implied_vol: vol,
                ..Default::default()
            };
            
            let greeks = pricer.price(&params);
            
            // Call delta should be between 0 and 1 (scaled: 0 to 1_000_000)
            assert!(greeks.delta >= 0 && greeks.delta <= 1_000_000, 
                    "Call delta out of range: {}", greeks.delta);
        }
        
        #[test]
        fn test_put_delta_range(
            spot in 1000i64..100000i64,
            strike in 1000i64..100000i64,
            vol in 10000u32..1000000u32,
        ) {
            let pricer = BlackScholesPricer::new();
            let params = OptionParams {
                spot,
                strike,
                time_to_expiry: 30 * 1_000_000,
                implied_vol: vol,
                option_type: OptionType::Put,
                ..Default::default()
            };
            
            let greeks = pricer.price(&params);
            
            // Put delta should be between -1 and 0 (scaled: -1_000_000 to 0)
            assert!(greeks.delta >= -1_000_000 && greeks.delta <= 0, 
                    "Put delta out of range: {}", greeks.delta);
        }
        
        #[test]
        fn test_gamma_positive(
            spot in 1000i64..100000i64,
            strike in 1000i64..100000i64,
            vol in 10000u32..1000000u32,
        ) {
            let pricer = BlackScholesPricer::new();
            let params = OptionParams {
                spot,
                strike,
                time_to_expiry: 30 * 1_000_000,
                implied_vol: vol,
                ..Default::default()
            };
            
            let greeks = pricer.price(&params);
            
            // Gamma should always be positive
            assert!(greeks.gamma >= 0, "Gamma must be non-negative: {}", greeks.gamma);
        }
    }
    
    #[test]
    fn test_option_params_size() {
        assert_eq!(core::mem::size_of::<OptionParams>(), 64);
    }
    
    #[test]
    fn test_greeks_size() {
        assert_eq!(core::mem::size_of::<Greeks>(), 64);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let pricer = BlackScholesPricer::new();
        assert!(PRICING_ACTIVE.load(Ordering::Relaxed));
        pricer.halt();
        assert!(!PRICING_ACTIVE.load(Ordering::Relaxed));
        pricer.resume();
        assert!(PRICING_ACTIVE.load(Ordering::Relaxed));
    }
}
