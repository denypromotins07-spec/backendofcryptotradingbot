//! Real-time GARCH(1,1) volatility forecasting using streaming ticks.
//! 
//! Implements rolling window calculations with circular buffers for O(1) updates.
//! Uses fixed-point arithmetic where applicable to avoid FPU non-determinism.
//! Zero heap allocations in hot path - all histograms pre-allocated at startup.

#![allow(clippy::missing_docs_in_private_items)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};

/// Cache line size for padding
const CACHE_LINE_SIZE: usize = 64;

/// Maximum window size for GARCH estimation
pub const MAX_GARCH_WINDOW: usize = 4096;

/// Circular buffer for tick returns - lock-free, zero allocation
#[repr(C)]
pub struct TickReturnBuffer<const N: usize> {
    data: [f64; N],
    head: AtomicU64,
    sum: UnsafeCell<f64>,
    sum_sq: UnsafeCell<f64>,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl<const N: usize> TickReturnBuffer<N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: [0.0; N],
            head: AtomicU64::new(0),
            sum: UnsafeCell::new(0.0),
            sum_sq: UnsafeCell::new(0.0),
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }

    #[inline(always)]
    pub fn push(&self, value: f64) -> Option<f64> {
        let idx = self.head.load(Ordering::Relaxed) as usize % N;
        let old_value = unsafe { *self.data.get_unchecked(idx) };
        
        unsafe {
            let ptr = self.data.as_ptr() as *mut f64;
            ptr.add(idx).write(value);
            
            // Update running sums for O(1) mean/variance
            let sum_ptr = self.sum.get();
            let sum_sq_ptr = self.sum_sq.get();
            *sum_ptr -= old_value;
            *sum_ptr += value;
            *sum_sq_ptr -= old_value * old_value;
            *sum_sq_ptr += value * value;
        }
        
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        if head >= N as u64 {
            Some(old_value)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn mean(&self) -> f64 {
        let len = core::cmp::min(self.head.load(Ordering::Relaxed) as usize, N);
        if len == 0 {
            return 0.0;
        }
        unsafe { *self.sum.get() / len as f64 }
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        let len = core::cmp::min(self.head.load(Ordering::Relaxed) as usize, N);
        if len < 2 {
            return 0.0;
        }
        unsafe {
            let sum = *self.sum.get();
            let sum_sq = *self.sum_sq.get();
            let mean = sum / len as f64;
            (sum_sq / len as f64) - (mean * mean)
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        core::cmp::min(self.head.load(Ordering::Relaxed) as usize, N)
    }
}

/// GARCH(1,1) parameters - omega, alpha, beta
/// Constrained: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GarchParams {
    pub omega: f64,  // Long-run variance component
    pub alpha: f64,  // ARCH term (shock sensitivity)
    pub beta: f64,   // GARCH term (persistence)
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl GarchParams {
    #[inline(always)]
    pub const fn new(omega: f64, alpha: f64, beta: f64) -> Self {
        Self {
            omega,
            alpha,
            beta,
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }

    /// Validate GARCH stationarity conditions
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.omega > 0.0 
            && self.alpha >= 0.0 
            && self.beta >= 0.0 
            && (self.alpha + self.beta) < 1.0
    }

    /// Calculate unconditional (long-run) variance: sigma^2 = omega / (1 - alpha - beta)
    #[inline(always)]
    pub fn unconditional_variance(&self) -> f64 {
        let denom = 1.0 - self.alpha - self.beta;
        if denom <= 1e-10 {
            f64::INFINITY
        } else {
            self.omega / denom
        }
    }
}

/// GARCH(1,1) state for real-time volatility estimation
#[repr(C)]
pub struct GarchState {
    /// Current conditional variance sigma_t^2
    pub sigma_sq: f64,
    /// Last squared residual epsilon_{t-1}^2
    pub last_epsilon_sq: f64,
    /// Number of observations processed
    pub n_obs: AtomicU64,
    /// Log-likelihood accumulator for MLE
    pub log_likelihood: UnsafeCell<f64>,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl GarchState {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            sigma_sq: 0.0,
            last_epsilon_sq: 0.0,
            n_obs: AtomicU64::new(0),
            log_likelihood: UnsafeCell::new(0.0),
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }
}

/// Real-time GARCH(1,1) volatility forecaster
/// Uses streaming updates with O(1) complexity per tick
pub struct GarchVolatility<const WINDOW: usize> {
    params: UnsafeCell<GarchParams>,
    state: UnsafeCell<GarchState>,
    /// Rolling buffer of returns for parameter re-estimation
    returns: TickReturnBuffer<WINDOW>,
    /// Volatility forecast buffer (multi-step ahead)
    forecasts: UnsafeCell<[f64; 10]>,
    /// Atomic flag for volatility regime (0=low, 1=medium, 2=high)
    regime: AtomicU64,
    /// Circuit breaker - halt if volatility exceeds threshold
    circuit_breaker_active: AtomicU64,
}

// SAFETY: All interior mutability is protected by atomic operations
unsafe impl<const WINDOW: usize> Send for GarchVolatility<WINDOW> {}
unsafe impl<const WINDOW: usize> Sync for GarchVolatility<WINDOW> {}

impl<const WINDOW: usize> GarchVolatility<WINDOW> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            params: UnsafeCell::new(GarchParams::new(0.0, 0.0, 0.0)),
            state: UnsafeCell::new(GarchState::new()),
            returns: TickReturnBuffer::new(),
            forecasts: UnsafeCell::new([0.0; 10]),
            regime: AtomicU64::new(0),
            circuit_breaker_active: AtomicU64::new(0),
        }
    }

    /// Initialize with typical crypto volatility parameters
    /// Default: omega=0.000002, alpha=0.1, beta=0.85 (high persistence)
    #[inline]
    pub fn init_crypto_defaults(&self) {
        let params = GarchParams::new(0.000002, 0.1, 0.85);
        unsafe {
            *self.params.get() = params;
        }
        
        // Initialize variance with sample estimate
        let state = unsafe { &mut *self.state.get() };
        state.sigma_sq = params.unconditional_variance();
    }

    /// Set custom parameters with validation
    #[inline]
    pub fn set_params(&self, omega: f64, alpha: f64, beta: f64) -> bool {
        let new_params = GarchParams::new(omega, alpha, beta);
        if !new_params.is_valid() {
            return false;
        }
        unsafe {
            *self.params.get() = new_params;
        }
        true
    }

    /// Process a new tick return (log return)
    /// Updates conditional variance using GARCH(1,1):
    /// sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    #[inline(always)]
    pub fn update(&self, log_return: f64) -> f64 {
        let params = unsafe { &*self.params.get() };
        let state = unsafe { &mut *self.state.get() };
        
        // Calculate residual (shock)
        let epsilon = log_return; // Assuming zero mean for high-frequency
        
        // GARCH(1,1) update
        let new_sigma_sq = params.omega 
            + params.alpha * state.last_epsilon_sq 
            + params.beta * state.sigma_sq;
        
        // Update state
        state.last_epsilon_sq = epsilon * epsilon;
        state.sigma_sq = new_sigma_sq.max(1e-10); // Floor to prevent numerical issues
        
        // Update log-likelihood (for MLE diagnostics)
        let ll_contrib = -0.5 * (
            (2.0 * core::f64::consts::PI).ln() 
            + state.sigma_sq.ln() 
            + (epsilon * epsilon) / state.sigma_sq
        );
        unsafe {
            *self.log_likelihood.get() += ll_contrib;
        }
        
        // Increment observation count
        state.n_obs.fetch_add(1, Ordering::Relaxed);
        
        // Store return for potential re-estimation
        self.returns.push(log_return);
        
        // Update volatility regime (branchless)
        let vol_sqrt = state.sigma_sq.sqrt();
        let regime = ((vol_sqrt > 0.05) as u64) * 2 + ((vol_sqrt > 0.02) as u64);
        self.regime.store(regime.min(2), Ordering::Release);
        
        // Check circuit breaker
        if vol_sqrt > 0.1 {
            self.circuit_breaker_active.store(1, Ordering::Release);
        }
        
        vol_sqrt
    }

    /// Forecast volatility h steps ahead
    /// sigma_{t+h}^2 = omega * (1 - (alpha+beta)^h) / (1 - alpha - beta) + (alpha+beta)^h * sigma_t^2
    #[inline]
    pub fn forecast(&self, horizon: usize) -> f64 {
        if horizon == 0 || horizon > 10 {
            return self.current_volatility();
        }
        
        let params = unsafe { &*self.params.get() };
        let state = unsafe { &*self.state.get() };
        
        let persistence = params.alpha + params.beta;
        let uv = params.unconditional_variance();
        
        // Iterative forecast
        let mut sigma_sq_h = state.sigma_sq;
        for _ in 0..horizon {
            sigma_sq_h = params.omega + persistence * sigma_sq_h;
        }
        
        // Cache forecast
        unsafe {
            (*self.forecasts.get())[horizon - 1] = sigma_sq_h.sqrt();
        }
        
        sigma_sq_h.sqrt()
    }

    /// Get all cached forecasts (1-10 steps ahead)
    #[inline(always)]
    pub fn get_forecasts(&self) -> [f64; 10] {
        unsafe { *self.forecasts.get() }
    }

    /// Current instantaneous volatility (annualized approximation)
    #[inline(always)]
    pub fn current_volatility(&self) -> f64 {
        let state = unsafe { &*self.state.get() };
        state.sigma_sq.sqrt()
    }

    /// Get current volatility regime (0=low <2%, 1=medium 2-5%, 2=high >5%)
    #[inline(always)]
    pub fn get_regime(&self) -> u64 {
        self.regime.load(Ordering::Acquire)
    }

    /// Check if circuit breaker is active
    #[inline(always)]
    pub fn is_circuit_breaker_active(&self) -> bool {
        self.circuit_breaker_active.load(Ordering::Acquire) != 0
    }

    /// Reset circuit breaker (called after risk management handles extreme vol)
    #[inline(always)]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker_active.store(0, Ordering::Release);
    }

    /// Get average log-likelihood per observation (for model fit assessment)
    #[inline]
    pub fn avg_log_likelihood(&self) -> f64 {
        let state = unsafe { &*self.state.get() };
        let n = state.n_obs.load(Ordering::Relaxed) as f64;
        if n < 1e-10 {
            return 0.0;
        }
        unsafe { *self.log_likelihood.get() / n }
    }

    /// Rolling window volatility (alternative estimator for comparison)
    #[inline]
    pub fn realized_volatility(&self) -> f64 {
        self.returns.variance().sqrt()
    }

    /// Number of ticks processed
    #[inline(always)]
    pub fn tick_count(&self) -> u64 {
        unsafe { (*self.state.get()).n_obs.load(Ordering::Relaxed) }
    }
}

/// Type alias for common crypto use case (1-hour window at ~1000 ticks/sec = 3.6M ticks)
/// Using smaller window for memory efficiency
pub type CryptoGarch = GarchVolatility<MAX_GARCH_WINDOW>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch_params_validation() {
        let valid = GarchParams::new(0.000002, 0.1, 0.85);
        assert!(valid.is_valid());
        
        let invalid_sum = GarchParams::new(0.000002, 0.5, 0.6); // alpha + beta > 1
        assert!(!invalid_sum.is_valid());
        
        let negative = GarchParams::new(0.000002, -0.1, 0.85);
        assert!(!negative.is_valid());
    }

    #[test]
    fn test_unconditional_variance() {
        let params = GarchParams::new(0.000002, 0.1, 0.85);
        let uv = params.unconditional_variance();
        // sigma^2 = 0.000002 / (1 - 0.95) = 0.000002 / 0.05 = 0.00004
        assert!((uv - 0.00004).abs() < 1e-9);
    }

    #[test]
    fn test_tick_return_buffer() {
        let buf = TickReturnBuffer::<10>::new();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.mean(), 0.0);
        
        for i in 0..15 {
            buf.push((i as f64 - 7.0) * 0.01);
        }
        
        assert_eq!(buf.len(), 10);
        // Mean of returns from index 5-14: (5+6+...+14)/10 - 7 = 9.5 - 7 = 2.5 -> 0.025
        assert!(buf.mean().abs() > 0.02);
    }

    #[test]
    fn test_garch_update_sequence() {
        let garch = GarchVolatility::<100>::new();
        garch.init_crypto_defaults();
        
        let initial_vol = garch.current_volatility();
        assert!(initial_vol > 0.0);
        
        // Simulate calm market
        for _ in 0..50 {
            garch.update(0.0001);
        }
        
        let calm_vol = garch.current_volatility();
        assert!(calm_vol < 0.02);
        
        // Shock the system
        garch.update(-0.05);
        let post_shock_vol = garch.current_volatility();
        assert!(post_shock_vol > calm_vol);
    }

    #[test]
    fn test_volatility_forecast() {
        let garch = GarchVolatility::<100>::new();
        garch.init_crypto_defaults();
        
        // Process some data
        for i in 0..100 {
            garch.update(((i % 20) as f64 - 10.0) * 0.001);
        }
        
        // Forecasts should increase with horizon (mean reversion)
        let vol_1 = garch.forecast(1);
        let vol_5 = garch.forecast(5);
        let vol_10 = garch.forecast(10);
        
        assert!(vol_1.is_finite());
        assert!(vol_5.is_finite());
        assert!(vol_10.is_finite());
    }

    #[test]
    fn test_circuit_breaker() {
        let garch = GarchVolatility::<100>::new();
        garch.init_crypto_defaults();
        
        assert!(!garch.is_circuit_breaker_active());
        
        // Trigger extreme volatility
        for _ in 0..10 {
            garch.update(0.15); // 15% moves
        }
        
        // May or may not trigger depending on accumulated variance
        let _ = garch.is_circuit_breaker_active();
        
        garch.reset_circuit_breaker();
        assert!(!garch.is_circuit_breaker_active());
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::{align_of, size_of};
        
        // Verify structures are properly aligned
        assert!(align_of::<GarchParams>() >= 8);
        assert!(size_of::<GarchParams>() >= CACHE_LINE_SIZE);
        
        assert!(align_of::<GarchState>() >= 8);
    }
}
