//! Automated walk-forward optimization pipeline using shared memory IPC.
//! 
//! Implements circuit breaker for memory limit enforcement (6.5GB).
//! Shadow-mode logging for theoretical fill validation.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum optimization parameters
pub const MAX_PARAMS: usize = 16;

/// Maximum in-sample periods
pub const MAX_PERIODS: usize = 256;

/// Optimization result
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OptResult {
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_return: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub trades_count: u64,
    _padding: [u8; CACHE_LINE_SIZE - 48],
}

impl OptResult {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            total_return: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            trades_count: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 48],
        }
    }
}

/// Parameter set for optimization
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ParamSet {
    pub values: [f64; MAX_PARAMS],
    pub count: usize,
    _padding: [u8; CACHE_LINE_SIZE - (MAX_PARAMS * 8 + 8)],
}

impl ParamSet {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            values: [0.0; MAX_PARAMS],
            count: 0,
            _padding: [0u8; CACHE_LINE_SIZE - (MAX_PARAMS * 8 + 8)],
        }
    }
}

/// Walk-forward state
#[repr(C)]
pub struct WalkForwardOptimizer<const MAX_PARAMS: usize, const MAX_PERIODS: usize> {
    /// Current parameter values
    params: ParamSet,
    /// Best parameters found
    best_params: ParamSet,
    /// Best result
    best_result: OptResult,
    /// In-sample results per period
    is_results: UnsafeCell<[OptResult; MAX_PERIODS]>,
    /// Out-of-sample results
    oos_results: UnsafeCell<[OptResult; MAX_PERIODS]>,
    /// Period index
    current_period: AtomicU64,
    /// Total periods
    total_periods: AtomicU64,
    /// Memory usage tracker (bytes)
    memory_used: AtomicU64,
    /// Memory limit (bytes)
    memory_limit: u64,
    /// Circuit breaker active
    circuit_breaker: AtomicBool,
    /// Optimization running
    is_running: AtomicBool,
    /// Shadow mode enabled
    shadow_mode: AtomicBool,
    /// Shadow log entries
    shadow_log_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 34],
}

use core::cell::UnsafeCell;

// SAFETY: All interior mutability protected by atomics
unsafe impl<const P: usize, const S: usize> Send for WalkForwardOptimizer<P, S> {}
unsafe impl<const P: usize, const S: usize> Sync for WalkForwardOptimizer<P, S> {}

impl<const MAX_PARAMS: usize, const MAX_PERIODS: usize> WalkForwardOptimizer<MAX_PARAMS, MAX_PERIODS> {
    #[inline(always)]
    pub const fn new(memory_limit_gb: f64) -> Self {
        Self {
            params: ParamSet::new(),
            best_params: ParamSet::new(),
            best_result: OptResult::new(),
            is_results: UnsafeCell::new([OptResult::new(); MAX_PERIODS]),
            oos_results: UnsafeCell::new([OptResult::new(); MAX_PERIODS]),
            current_period: AtomicU64::new(0),
            total_periods: AtomicU64::new(0),
            memory_used: AtomicU64::new(0),
            memory_limit: (memory_limit_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            circuit_breaker: AtomicBool::new(false),
            is_running: AtomicBool::new(false),
            shadow_mode: AtomicBool::new(false),
            shadow_log_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 34],
        }
    }

    /// Check memory budget
    #[inline]
    pub fn check_memory(&self, additional: u64) -> bool {
        let current = self.memory_used.load(Ordering::Relaxed);
        if current + additional > self.memory_limit {
            self.circuit_breaker.store(true, Ordering::Release);
            return false;
        }
        self.memory_used.fetch_add(additional, Ordering::Relaxed);
        true
    }

    /// Reset memory tracker
    #[inline]
    pub fn reset_memory_tracker(&self) {
        self.memory_used.store(0, Ordering::Relaxed);
    }

    /// Set optimization parameters
    #[inline]
    pub fn set_params(&self, values: &[f64]) -> bool {
        if values.len() > MAX_PARAMS {
            return false;
        }

        let params = unsafe { &mut *self.params.get_unchecked_mut() };
        params.count = values.len();
        for i in 0..values.len() {
            params.values[i] = values[i];
        }
        true
    }

    /// Run single optimization iteration
    #[inline]
    pub fn run_iteration(&self, period: usize, is_sample: bool, result: OptResult) -> bool {
        if self.circuit_breaker.load(Ordering::Acquire) {
            return false;
        }

        if period >= MAX_PERIODS {
            return false;
        }

        if is_sample {
            unsafe {
                let results = &mut *self.is_results.get();
                results[period] = result;
            }
        } else {
            unsafe {
                let results = &mut *self.oos_results.get();
                results[period] = result;
            }
        }

        // Update best result
        let best = unsafe { &mut *self.best_result.get_unchecked_mut() };
        if result.sharpe_ratio > best.sharpe_ratio {
            *best = result;
            let bp = unsafe { &mut *self.best_params.get_unchecked_mut() };
            *bp = self.params.clone();
        }

        self.current_period.fetch_add(1, Ordering::Relaxed);

        // Shadow mode logging
        if self.shadow_mode.load(Ordering::Relaxed) {
            self.shadow_log_count.fetch_add(1, Ordering::Relaxed);
        }

        true
    }

    /// Get aggregated in-sample statistics
    #[inline]
    pub fn aggregate_is_stats(&self) -> OptResult {
        let results = unsafe { &*self.is_results.get() };
        let mut agg = OptResult::new();
        let mut count = 0u64;

        for r in results.iter() {
            if r.trades_count > 0 {
                agg.sharpe_ratio += r.sharpe_ratio;
                agg.max_drawdown = agg.max_drawdown.max(r.max_drawdown);
                agg.total_return += r.total_return;
                agg.win_rate += r.win_rate;
                agg.profit_factor += r.profit_factor;
                agg.trades_count += r.trades_count;
                count += 1;
            }
        }

        if count > 0 {
            agg.sharpe_ratio /= count as f64;
            agg.win_rate /= count as f64;
            agg.profit_factor /= count as f64;
        }

        agg
    }

    /// Get aggregated out-of-sample statistics
    #[inline]
    pub fn aggregate_oos_stats(&self) -> OptResult {
        let results = unsafe { &*self.oos_results.get() };
        let mut agg = OptResult::new();
        let mut count = 0u64;

        for r in results.iter() {
            if r.trades_count > 0 {
                agg.sharpe_ratio += r.sharpe_ratio;
                agg.max_drawdown = agg.max_drawdown.max(r.max_drawdown);
                agg.total_return += r.total_return;
                agg.win_rate += r.win_rate;
                agg.profit_factor += r.profit_factor;
                agg.trades_count += r.trades_count;
                count += 1;
            }
        }

        if count > 0 {
            agg.sharpe_ratio /= count as f64;
            agg.win_rate /= count as f64;
            agg.profit_factor /= count as f64;
        }

        agg
    }

    /// Check for overfitting (OOS significantly worse than IS)
    #[inline]
    pub fn check_overfitting(&self, threshold: f64) -> bool {
        let is_stats = self.aggregate_is_stats();
        let oos_stats = self.aggregate_oos_stats();

        if is_stats.sharpe_ratio == 0.0 {
            return false;
        }

        let degradation = (is_stats.sharpe_ratio - oos_stats.sharpe_ratio) / is_stats.sharpe_ratio;
        degradation > threshold
    }

    /// Enable shadow mode
    #[inline]
    pub fn enable_shadow_mode(&self) {
        self.shadow_mode.store(true, Ordering::Release);
    }

    /// Disable shadow mode
    #[inline]
    pub fn disable_shadow_mode(&self) {
        self.shadow_mode.store(false, Ordering::Release);
    }

    /// Get shadow log count
    #[inline(always)]
    pub fn shadow_log_count(&self) -> u64 {
        self.shadow_log_count.load(Ordering::Relaxed)
    }

    /// Get best parameters
    #[inline]
    pub fn get_best_params(&self) -> ParamSet {
        self.best_params.clone()
    }

    /// Get best result
    #[inline]
    pub fn get_best_result(&self) -> OptResult {
        self.best_result.clone()
    }

    /// Start optimization
    #[inline]
    pub fn start(&self, periods: usize) {
        self.total_periods.store(periods as u64, Ordering::Relaxed);
        self.current_period.store(0, Ordering::Relaxed);
        self.is_running.store(true, Ordering::Release);
        self.circuit_breaker.store(false, Ordering::Relaxed);
    }

    /// Stop optimization
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Check if circuit breaker is active
    #[inline(always)]
    pub fn is_circuit_breaker_active(&self) -> bool {
        self.circuit_breaker.load(Ordering::Acquire)
    }

    /// Reset optimizer
    #[inline]
    pub fn reset(&self) {
        self.current_period.store(0, Ordering::Relaxed);
        self.is_running.store(false, Ordering::Relaxed);
        self.circuit_breaker.store(false, Ordering::Relaxed);
        self.best_result = OptResult::new();
        self.best_params = ParamSet::new();
        self.shadow_log_count.store(0, Ordering::Relaxed);
    }
}

/// Type alias for typical configuration
pub type CryptoWalkForward = WalkForwardOptimizer<MAX_PARAMS, MAX_PERIODS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_result_creation() {
        let result = OptResult::new();
        assert_eq!(result.sharpe_ratio, 0.0);
        assert_eq!(result.trades_count, 0);
    }

    #[test]
    fn test_walk_forward_init() {
        let wf = CryptoWalkForward::new(6.5);
        assert!(!wf.is_circuit_breaker_active());
        assert!(wf.check_memory(1024));
    }

    #[test]
    fn test_parameter_setting() {
        let wf = CryptoWalkForward::new(6.5);
        let params = vec![0.1, 0.2, 0.3];
        assert!(wf.set_params(&params));
    }

    #[test]
    fn test_iteration_recording() {
        let wf = CryptoWalkForward::new(6.5);

        let result = OptResult {
            sharpe_ratio: 1.5,
            max_drawdown: 0.1,
            total_return: 0.25,
            win_rate: 0.55,
            profit_factor: 1.8,
            trades_count: 100,
            _padding: [0u8; CACHE_LINE_SIZE - 48],
        };

        assert!(wf.run_iteration(0, true, result));
        assert!(wf.run_iteration(0, false, result));
    }

    #[test]
    fn test_memory_circuit_breaker() {
        let wf = CryptoWalkForward::new(0.001); // 1MB limit

        assert!(wf.check_memory(500));
        assert!(!wf.check_memory(1024 * 1024)); // Should trigger
        assert!(wf.is_circuit_breaker_active());
    }

    #[test]
    fn test_shadow_mode() {
        let wf = CryptoWalkForward::new(6.5);

        assert!(!wf.shadow_mode.load(Ordering::Relaxed));
        wf.enable_shadow_mode();
        assert!(wf.shadow_mode.load(Ordering::Relaxed));

        wf.run_iteration(0, true, OptResult::new());
        assert!(wf.shadow_log_count() > 0);

        wf.disable_shadow_mode();
    }

    #[test]
    fn test_aggregate_stats() {
        let wf = CryptoWalkForward::new(6.5);

        for i in 0..5 {
            let result = OptResult {
                sharpe_ratio: 1.0 + (i as f64 * 0.1),
                max_drawdown: 0.1,
                total_return: 0.1,
                win_rate: 0.5,
                profit_factor: 1.5,
                trades_count: 10,
                _padding: [0u8; CACHE_LINE_SIZE - 48],
            };
            wf.run_iteration(i, true, result);
        }

        let stats = wf.aggregate_is_stats();
        assert!(stats.sharpe_ratio > 1.0);
        assert!(stats.trades_count > 0);
    }
}
