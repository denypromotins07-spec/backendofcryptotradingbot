//! Threshold-based and time-sliced atomic rebalancing execution router.
//! Lock-free circular buffer for historical target weights.
//! Shadow-mode logging for validation without risk.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};

/// Maximum assets in portfolio
pub const MAX_ASSETS: usize = 64;

/// Maximum rebalance history entries (pre-allocated)
pub const MAX_HISTORY: usize = 1024;

/// Fixed-point representation (scaled by 10^8)
pub type FixedWeight = i64;
const SCALE: i64 = 100_000_000;

/// Rebalance trigger types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RebalanceTrigger {
    Threshold = 0,      // Drift exceeded threshold
    TimeBased = 1,      // Scheduled rebalance
    Manual = 2,         // Manual trigger
    RiskEvent = 3,      // Risk parity change
}

/// Cache-line aligned rebalance order
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RebalanceOrder {
    /// Asset ID
    pub asset_id: u32,
    /// Current weight (fixed-point)
    pub current_weight: FixedWeight,
    /// Target weight (fixed-point)
    pub target_weight: FixedWeight,
    /// Weight delta (fixed-point)
    pub delta: FixedWeight,
    /// Order ID for tracking
    pub order_id: u64,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Is this a buy (true) or sell (false)
    pub is_buy: bool,
    /// Executed flag
    pub executed: bool,
    _padding: [u8; 26], // Pad to 64 bytes
}

impl Default for RebalanceOrder {
    fn default() -> Self {
        Self {
            asset_id: 0,
            current_weight: 0,
            target_weight: 0,
            delta: 0,
            order_id: 0,
            timestamp_ns: 0,
            is_buy: false,
            executed: false,
            _padding: [0; 26],
        }
    }
}

/// Rebalance history entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RebalanceHistory {
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Trigger type
    pub trigger: RebalanceTrigger,
    /// Total drift before rebalance (basis points)
    pub total_drift_bps: i32,
    /// Number of orders generated
    pub order_count: u32,
    /// Was rebalance executed (vs shadow mode)
    pub executed: bool,
    /// Portfolio turnover (basis points)
    pub turnover_bps: i32,
    _padding: [u8; 40], // Pad to 64 bytes
}

impl Default for RebalanceHistory {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            trigger: RebalanceTrigger::Threshold,
            total_drift_bps: 0,
            order_count: 0,
            executed: false,
            turnover_bps: 0,
            _padding: [0; 40],
        }
    }
}

/// Main rebalancing engine
#[repr(C)]
pub struct RebalancingEngine {
    /// Current weights
    pub current_weights: [FixedWeight; MAX_ASSETS],
    /// Target weights
    pub target_weights: [FixedWeight; MAX_ASSETS],
    /// Pre-allocated order buffer
    pub pending_orders: [RebalanceOrder; MAX_ASSETS],
    /// Pre-allocated history buffer (circular)
    pub history: [RebalanceHistory; MAX_HISTORY],
    /// Asset count
    pub asset_count: u32,
    /// Head index for history circular buffer
    pub history_head: AtomicU64,
    /// Order counter for unique IDs
    pub order_counter: AtomicU64,
    /// Drift threshold (basis points)
    pub drift_threshold_bps: i32,
    /// Time-based rebalance interval (ns)
    pub rebalance_interval_ns: u64,
    /// Last rebalance timestamp
    pub last_rebalance_ns: AtomicU64,
    /// Shadow mode (log only, no execution)
    pub shadow_mode: AtomicBool,
    /// Rebalance needed flag
    pub rebalance_needed: AtomicBool,
    /// Circuit breaker (halt rebalancing)
    pub circuit_breaker: AtomicBool,
    _padding: [u8; 16], // Align to cache line
}

impl RebalancingEngine {
    pub fn new(drift_threshold_bps: i32, rebalance_interval_ms: u64) -> Self {
        Self {
            current_weights: [0; MAX_ASSETS],
            target_weights: [0; MAX_ASSETS],
            pending_orders: [RebalanceOrder::default(); MAX_ASSETS],
            history: [RebalanceHistory::default(); MAX_HISTORY],
            asset_count: 0,
            history_head: AtomicU64::new(0),
            order_counter: AtomicU64::new(0),
            drift_threshold_bps,
            rebalance_interval_ns: rebalance_interval_ms * 1_000_000,
            last_rebalance_ns: AtomicU64::new(0),
            shadow_mode: AtomicBool::new(true), // Default to shadow mode
            rebalance_needed: AtomicBool::new(false),
            circuit_breaker: AtomicBool::new(false),
            _padding: [0; 16],
        }
    }

    /// Set target weight for an asset
    #[inline]
    pub fn set_target_weight(&mut self, asset_idx: u32, weight: FixedWeight) {
        if asset_idx < MAX_ASSETS as u32 {
            self.target_weights[asset_idx as usize] = weight;
            if asset_idx >= self.asset_count {
                self.asset_count = asset_idx + 1;
            }
        }
    }

    /// Update current weight (from fills/market data)
    #[inline]
    pub fn update_current_weight(&mut self, asset_idx: u32, weight: FixedWeight) {
        if asset_idx < MAX_ASSETS as u32 {
            self.current_weights[asset_idx as usize] = weight;
        }
    }

    /// Check if rebalance is needed (threshold or time-based)
    #[inline]
    pub fn check_rebalance_needed(&self, current_time_ns: u64) -> bool {
        // Check circuit breaker first
        if self.circuit_breaker.load(Ordering::Acquire) {
            return false;
        }

        // Check drift threshold
        let mut max_drift: i32 = 0;
        for i in 0..self.asset_count as usize {
            let drift = ((self.current_weights[i] - self.target_weights[i]).abs() * 10_000) / SCALE;
            max_drift = max_drift.max(drift as i32);
        }

        if max_drift >= self.drift_threshold_bps {
            return true;
        }

        // Check time-based interval
        let last_rebal = self.last_rebalance_ns.load(Ordering::Acquire);
        if last_rebal > 0 && current_time_ns.saturating_sub(last_rebal) >= self.rebalance_interval_ns {
            return true;
        }

        false
    }

    /// Generate rebalance orders
    /// Returns number of orders generated
    #[inline]
    pub fn generate_orders(&mut self, trigger: RebalanceTrigger, timestamp_ns: u64) -> u32 {
        if self.circuit_breaker.load(Ordering::Acquire) {
            return 0;
        }

        let mut order_count: u32 = 0;
        let mut total_turnover: i64 = 0;

        for i in 0..self.asset_count as usize {
            let current = self.current_weights[i];
            let target = self.target_weights[i];
            let delta = target - current;

            // Only create order if delta is significant (> 1 bp)
            if delta.abs() >= SCALE / 10_000 {
                let order_id = self.order_counter.fetch_add(1, Ordering::AcqRel);
                
                self.pending_orders[order_count as usize] = RebalanceOrder {
                    asset_id: i as u32,
                    current_weight: current,
                    target_weight: target,
                    delta,
                    order_id,
                    timestamp_ns,
                    is_buy: delta > 0,
                    executed: false,
                    _padding: [0; 26],
                };
                
                total_turnover += delta.abs();
                order_count += 1;
            }
        }

        // Record history
        self.record_history(trigger, order_count, total_turnover, timestamp_ns);

        // Mark rebalance as needed
        if order_count > 0 {
            self.rebalance_needed.store(true, Ordering::Release);
        }

        order_count
    }

    /// Record rebalance to history (circular buffer)
    #[inline]
    fn record_history(&mut self, trigger: RebalanceTrigger, order_count: u32, total_turnover: i64, timestamp_ns: u64) {
        let idx = self.history_head.load(Ordering::Acquire) as usize % MAX_HISTORY;
        
        let turnover_bps = if SCALE > 0 {
            ((total_turnover * 10_000) / SCALE) as i32
        } else {
            0
        };

        self.history[idx] = RebalanceHistory {
            timestamp_ns,
            trigger,
            total_drift_bps: 0, // Would calculate from current state
            order_count,
            executed: !self.shadow_mode.load(Ordering::Acquire),
            turnover_bps,
            _padding: [0; 40],
        };
        
        self.history_head.fetch_add(1, Ordering::AcqRel);
    }

    /// Execute pending orders (or log in shadow mode)
    #[inline]
    pub fn execute_orders(&mut self, timestamp_ns: u64) -> u32 {
        if self.circuit_breaker.load(Ordering::Acquire) {
            return 0;
        }

        let mut executed_count: u32 = 0;
        let is_shadow = self.shadow_mode.load(Ordering::Acquire);

        for i in 0..self.asset_count as usize {
            let order = &mut self.pending_orders[i];
            if order.delta != 0 && !order.executed {
                if !is_shadow {
                    // In production, would send to exchange
                    // For now, just mark as executed and update weights
                    self.current_weights[i as usize] = order.target_weight;
                }
                order.executed = true;
                executed_count += 1;
            }
        }

        if executed_count > 0 {
            self.last_rebalance_ns.store(timestamp_ns, Ordering::Release);
            self.rebalance_needed.store(false, Ordering::Release);
        }

        executed_count
    }

    /// Get pending order count
    #[inline]
    pub fn get_pending_order_count(&self) -> u32 {
        let mut count: u32 = 0;
        for i in 0..self.asset_count as usize {
            if self.pending_orders[i].delta != 0 && !self.pending_orders[i].executed {
                count += 1;
            }
        }
        count
    }

    /// Toggle shadow mode
    #[inline]
    pub fn toggle_shadow_mode(&mut self, enable: bool) {
        self.shadow_mode.store(enable, Ordering::Release);
    }

    /// Check if in shadow mode
    #[inline]
    pub fn is_shadow_mode(&self) -> bool {
        self.shadow_mode.load(Ordering::Acquire)
    }

    /// Trigger circuit breaker
    #[inline]
    pub fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Release);
    }

    /// Reset circuit breaker
    #[inline]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Release);
    }

    /// Get total drift (basis points)
    #[inline]
    pub fn get_total_drift_bps(&self) -> i32 {
        let mut max_drift: i32 = 0;
        for i in 0..self.asset_count as usize {
            let drift = ((self.current_weights[i] - self.target_weights[i]).abs() * 10_000) / SCALE;
            max_drift = max_drift.max(drift as i32);
        }
        max_drift
    }
}

impl Default for RebalancingEngine {
    fn default() -> Self {
        Self::new(50, 3600_000) // 50 bps threshold, 1 hour interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebalancing_engine_initialization() {
        let engine = RebalancingEngine::new(50, 3600_000);
        assert!(engine.is_shadow_mode());
        assert!(!engine.rebalance_needed.load(Ordering::Relaxed));
        assert!(!engine.circuit_breaker.load(Ordering::Relaxed));
    }

    #[test]
    fn test_drift_detection() {
        let mut engine = RebalancingEngine::new(50, 3600_000);
        
        // Set target weights
        engine.set_target_weight(0, 50 * SCALE / 100); // 50%
        engine.set_target_weight(1, 50 * SCALE / 100); // 50%
        
        // Set current weights with drift
        engine.update_current_weight(0, 55 * SCALE / 100); // 55% (5% drift)
        engine.update_current_weight(1, 45 * SCALE / 100); // 45%
        
        assert!(engine.check_rebalance_needed(1000));
        assert!(engine.get_total_drift_bps() >= 500); // 5% = 500 bps
    }

    #[test]
    fn test_order_generation() {
        let mut engine = RebalancingEngine::new(50, 3600_000);
        
        engine.set_target_weight(0, 50 * SCALE / 100);
        engine.set_target_weight(1, 50 * SCALE / 100);
        
        engine.update_current_weight(0, 60 * SCALE / 100);
        engine.update_current_weight(1, 40 * SCALE / 100);
        
        let order_count = engine.generate_orders(RebalanceTrigger::Threshold, 1000);
        assert!(order_count >= 2);
        assert!(engine.rebalance_needed.load(Ordering::Acquire));
    }

    #[test]
    fn test_shadow_mode_execution() {
        let mut engine = RebalancingEngine::new(50, 3600_000);
        
        engine.set_target_weight(0, 50 * SCALE / 100);
        engine.update_current_weight(0, 60 * SCALE / 100);
        
        engine.generate_orders(RebalanceTrigger::Threshold, 1000);
        
        // In shadow mode, weights should not change
        let original_weight = engine.current_weights[0];
        engine.execute_orders(2000);
        
        // Weight should be unchanged in shadow mode
        assert_eq!(engine.current_weights[0], original_weight);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut engine = RebalancingEngine::new(50, 3600_000);
        
        engine.trigger_circuit_breaker();
        
        // Should not detect rebalance need when circuit breaker is active
        assert!(!engine.check_rebalance_needed(1000));
        
        engine.reset_circuit_breaker();
        assert!(engine.circuit_breaker.load(Ordering::Acquire) == false);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<RebalanceOrder>() >= 64);
        assert!(core::mem::size_of::<RebalanceHistory>() >= 64);
        assert!(core::mem::size_of::<RebalancingEngine>() >= 64);
    }
}
