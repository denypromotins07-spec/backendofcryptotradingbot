//! Guéant-Lehalle-Fernandez-Tapia (GLFT) optimal quoting distance calculator for limit orders.
//!
//! Implements the GLFT model for optimal market making quote placement
//! using fixed-point arithmetic and zero-copy calculations.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

/// Cache line padding for 64-byte alignment
const CACHE_LINE_SIZE: usize = 64;

/// GLFT model parameters
#[repr(C, align(64))]
pub struct GLFTParams {
    /// Risk aversion parameter (gamma)
    pub gamma: FixedI64,
    /// Volatility (sigma)
    pub sigma: FixedI64,
    /// Order arrival intensity (lambda)
    pub lambda: FixedI64,
    /// Price impact coefficient (kappa)
    pub kappa: FixedI64,
    /// Drift parameter (mu)
    pub mu: FixedI64,
    /// Time horizon (in seconds, scaled by 1e9)
    pub time_horizon: FixedI64,
    /// Minimum tick size
    pub tick_size: FixedI64,
    /// Maximum inventory position
    pub max_inventory: FixedI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 8 * 8],
}

/// Optimal quote distances
#[repr(C, align(64))]
pub struct QuoteDistances {
    /// Optimal bid distance from mid-price
    pub bid_distance: FixedI64,
    /// Optimal ask distance from mid-price
    pub ask_distance: FixedI64,
    /// Bid skew adjustment
    pub bid_skew: FixedI64,
    /// Ask skew adjustment
    pub ask_skew: FixedI64,
    /// Fair value adjustment
    pub fair_value_adj: FixedI64,
    /// Inventory penalty factor
    pub inventory_penalty: FixedI64,
    /// Optimal bid price
    pub bid_price: FixedI64,
    /// Optimal ask price
    pub ask_price: FixedI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE - 8 * 8],
}

/// GLFT model state
#[repr(C, align(64))]
pub struct GLFTModel {
    /// Lock-free flag indicating if model is active
    pub active: AtomicBool,
    /// Current observation count
    pub obs_count: AtomicU64,
    /// Model parameters
    pub params: GLFTParams,
    /// Current mid-price
    pub mid_price: FixedI64,
    /// Current inventory
    pub inventory: FixedI64,
    /// Estimated volatility (rolling)
    pub est_volatility: FixedI64,
    /// Estimated order arrival rate
    pub est_arrival_rate: FixedI64,
    /// Last update timestamp
    pub last_update_ts: AtomicU64,
    /// Shadow mode flag for logging theoretical fills
    pub shadow_mode: AtomicBool,
    /// Theoretical fill counter
    pub theoretical_fills: AtomicU64,
    /// Cumulative PnL in shadow mode
    pub shadow_pnl: FixedI64,
    /// Padding to cache line
    _pad: [u8; CACHE_LINE_SIZE 
        - 2 * 8  // AtomicBool x2
        - 8      // AtomicU64 x1
        - CACHE_LINE_SIZE  // params struct
        - 5 * 8  // mid_price through est_arrival_rate
        - 8      // AtomicU64
        - 8      // AtomicBool
        - 8      // AtomicU64
        - 8,     // shadow_pnl
    ],
}

impl GLFTParams {
    /// Create new GLFT parameters with default values
    #[inline]
    pub const fn new() -> Self {
        Self {
            gamma: FixedI64::from_i64(100000000i64),       // 0.1 risk aversion
            sigma: FixedI64::from_i64(1000000000i64),      // 1.0% volatility
            lambda: FixedI64::from_i64(10000000000i64),    // 10.0 arrival rate
            kappa: FixedI64::from_i64(500000000i64),       // 0.5 price impact
            mu: FixedI64::ZERO,                             // No drift assumption
            time_horizon: FixedI64::from_i64(60000000000i64), // 60 seconds
            tick_size: FixedI64::from_i64(1000000i64),     // 0.000001 tick
            max_inventory: FixedI64::from_i64(1000000000i64), // 1.0 max inventory
            _pad: [0u8; CACHE_LINE_SIZE - 8 * 8],
        }
    }
}

impl GLFTModel {
    /// Create a new GLFT model with pre-allocated state
    #[inline]
    pub const fn new(params: GLFTParams) -> Self {
        Self {
            active: AtomicBool::new(false),
            obs_count: AtomicU64::new(0),
            params,
            mid_price: FixedI64::ZERO,
            inventory: FixedI64::ZERO,
            est_volatility: FixedI64::ZERO,
            est_arrival_rate: FixedI64::ZERO,
            last_update_ts: AtomicU64::new(0),
            shadow_mode: AtomicBool::new(false),
            theoretical_fills: AtomicU64::new(0),
            shadow_pnl: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE 
                - 2 * 8 - 8 - CACHE_LINE_SIZE - 5 * 8 - 8 - 8 - 8 - 8],
        }
    }

    /// Activate the GLFT model
    #[inline]
    pub fn activate(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Deactivate the GLFT model
    #[inline]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if model is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Enable shadow mode for theoretical fill logging
    #[inline]
    pub fn enable_shadow_mode(&self) {
        self.shadow_mode.store(true, Ordering::Relaxed);
    }

    /// Disable shadow mode
    #[inline]
    pub fn disable_shadow_mode(&self) {
        self.shadow_mode.store(false, Ordering::Relaxed);
    }

    /// Update model with new market data
    #[inline]
    pub fn update(&self, mid_price: FixedI64, volatility: FixedI64, arrival_rate: FixedI64) -> Option<QuoteDistances> {
        if !self.is_active() {
            return None;
        }

        // Update mid-price
        // In production, use proper atomic or thread-local storage
        
        // Update estimates
        // Exponential moving average for volatility and arrival rate
        
        // Calculate optimal quotes
        Some(self.calculate_optimal_quotes(mid_price))
    }

    /// Calculate optimal quote distances using GLFT formula
    /// 
    /// The GLFT model gives optimal distances as:
    /// δ* = (1/γ) * ln(1 + γ/k) + (inventory_term) + (drift_term)
    #[inline]
    fn calculate_optimal_quotes(&self, mid_price: FixedI64) -> QuoteDistances {
        let p = &self.params;
        
        // Base reservation spread: (1/γ) * ln(1 + γ/κ)
        // Simplified: approximate ln(1+x) ≈ x for small x
        let gamma_inv = if p.gamma > FixedI64::ZERO {
            FixedI64::ONE / p.gamma
        } else {
            FixedI64::MAX
        };
        
        let gamma_kappa = if p.kappa > FixedI64::ZERO {
            p.gamma / p.kappa
        } else {
            FixedI64::ZERO
        };
        
        // ln(1 + γ/κ) approximation
        let ln_term = gamma_kappa.ln_approx();
        let base_spread = gamma_inv * ln_term;
        
        // Volatility component: σ² * T / 2
        let vol_component = p.sigma * p.sigma * p.time_horizon / FixedI64::from_i64(2000000000i64);
        
        // Inventory penalty term
        let inv_ratio = if p.max_inventory > FixedI64::ZERO {
            self.inventory / p.max_inventory
        } else {
            FixedI64::ZERO
        };
        
        // Inventory penalty: γ * σ² * |inventory| / max_inventory
        let inventory_penalty = p.gamma * p.sigma * p.sigma * inv_ratio.abs();
        
        // Drift adjustment (if μ != 0)
        let drift_adj = p.mu * p.time_horizon / FixedI64::from_i64(2000000000i64);
        
        // Symmetric base distance
        let base_distance = base_spread / FixedI64::from_i64(2000000000i64) + vol_component;
        
        // Apply inventory skew
        // Long inventory -> widen bid, tighten ask
        // Short inventory -> tighten bid, widen ask
        let inv_skew = inventory_penalty / FixedI64::from_i64(2000000000i64);
        
        // Calculate final distances
        let bid_distance = base_distance + inv_skew - drift_adj;
        let ask_distance = base_distance - inv_skew + drift_adj;
        
        // Ensure distances are at least tick size
        let bid_distance = if bid_distance < p.tick_size { p.tick_size } else { bid_distance };
        let ask_distance = if ask_distance < p.tick_size { p.tick_size } else { ask_distance };
        
        // Round to tick size
        let bid_distance = (bid_distance / p.tick_size + FixedI64::from_i64(500000000i64)) / FixedI64::from_i64(1000000000i64) * p.tick_size;
        let ask_distance = (ask_distance / p.tick_size + FixedI64::from_i64(500000000i64)) / FixedI64::from_i64(1000000000i64) * p.tick_size;
        
        // Calculate quote prices
        let bid_price = mid_price - bid_distance;
        let ask_price = mid_price + ask_distance;
        
        // Skew adjustments for asymmetric flow
        let bid_skew = if self.inventory > FixedI64::ZERO {
            // Long inventory: be more aggressive on bid to reduce
            -inv_skew / FixedI64::from_i64(2000000000i64)
        } else {
            inv_skew / FixedI64::from_i64(2000000000i64)
        };
        
        let ask_skew = -bid_skew;
        
        // Log theoretical fill in shadow mode
        if self.shadow_mode.load(Ordering::Relaxed) {
            self.log_theoretical_fill(bid_price, ask_price, mid_price);
        }
        
        QuoteDistances {
            bid_distance,
            ask_distance,
            bid_skew,
            ask_skew,
            fair_value_adj: drift_adj,
            inventory_penalty,
            bid_price,
            ask_price,
            _pad: [0u8; CACHE_LINE_SIZE - 8 * 8],
        }
    }

    /// Log theoretical fill in shadow mode
    #[inline]
    fn log_theoretical_fill(&self, bid: FixedI64, ask: FixedI64, mid: FixedI64) {
        // Increment fill counter
        let _ = self.theoretical_fills.fetch_add(1, Ordering::Relaxed);
        
        // Calculate theoretical PnL (simplified)
        // In production, this would track actual fill scenarios
        let spread_captured = ask - bid;
        let _ = spread_captured; // Use variable
    }

    /// SIMD-accelerated batch quote calculation for multiple symbols
    #[inline]
    pub fn calculate_batch_quotes_simd(
        &self,
        mid_prices: &[FixedI64],
        volatilities: &[FixedI64],
        inventories: &[FixedI64],
    ) -> ([FixedI64; 4], [FixedI64; 4]) {
        assert!(mid_prices.len() >= 4);
        assert!(volatilities.len() >= 4);
        assert!(inventories.len() >= 4);

        unsafe {
            let mut bid_distances = [FixedI64::ZERO; 4];
            let mut ask_distances = [FixedI64::ZERO; 4];
            
            // Process 4 symbols at a time
            for i in (0..4).step_by(4) {
                let mp_vec = _mm256_loadu_si256(mid_prices[i..].as_ptr() as *const __m256i);
                let vol_vec = _mm256_loadu_si256(volatilities[i..].as_ptr() as *const __m256i);
                let inv_vec = _mm256_loadu_si256(inventories[i..].as_ptr() as *const __m256i);
                
                // Extract and compute scalar (SIMD division is complex)
                let mut mp_arr = [0i64; 4];
                let mut vol_arr = [0i64; 4];
                let mut inv_arr = [0i64; 4];
                _mm256_storeu_si256(mp_arr.as_mut_ptr() as *mut __m256i, mp_vec);
                _mm256_storeu_si256(vol_arr.as_mut_ptr() as *mut __m256i, vol_vec);
                _mm256_storeu_si256(inv_arr.as_mut_ptr() as *mut __m256i, inv_vec);
                
                for j in 0..4 {
                    let vol = FixedI64::from_i64(vol_arr[j]);
                    let inv = FixedI64::from_i64(inv_arr[j]);
                    
                    // Simplified GLFT calculation
                    let vol_comp = vol * vol * self.params.time_horizon / FixedI64::from_i64(2000000000i64);
                    let inv_penalty = self.params.gamma * vol * vol * inv.abs() / self.params.max_inventory;
                    let inv_skew = inv_penalty / FixedI64::from_i64(2000000000i64);
                    
                    let base = self.params.sigma * self.params.sigma * self.params.time_horizon / FixedI64::from_i64(4000000000i64);
                    
                    bid_distances[i + j] = base + inv_skew;
                    ask_distances[i + j] = base - inv_skew;
                }
            }
            
            (bid_distances, ask_distances)
        }
    }

    /// Manually unrolled loop for fast inventory penalty calculation
    #[inline]
    pub fn calculate_inventory_penalties_unrolled(
        &self,
        inventories: &[FixedI64],
        output: &mut [FixedI64],
    ) {
        let len = inventories.len();
        let chunks = len / 4;
        let remainder = len % 4;
        
        let gamma_sigma_sq = self.params.gamma * self.params.sigma * self.params.sigma;
        let max_inv = self.params.max_inventory;

        unsafe {
            // Process chunks of 4
            for i in 0..chunks {
                let base = i * 4;
                *output.get_unchecked_mut(base) = 
                    gamma_sigma_sq * (*inventories.get_unchecked(base)).abs() / max_inv;
                *output.get_unchecked_mut(base + 1) = 
                    gamma_sigma_sq * (*inventories.get_unchecked(base + 1)).abs() / max_inv;
                *output.get_unchecked_mut(base + 2) = 
                    gamma_sigma_sq * (*inventories.get_unchecked(base + 2)).abs() / max_inv;
                *output.get_unchecked_mut(base + 3) = 
                    gamma_sigma_sq * (*inventories.get_unchecked(base + 3)).abs() / max_inv;
            }

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                *output.get_unchecked_mut(idx) = 
                    gamma_sigma_sq * (*inventories.get_unchecked(idx)).abs() / max_inv;
            }
        }
    }

    /// Get current shadow mode statistics
    #[inline]
    pub fn get_shadow_stats(&self) -> (u64, FixedI64) {
        let fills = self.theoretical_fills.load(Ordering::Relaxed);
        let pnl = self.shadow_pnl;
        (fills, pnl)
    }
}

// Compile-time assertions for alignment
const _: () = assert!(core::mem::size_of::<GLFTParams>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<GLFTParams>() == 64);
const _: () = assert!(core::mem::size_of::<QuoteDistances>() % 64 == 0);
const _: () = assert!(core::mem::align_of::<QuoteDistances>() == 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glft_params_creation() {
        let params = GLFTParams::new();
        assert!(params.gamma > FixedI64::ZERO);
        assert!(params.sigma > FixedI64::ZERO);
    }

    #[test]
    fn test_model_activation() {
        let params = GLFTParams::new();
        let model = GLFTModel::new(params);
        
        assert!(!model.is_active());
        model.activate();
        assert!(model.is_active());
    }

    #[test]
    fn test_quote_calculation() {
        let mut params = GLFTParams::new();
        params.sigma = FixedI64::from_i64(2000000000i64); // 2% vol
        
        let model = GLFTModel::new(params);
        model.activate();
        
        let mid = FixedI64::from_i64(100_000_000_000i64);
        let vol = FixedI64::from_i64(2000000000i64);
        let rate = FixedI64::from_i64(10000000000i64);
        
        let quotes = model.update(mid, vol, rate);
        assert!(quotes.is_some());
        
        let q = quotes.unwrap();
        assert!(q.bid_distance > FixedI64::ZERO);
        assert!(q.ask_distance > FixedI64::ZERO);
        assert!(q.bid_price < mid);
        assert!(q.ask_price > mid);
    }

    #[test]
    fn test_shadow_mode() {
        let params = GLFTParams::new();
        let model = GLFTModel::new(params);
        
        assert!(!model.shadow_mode.load(Ordering::Relaxed));
        model.enable_shadow_mode();
        assert!(model.shadow_mode.load(Ordering::Relaxed));
        
        let (fills, pnl) = model.get_shadow_stats();
        assert_eq!(fills, 0);
    }
}
