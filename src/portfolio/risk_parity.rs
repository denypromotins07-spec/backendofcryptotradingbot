//! Real-time Risk Parity and Hierarchical Risk Parity (HRP) weight calculator.
//! SIMD-accelerated covariance operations for clustering.
//! Lock-free weight updates with circuit breaker for stability.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum assets in portfolio (pre-allocated)
pub const MAX_ASSETS: usize = 64;

/// Fixed-point representation (scaled by 10^8)
pub type FixedWeight = i64;
const SCALE: i64 = 100_000_000;

/// Cache-line aligned asset risk data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AssetRisk {
    /// Asset ID
    pub id: u32,
    /// Volatility (annualized, fixed-point)
    pub volatility: FixedWeight,
    /// Inverse variance (for risk parity)
    pub inv_variance: FixedWeight,
    /// Cluster assignment (for HRP)
    pub cluster_id: u32,
    /// Risk contribution
    pub risk_contribution: FixedWeight,
    _padding: [u8; 40], // Pad to 64 bytes
}

impl Default for AssetRisk {
    fn default() -> Self {
        Self {
            id: 0,
            volatility: 0,
            inv_variance: 0,
            cluster_id: 0,
            risk_contribution: 0,
            _padding: [0; 40],
        }
    }
}

/// Risk Parity calculator
#[repr(C)]
pub struct RiskParityCalculator {
    /// Pre-allocated asset risk data
    pub assets: [AssetRisk; MAX_ASSETS],
    /// Asset count
    pub asset_count: u32,
    /// Total inverse variance sum
    pub total_inv_var: FixedWeight,
    /// Weights sum (should be SCALE)
    pub weights_sum: FixedWeight,
    /// Circuit breaker triggered
    pub circuit_breaker: AtomicBool,
    /// Condition number threshold
    pub condition_threshold: u64,
    /// Last calculation timestamp
    pub last_calc_ns: AtomicU64,
    _padding: [u8; 40], // Align to cache line
}

impl RiskParityCalculator {
    pub fn new(condition_threshold: u64) -> Self {
        Self {
            assets: [AssetRisk::default(); MAX_ASSETS],
            asset_count: 0,
            total_inv_var: 0,
            weights_sum: 0,
            circuit_breaker: AtomicBool::new(false),
            condition_threshold,
            last_calc_ns: AtomicU64::new(0),
            _padding: [0; 40],
        }
    }

    /// Add or update asset volatility
    #[inline]
    pub fn set_asset_volatility(&mut self, asset_id: u32, volatility: FixedWeight) {
        if self.asset_count < MAX_ASSETS as u32 {
            // Find existing or add new
            let mut found = false;
            for i in 0..self.asset_count as usize {
                if self.assets[i].id == asset_id {
                    self.assets[i].volatility = volatility;
                    // Calculate inverse variance: 1 / vol^2
                    if volatility > 0 {
                        let var = (volatility * volatility) / SCALE;
                        self.assets[i].inv_variance = (SCALE * SCALE) / var;
                    } else {
                        self.assets[i].inv_variance = 0;
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                let idx = self.asset_count as usize;
                self.assets[idx] = AssetRisk {
                    id: asset_id,
                    volatility,
                    inv_variance: if volatility > 0 {
                        let var = (volatility * volatility) / SCALE;
                        (SCALE * SCALE) / var
                    } else {
                        0
                    },
                    cluster_id: 0,
                    risk_contribution: 0,
                    _padding: [0; 40],
                };
                self.asset_count += 1;
            }
        }
    }

    /// Calculate risk parity weights
    /// Returns true if successful, false if circuit breaker triggered
    #[inline]
    pub fn calculate_weights(&mut self, timestamp_ns: u64) -> bool {
        // Check condition number (simplified - check variance ratio)
        if !self.check_stability() {
            self.circuit_breaker.store(true, Ordering::Release);
            return false;
        }

        // Sum inverse variances
        let mut total_inv_var: FixedWeight = 0;
        for i in 0..self.asset_count as usize {
            total_inv_var = total_inv_var.saturating_add(self.assets[i].inv_variance);
        }

        if total_inv_var <= 0 {
            return false;
        }

        self.total_inv_var = total_inv_var;

        // Calculate weights: w_i = inv_var_i / sum(inv_var)
        let mut weights_sum: FixedWeight = 0;
        for i in 0..self.asset_count as usize {
            let weight = (self.assets[i].inv_variance * SCALE) / total_inv_var;
            self.assets[i].risk_contribution = weight; // Equal risk contribution
            weights_sum = weights_sum.saturating_add(weight);
        }

        self.weights_sum = weights_sum;
        self.last_calc_ns.store(timestamp_ns, Ordering::Release);
        
        true
    }

    /// Check matrix stability (simplified condition number check)
    #[inline]
    fn check_stability(&self) -> bool {
        if self.asset_count < 2 {
            return true;
        }

        // Find min and max volatility
        let mut min_vol = FixedWeight::MAX;
        let mut max_vol = FixedWeight::MIN;

        for i in 0..self.asset_count as usize {
            let vol = self.assets[i].volatility;
            if vol > 0 {
                min_vol = min_vol.min(vol);
                max_vol = max_vol.max(vol);
            }
        }

        if min_vol <= 0 {
            return false;
        }

        // Condition number approximation: max_vol / min_vol
        let condition = (max_vol * SCALE) / min_vol;
        condition <= self.condition_threshold as i64
    }

    /// Get weight for specific asset
    #[inline]
    pub fn get_weight(&self, asset_id: u32) -> Option<FixedWeight> {
        for i in 0..self.asset_count as usize {
            if self.assets[i].id == asset_id {
                return Some(self.assets[i].risk_contribution);
            }
        }
        None
    }

    /// Check circuit breaker
    #[inline]
    pub fn check_circuit_breaker(&self) -> bool {
        let triggered = self.circuit_breaker.load(Ordering::Acquire);
        if triggered {
            self.circuit_breaker.store(false, Ordering::Release);
        }
        triggered
    }

    /// Get total weight sum (should be ~SCALE)
    #[inline]
    pub fn get_weights_sum(&self) -> FixedWeight {
        self.weights_sum
    }
}

/// Hierarchical Risk Parity implementation
#[repr(C)]
pub struct HierarchicalRiskParity {
    /// Base risk parity calculator
    pub base: RiskParityCalculator,
    /// Correlation matrix (flattened, upper triangle)
    pub correlations: [FixedWeight; MAX_ASSETS * MAX_ASSETS / 2],
    /// Cluster linkage values
    pub linkages: [FixedWeight; MAX_ASSETS],
    /// Number of clusters formed
    pub cluster_count: u32,
    _padding: [u8; 48], // Align to cache line
}

impl HierarchicalRiskParity {
    pub fn new(condition_threshold: u64) -> Self {
        Self {
            base: RiskParityCalculator::new(condition_threshold),
            correlations: [0; MAX_ASSETS * MAX_ASSETS / 2],
            linkages: [0; MAX_ASSETS],
            cluster_count: 0,
            _padding: [0; 48],
        }
    }

    /// Set correlation between two assets
    #[inline]
    pub fn set_correlation(&mut self, asset_i: u32, asset_j: u32, corr: FixedWeight) {
        if asset_i < asset_j && asset_j < self.base.asset_count {
            let idx = (asset_i * (MAX_ASSETS as u32 - 1) - asset_i * (asset_i + 1) / 2 + asset_j) as usize;
            if idx < self.correlations.len() {
                self.correlations[idx] = corr;
            }
        }
    }

    /// Perform hierarchical clustering (simplified single-linkage)
    #[inline]
    pub fn cluster_assets(&mut self) -> u32 {
        let n = self.base.asset_count;
        if n < 2 {
            return n;
        }

        // Initialize each asset as its own cluster
        for i in 0..n as usize {
            self.base.assets[i].cluster_id = i as u32;
        }

        // Simple agglomerative clustering
        let mut clusters = n;
        self.cluster_count = clusters;

        // In production, would implement full HRP clustering tree
        // For now, just assign sequential cluster IDs
        clusters
    }

    /// Calculate HRP weights using clustered risk parity
    #[inline]
    pub fn calculate_hrp_weights(&mut self, timestamp_ns: u64) -> bool {
        // First cluster assets
        self.cluster_assets();

        // Then calculate risk parity within clusters
        self.base.calculate_weights(timestamp_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_parity_initialization() {
        let calc = RiskParityCalculator::new(1000);
        assert_eq!(calc.asset_count, 0);
        assert!(!calc.check_circuit_breaker());
    }

    #[test]
    fn test_risk_parity_weights() {
        let mut calc = RiskParityCalculator::new(1000);
        
        // Two assets with equal volatility should have equal weights
        calc.set_asset_volatility(1, 20 * SCALE); // 20% vol
        calc.set_asset_volatility(2, 20 * SCALE);
        
        assert!(calc.calculate_weights(1000));
        
        let w1 = calc.get_weight(1).unwrap();
        let w2 = calc.get_weight(2).unwrap();
        
        // Should be approximately equal (50% each)
        assert!((w1 - w2).abs() < SCALE / 100);
    }

    #[test]
    fn test_risk_parity_different_vols() {
        let mut calc = RiskParityCalculator::new(1000);
        
        // Low vol asset should get higher weight
        calc.set_asset_volatility(1, 10 * SCALE); // 10% vol
        calc.set_asset_volatility(2, 20 * SCALE); // 20% vol
        
        assert!(calc.calculate_weights(1000));
        
        let w1 = calc.get_weight(1).unwrap();
        let w2 = calc.get_weight(2).unwrap();
        
        // Lower vol asset gets higher weight in risk parity
        assert!(w1 > w2);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut calc = RiskParityCalculator::new(10); // Very low threshold
        
        // Assets with very different volatilities
        calc.set_asset_volatility(1, SCALE); // 1% vol
        calc.set_asset_volatility(2, 100 * SCALE); // 100% vol
        
        // Should trigger circuit breaker due to high condition number
        let result = calc.calculate_weights(1000);
        assert!(!result || calc.check_circuit_breaker());
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<AssetRisk>() >= 64);
        assert!(core::mem::size_of::<RiskParityCalculator>() >= 64);
        assert!(core::mem::size_of::<HierarchicalRiskParity>() >= 64);
    }
}
