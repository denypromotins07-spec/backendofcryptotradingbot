//! Chapter 4: Streaming Social Sentiment & Explainable AI (XAI)
//! Lightweight, lock-free SHAP value approximator for explaining ML model predictions.

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum features tracked
const MAX_FEATURES: usize = 32;

/// Historical contribution buffer size
const HISTORY_BUFFER_SIZE: usize = 128;

#[repr(C, align(64))]
pub struct ShapValueApproximator {
    /// Feature hashes
    feature_hashes: [AtomicU64; MAX_FEATURES],
    /// Current SHAP value estimates (fixed-point, scaled by 1e9)
    shap_values: [AtomicI64; MAX_FEATURES],
    /// Contribution count per feature
    contribution_count: [AtomicU64; MAX_FEATURES],
    /// Running mean for each feature
    running_mean: [AtomicI64; MAX_FEATURES],
    _padding: [u8; CACHE_LINE_SIZE],
}

#[repr(C, align(64))]
pub struct ContributionHistoryBuffer {
    /// Circular buffer for historical contributions
    /// Stored as flat array: [feature_0_hist, feature_1_hist, ...]
    history: [i64; MAX_FEATURES * HISTORY_BUFFER_SIZE],
    /// Write index per feature
    write_indices: [AtomicU64; MAX_FEATURES],
    /// Total writes per feature
    total_writes: [AtomicU64; MAX_FEATURES],
    _padding: [u8; CACHE_LINE_SIZE - MAX_FEATURES * 2 * 8],
}

#[repr(C, align(64))]
pub struct ShadowModeLogger {
    /// Theoretical sentiment spike records
    theoretical_spikes: [AtomicI64; 64],
    /// Actual signal outputs
    actual_signals: [AtomicI64; 64],
    /// Feature weight validation buffer
    weight_validation: [AtomicI64; MAX_FEATURES],
    /// Record index
    record_idx: AtomicU64,
    /// Validation mode active
    validation_active: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 64 * 2 * 8 - MAX_FEATURES * 8 - 8 - 1],
}

// Compile-time assertions
const _: () = {
    assert!(MAX_FEATURES <= 32, "MAX_FEATURES exceeds limit");
    assert!(HISTORY_BUFFER_SIZE.is_power_of_two(), "History buffer must be power of 2");
};

impl Default for ShapValueApproximator {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            feature_hashes: [INIT_U64; MAX_FEATURES],
            shap_values: [INIT_I64; MAX_FEATURES],
            contribution_count: [INIT_U64; MAX_FEATURES],
            running_mean: [INIT_I64; MAX_FEATURES],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }
}

impl Default for ContributionHistoryBuffer {
    fn default() -> Self {
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        
        Self {
            history: [0i64; MAX_FEATURES * HISTORY_BUFFER_SIZE],
            write_indices: [INIT_U64; MAX_FEATURES],
            total_writes: [INIT_U64; MAX_FEATURES],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_FEATURES * 2 * 8],
        }
    }
}

impl Default for ShadowModeLogger {
    fn default() -> Self {
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        const INIT_BOOL: AtomicBool = AtomicBool::new(false);
        
        Self {
            theoretical_spikes: [INIT_I64; 64],
            actual_signals: [INIT_I64; 64],
            weight_validation: [INIT_I64; MAX_FEATURES],
            record_idx: INIT_U64,
            validation_active: INIT_BOOL,
            _padding: [0u8; CACHE_LINE_SIZE - 64 * 2 * 8 - MAX_FEATURES * 8 - 8 - 1],
        }
    }
}

impl ShapValueApproximator {
    /// Initialize a feature
    #[inline]
    pub fn init_feature(&self, idx: usize, hash: u64) {
        if idx >= MAX_FEATURES {
            return;
        }
        
        self.feature_hashes[idx].store(hash, Ordering::Relaxed);
        self.shap_values[idx].store(0, Ordering::Relaxed);
        self.contribution_count[idx].store(0, Ordering::Relaxed);
        self.running_mean[idx].store(0, Ordering::Relaxed);
    }

    /// Record a contribution (lock-free incremental update)
    /// Uses Welford's online algorithm for numerical stability
    #[inline]
    pub fn record_contribution(&self, feature_idx: usize, contribution: i64) {
        if feature_idx >= MAX_FEATURES {
            return;
        }
        
        let count = self.contribution_count[feature_idx].fetch_add(1, Ordering::Relaxed);
        let new_count = count + 1;
        
        // Welford's online mean update
        let old_mean = self.running_mean[feature_idx].load(Ordering::Relaxed);
        let delta = contribution - old_mean;
        let new_mean = old_mean + delta / new_count as i64;
        
        self.running_mean[feature_idx].store(new_mean, Ordering::Relaxed);
        
        // Update SHAP value estimate (exponential moving average)
        let current_shap = self.shap_values[feature_idx].load(Ordering::Relaxed);
        let alpha = 250_000_000; // 0.25 in fixed-point
        let updated_shap = ((current_shap * (1_000_000_000 - alpha)) / 1_000_000_000) 
                         + ((contribution * alpha) / 1_000_000_000);
        
        self.shap_values[feature_idx].store(updated_shap, Ordering::Relaxed);
    }

    /// Get SHAP value for feature
    #[inline]
    pub fn get_shap_value(&self, feature_idx: usize) -> i64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        self.shap_values[feature_idx].load(Ordering::Relaxed)
    }

    /// Get running mean for feature
    #[inline]
    pub fn get_running_mean(&self, feature_idx: usize) -> i64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        self.running_mean[feature_idx].load(Ordering::Relaxed)
    }

    /// Get contribution count
    #[inline]
    pub fn get_contribution_count(&self, feature_idx: usize) -> u64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        self.contribution_count[feature_idx].load(Ordering::Relaxed)
    }

    /// Get top N features by absolute SHAP value
    #[inline]
    pub fn get_top_features(&self, output: &mut [(usize, i64); 8]) -> usize {
        let mut sorted = [(0usize, 0i64); 8];
        let mut count = 0;
        
        for i in 0..MAX_FEATURES {
            let shap = self.shap_values[i].load(Ordering::Relaxed).abs();
            let hash = self.feature_hashes[i].load(Ordering::Relaxed);
            
            if hash == 0 || shap == 0 {
                continue;
            }
            
            // Insert into sorted array (simple insertion sort for small N)
            let mut pos = count;
            while pos > 0 && sorted[pos - 1].1 < shap {
                if pos < 8 {
                    sorted[pos] = sorted[pos - 1];
                }
                pos -= 1;
            }
            
            if count < 8 {
                if pos < 8 {
                    for j in (pos + 1..count + 1).rev() {
                        if j < 8 {
                            sorted[j] = sorted[j - 1];
                        }
                    }
                }
                sorted[pos.min(7)] = (i, shap);
                count = (count + 1).min(8);
            }
        }
        
        for i in 0..count {
            output[i] = sorted[i];
        }
        
        count
    }

    /// SIMD-accelerated SHAP value normalization
    #[inline]
    pub fn simd_normalize<const N: usize>(&self, indices: [usize; N]) -> [i64; N]
    where [usize; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut values = [0i64; N];
        
        unsafe {
            if N == 4 {
                // Load SHAP values
                for i in 0..4 {
                    values[i] = self.shap_values[indices[i]].load(Ordering::Relaxed);
                }
                
                let val_vec = _mm256_load_si256(values.as_ptr() as *const __m256i);
                
                // Calculate sum of absolute values
                let abs_vec = _mm256_abs_epi64(val_vec);
                
                // Horizontal sum (simplified)
                let vals = [0i64; 4];
                let abs_vals = [0i64; 4];
                _mm256_storeu_si256(vals.as_ptr() as *mut __m256i, val_vec);
                _mm256_storeu_si256(abs_vals.as_ptr() as *mut __m256i, abs_vec);
                
                let total: i64 = abs_vals.iter().sum();
                
                if total != 0 {
                    for i in 0..4 {
                        values[i] = (values[i] * 1_000_000_000) / total;
                    }
                }
                
                _mm256_zeroupper();
            } else {
                let mut total = 0i64;
                for i in 0..N {
                    values[i] = self.shap_values[indices[i]].load(Ordering::Relaxed);
                    total += values[i].abs();
                }
                
                if total != 0 {
                    for i in 0..N {
                        values[i] = (values[i] * 1_000_000_000) / total;
                    }
                }
            }
        }
        
        values
    }

    /// Reset all features
    #[inline]
    pub fn reset(&self) {
        for i in 0..MAX_FEATURES {
            self.shap_values[i].store(0, Ordering::Relaxed);
            self.contribution_count[i].store(0, Ordering::Relaxed);
            self.running_mean[i].store(0, Ordering::Relaxed);
        }
    }
}

impl ContributionHistoryBuffer {
    /// Record contribution to history (lock-free circular)
    #[inline]
    pub fn record(&self, feature_idx: usize, contribution: i64) {
        if feature_idx >= MAX_FEATURES {
            return;
        }
        
        let idx = self.write_indices[feature_idx].load(Ordering::Relaxed);
        let buffer_idx = feature_idx * HISTORY_BUFFER_SIZE + (idx % HISTORY_BUFFER_SIZE) as usize;
        
        unsafe {
            *self.history.get_unchecked_mut(buffer_idx) = contribution;
        }
        
        self.write_indices[feature_idx].fetch_add(1, Ordering::Relaxed);
        self.total_writes[feature_idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Get average contribution over last N samples
    #[inline]
    pub fn get_recent_average(&self, feature_idx: usize, n: u64) -> i64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        
        let total = self.total_writes[feature_idx].load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        
        let count = n.min(HISTORY_BUFFER_SIZE as u64).min(total);
        let write_idx = self.write_indices[feature_idx].load(Ordering::Relaxed);
        
        let mut sum = 0i64;
        for i in 0..count {
            let hist_idx = ((write_idx - i - 1) % HISTORY_BUFFER_SIZE as u64) as usize;
            let buffer_idx = feature_idx * HISTORY_BUFFER_SIZE + hist_idx;
            
            unsafe {
                sum += *self.history.get_unchecked(buffer_idx);
            }
        }
        
        sum / count as i64
    }

    /// Get contribution variance for feature
    #[inline]
    pub fn get_contribution_variance(&self, feature_idx: usize) -> u64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        
        let total = self.total_writes[feature_idx].load(Ordering::Relaxed);
        if total < 2 {
            return 0;
        }
        
        let count = (total as usize).min(HISTORY_BUFFER_SIZE);
        let mean = self.get_recent_average(feature_idx, count as u64);
        
        let mut sq_diff_sum = 0i64;
        for i in 0..count {
            let hist_idx = i % HISTORY_BUFFER_SIZE;
            let buffer_idx = feature_idx * HISTORY_BUFFER_SIZE + hist_idx;
            
            unsafe {
                let val = *self.history.get_unchecked(buffer_idx);
                sq_diff_sum += (val - mean) * (val - mean);
            }
        }
        
        (sq_diff_sum / count as i64) as u64
    }
}

impl ShadowModeLogger {
    /// Enable shadow mode validation
    #[inline]
    pub fn enable_validation(&self) {
        self.validation_active.store(true, Ordering::SeqCst);
    }

    /// Disable shadow mode validation
    #[inline]
    pub fn disable_validation(&self) {
        self.validation_active.store(false, Ordering::SeqCst);
    }

    /// Record theoretical vs actual comparison
    #[inline]
    pub fn record_comparison(&self, theoretical: i64, actual: i64, feature_weights: &[i64]) {
        let idx = self.record_idx.fetch_add(1, Ordering::Relaxed) % 64;
        
        self.theoretical_spikes[idx].store(theoretical, Ordering::Relaxed);
        self.actual_signals[idx].store(actual, Ordering::Relaxed);
        
        // Store feature weights for validation
        for (i, &weight) in feature_weights.iter().enumerate().take(MAX_FEATURES) {
            self.weight_validation[i].store(weight, Ordering::Relaxed);
        }
    }

    /// Calculate validation accuracy (how often theoretical matches actual direction)
    #[inline]
    pub fn get_validation_accuracy(&self) -> i64 {
        let total = self.record_idx.load(Ordering::Relaxed).min(64);
        if total == 0 {
            return 0;
        }
        
        let mut matches = 0i64;
        for i in 0..total {
            let theoretical = self.theoretical_spikes[i as usize].load(Ordering::Relaxed);
            let actual = self.actual_signals[i as usize].load(Ordering::Relaxed);
            
            // Check if signs match (both positive or both negative)
            let same_sign = (theoretical * actual) >= 0;
            matches += same_sign as i64;
        }
        
        (matches * 1_000_000_000) / total
    }

    /// Get stored feature weight
    #[inline]
    pub fn get_feature_weight(&self, feature_idx: usize) -> i64 {
        if feature_idx >= MAX_FEATURES {
            return 0;
        }
        self.weight_validation[feature_idx].load(Ordering::Relaxed)
    }

    /// Check if validation is active
    #[inline]
    pub fn is_validation_active(&self) -> bool {
        self.validation_active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shap_approximator_basic() {
        let shap = ShapValueApproximator::default();
        shap.init_feature(0, 0xFEATURE1);
        shap.init_feature(1, 0xFEATURE2);
        
        // Record contributions
        for _ in 0..100 {
            shap.record_contribution(0, 100_000_000);
            shap.record_contribution(1, -50_000_000);
        }
        
        assert!(shap.get_shap_value(0) > 0);
        assert!(shap.get_shap_value(1) < 0);
        assert_eq!(shap.get_contribution_count(0), 100);
    }

    #[test]
    fn test_top_features() {
        let shap = ShapValueApproximator::default();
        
        for i in 0..8 {
            shap.init_feature(i, i as u64);
            for _ in 0..(10 + i * 10) {
                shap.record_contribution(i, (i as i64) * 100_000_000);
            }
        }
        
        let mut top = [(0usize, 0i64); 8];
        let count = shap.get_top_features(&mut top);
        
        assert!(count >= 1);
        // Highest feature should be first
        assert_eq!(top[0].0, 7);
    }

    #[test]
    fn test_contribution_history() {
        let buffer = ContributionHistoryBuffer::default();
        
        for i in 0..150 {
            buffer.record(0, i as i64 * 100);
        }
        
        let avg = buffer.get_recent_average(0, 100);
        assert!(avg > 0);
        
        let variance = buffer.get_contribution_variance(0);
        assert!(variance > 0);
    }

    #[test]
    fn test_shadow_mode_logger() {
        let logger = ShadowModeLogger::default();
        logger.enable_validation();
        
        assert!(logger.is_validation_active());
        
        // Record some comparisons
        logger.record_comparison(100, 90, &[500_000_000, 300_000_000]);
        logger.record_comparison(-100, -80, &[400_000_000, 200_000_000]);
        logger.record_comparison(100, -50, &[600_000_000, 100_000_000]); // Mismatch
        
        let accuracy = logger.get_validation_accuracy();
        assert!(accuracy > 500_000_000); // Should be > 50% accurate (2/3 match)
    }

    #[test]
    fn test_simd_normalization() {
        let shap = ShapValueApproximator::default();
        
        shap.init_feature(0, 1);
        shap.init_feature(1, 2);
        shap.init_feature(2, 3);
        shap.init_feature(3, 4);
        
        // Set known SHAP values
        shap.shap_values[0].store(100_000_000, Ordering::Relaxed);
        shap.shap_values[1].store(200_000_000, Ordering::Relaxed);
        shap.shap_values[2].store(300_000_000, Ordering::Relaxed);
        shap.shap_values[3].store(400_000_000, Ordering::Relaxed);
        
        let indices = [0, 1, 2, 3];
        let normalized = shap.simd_normalize(indices);
        
        // Sum should be ~1e9 (normalized)
        let total: i64 = normalized.iter().sum();
        assert!(total > 900_000_000 && total < 1_100_000_000);
    }

    #[test]
    fn test_welford_mean_stability() {
        let shap = ShapValueApproximator::default();
        shap.init_feature(0, 0xTEST);
        
        // Record many identical values
        for _ in 0..1000 {
            shap.record_contribution(0, 500_000_000);
        }
        
        let mean = shap.get_running_mean(0);
        assert_eq!(mean, 500_000_000); // Should be exact
    }
}
