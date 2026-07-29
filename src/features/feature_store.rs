//! Lock-Free Memory-Mapped Feature Store
//! Consistent online/offline feature serving with zero-copy access.
//! Pre-allocated circular buffer for historical features.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum number of features
const MAX_FEATURES: usize = 1024;
/// Maximum history length (circular buffer)
const MAX_HISTORY: usize = 8192;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// Store valid flag
static STORE_VALID: AtomicBool = AtomicBool::new(false);

/// Feature metadata - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FeatureMeta {
    /// Feature name hash (FNV-1a)
    pub name_hash: u64,
    /// Feature type (0=float, 1=int, 2=bool)
    pub dtype: u8,
    /// Is normalized
    pub is_normalized: bool,
    /// Mean for normalization (scaled by 10^8)
    pub mean: i64,
    /// Std for normalization (scaled by 10^8)
    pub std: i64,
    /// Min value (scaled)
    pub min_val: i64,
    /// Max value (scaled)
    pub max_val: i64,
    /// Last update timestamp
    pub last_update: u64,
    _pad: [u8; 22],
}

impl Default for FeatureMeta {
    fn default() -> Self {
        Self {
            name_hash: 0,
            dtype: 0,
            is_normalized: false,
            mean: 0,
            std: 100_000_000, // 1.0 scaled
            min_val: i64::MIN,
            max_val: i64::MAX,
            last_update: 0,
            _pad: [0u8; 22],
        }
    }
}

const _: () = assert!(core::mem::size_of::<FeatureMeta>() == 64);

/// Feature store with circular history buffer
#[repr(C)]
pub struct FeatureStore {
    /// Current feature values (scaled by 10^8)
    pub current_values: [i64; MAX_FEATURES],
    /// Historical values: [feature][history_idx]
    pub history: [[i64; MAX_HISTORY]; MAX_FEATURES],
    /// Feature metadata
    pub metadata: [FeatureMeta; MAX_FEATURES],
    /// Write index for circular buffer
    pub write_idx: AtomicU64,
    /// Number of active features
    pub num_features: usize,
    /// Valid mask (bitmask of valid features)
    pub valid_mask: [u64; 16], // 16 * 64 = 1024 bits
    /// Store version
    pub version: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self {
            current_values: [0; MAX_FEATURES],
            history: [[0; MAX_HISTORY]; MAX_FEATURES],
            metadata: [FeatureMeta::default(); MAX_FEATURES],
            write_idx: AtomicU64::new(0),
            num_features: 0,
            valid_mask: [0; 16],
            version: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl FeatureStore {
    /// Create new feature store
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Register a new feature (returns feature index or None if full)
    pub fn register_feature(&self, name_hash: u64, dtype: u8) -> Option<usize> {
        if !STORE_VALID.load(Ordering::Relaxed) {
            return None;
        }
        
        // Find empty slot using valid mask
        for block in 0..16 {
            let mask = self.valid_mask[block];
            if mask != u64::MAX {
                // Find first zero bit
                let bit = mask.trailing_ones() as usize;
                if bit < 64 {
                    let idx = block * 64 + bit;
                    if idx >= MAX_FEATURES {
                        return None;
                    }
                    
                    unsafe {
                        let mask_ptr = self.valid_mask.as_ptr() as *mut u64;
                        *mask_ptr.add(block) |= 1u64 << bit;
                        
                        let meta_ptr = self.metadata.as_ptr() as *mut FeatureMeta;
                        *meta_ptr.add(idx) = FeatureMeta {
                            name_hash,
                            dtype,
                            ..FeatureMeta::default()
                        };
                    }
                    
                    let count = self.num_features;
                    unsafe {
                        let ptr = &self.num_features as *const usize as *mut usize;
                        *ptr = count + 1;
                    }
                    
                    return Some(idx);
                }
            }
        }
        
        None
    }
    
    /// Set feature value (lock-free, scaled)
    #[inline(always)]
    pub fn set_value(&self, feature_idx: usize, value: i64) {
        if feature_idx >= MAX_FEATURES {
            return;
        }
        
        if !self.is_valid_feature(feature_idx) {
            return;
        }
        
        // Update current value
        unsafe {
            let val_ptr = self.current_values.as_ptr() as *mut i64;
            *val_ptr.add(feature_idx) = value;
        }
        
        // Update history (circular buffer)
        let write_idx = self.write_idx.load(Ordering::Relaxed) % MAX_HISTORY as u64;
        unsafe {
            let hist_ptr = self.history.as_ptr() as *mut [i64; MAX_HISTORY];
            (*hist_ptr.add(feature_idx))[write_idx as usize] = value;
        }
        
        // Update metadata stats (online mean/variance)
        self.update_stats(feature_idx, value);
        
        // Increment write index
        self.write_idx.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current feature value
    #[inline(always)]
    pub fn get_value(&self, feature_idx: usize) -> i64 {
        if feature_idx >= MAX_FEATURES || !self.is_valid_feature(feature_idx) {
            return 0;
        }
        self.current_values[feature_idx]
    }
    
    /// Get normalized feature value
    #[inline(always)]
    pub fn get_normalized(&self, feature_idx: usize) -> f64 {
        let value = self.get_value(feature_idx) as f64;
        let meta = &self.metadata[feature_idx];
        
        if !meta.is_normalized || meta.std == 0 {
            return value / 100_000_000.0;
        }
        
        (value - meta.mean as f64) / meta.std as f64
    }
    
    /// Get historical value at offset (0 = current, 1 = previous, etc.)
    #[inline(always)]
    pub fn get_history(&self, feature_idx: usize, offset: usize) -> i64 {
        if feature_idx >= MAX_FEATURES || offset >= MAX_HISTORY {
            return 0;
        }
        
        let current = self.write_idx.load(Ordering::Relaxed) % MAX_HISTORY as u64;
        let idx = ((current as i64 - offset as i64 + MAX_HISTORY as i64) % MAX_HISTORY as i64) as usize;
        
        self.history[feature_idx][idx]
    }
    
    /// Check if feature is valid
    #[inline(always)]
    fn is_valid_feature(&self, idx: usize) -> bool {
        let block = idx / 64;
        let bit = idx % 64;
        (self.valid_mask[block] & (1u64 << bit)) != 0
    }
    
    /// Update running statistics (Welford's algorithm)
    fn update_stats(&self, feature_idx: usize, value: i64) {
        let meta = &self.metadata[feature_idx];
        let count = meta.last_update + 1;
        let n = count as f64;
        
        let delta = value as f64 - meta.mean as f64;
        let new_mean = meta.mean as f64 + delta / n;
        let new_var = ((meta.std as f64).powi(2) * (n - 1.0) + delta * (value as f64 - new_mean)) / n;
        
        unsafe {
            let meta_ptr = self.metadata.as_ptr() as *mut FeatureMeta;
            let m = &mut *meta_ptr.add(feature_idx);
            m.mean = new_mean as i64;
            m.std = (new_var.sqrt() * 100_000_000.0) as i64;
            m.min_val = m.min_val.min(value);
            m.max_val = m.max_val.max(value);
            m.last_update = count;
        }
    }
    
    /// Mark store as valid
    pub fn mark_valid(&self) {
        self.valid.store(true, Ordering::Release);
        STORE_VALID.store(true, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Release);
    }
    
    /// Invalidate store
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        STORE_VALID.store(false, Ordering::Relaxed);
    }
    
    /// Get feature count
    pub fn feature_count(&self) -> usize {
        self.num_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_feature_registration(name_hash in any::<u64>(), dtype in 0u8..3u8) {
            let store = FeatureStore::new();
            store.mark_valid();
            
            if let Some(idx) = store.register_feature(name_hash, dtype) {
                assert!(idx < MAX_FEATURES);
                assert!(store.is_valid_feature(idx));
                assert_eq!(store.metadata[idx].name_hash, name_hash);
                assert_eq!(store.metadata[idx].dtype, dtype);
            }
        }
        
        #[test]
        fn test_value_roundtrip(
            feature_idx in 0usize..100,
            value in -1_000_000_000i64..1_000_000_000i64,
        ) {
            let store = FeatureStore::new();
            store.mark_valid();
            
            // Register enough features
            for i in 0..feature_idx + 1 {
                store.register_feature(i as u64, 0);
            }
            
            store.set_value(feature_idx, value);
            assert_eq!(store.get_value(feature_idx), value);
        }
        
        #[test]
        fn test_history_circular_buffer(
            value in 0i64..100i64,
            offset in 0usize..100,
        ) {
            let store = FeatureStore::new();
            store.mark_valid();
            store.register_feature(0, 0);
            
            // Write some values
            for i in 0..offset + 1 {
                store.set_value(0, value + i as i64);
            }
            
            // Read back
            let retrieved = store.get_history(0, offset);
            assert_eq!(retrieved, value);
        }
    }
    
    #[test]
    fn test_feature_meta_size() {
        assert_eq!(core::mem::size_of::<FeatureMeta>(), 64);
    }
    
    #[test]
    fn test_store_validity() {
        let store = FeatureStore::new();
        assert!(!store.valid.load(Ordering::Relaxed));
        store.mark_valid();
        assert!(store.valid.load(Ordering::Relaxed));
        store.invalidate();
        assert!(!store.valid.load(Ordering::Relaxed));
    }
}
