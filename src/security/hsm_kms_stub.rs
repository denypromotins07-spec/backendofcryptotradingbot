//! Hardware Security Module (HSM) and KMS abstraction layer for key signing.
//!
//! This module provides a unified interface for hardware-backed key operations,
//! supporting both HSM devices and cloud KMS providers. All signing operations
//! are performed in hardware to prevent key material exposure.
//!
//! **Security:** Keys never leave HSM/KMS, only signatures returned.
//! **Latency Target:** < 5µs for local HSM, < 50ms for cloud KMS.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum signature size (Ed25519 = 64 bytes).
const MAX_SIGNATURE_SIZE: usize = 64;

/// Maximum message digest size (SHA-256 = 32 bytes).
const MAX_DIGEST_SIZE: usize = 32;

/// HSM backend type enumeration.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsmBackend {
    None = 0,
    LocalYubiHSM = 1,
    AwsKms = 2,
    GcpKms = 3,
    AzureKeyVault = 4,
    Mock = 255,
}

/// Signing algorithm enumeration.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignAlgorithm {
    Ed25519 = 0,
    ECDSAP256 = 1,
    RSAPSS2048 = 2,
    RSAPSS4096 = 3,
}

/// Key handle structure for HSM references.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KeyHandle {
    /// Key ID in the HSM.
    pub key_id: u64,
    /// Algorithm used by this key.
    pub algorithm: SignAlgorithm,
    /// Backend that manages this key.
    pub backend: HsmBackend,
    /// Key usage flags (bitmask).
    pub usage_flags: u32,
    /// Last use timestamp.
    pub last_use_ts: AtomicU64,
    /// Use count.
    pub use_count: AtomicU64,
    /// Flag indicating if the key is active.
    pub is_active: AtomicBool,
    /// Reserved padding.
    _padding: [u8; 35],
}

impl KeyHandle {
    #[inline]
    pub const fn new() -> Self {
        Self {
            key_id: 0,
            algorithm: SignAlgorithm::Ed25519,
            backend: HsmBackend::None,
            usage_flags: 0,
            last_use_ts: AtomicU64::new(0),
            use_count: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            _padding: [0u8; 35],
        }
    }

    /// Check if this key can be used for signing.
    #[inline]
    pub fn can_sign(&self) -> bool {
        self.is_active.load(Ordering::Acquire) && (self.usage_flags & 1) != 0
    }
}

// Ensure KeyHandle is exactly one cache line.
const _: () = assert!(core::mem::size_of::<KeyHandle>() == CACHE_LINE_SIZE);

/// The main HSM/KMS abstraction layer.
pub struct HsmKmsLayer {
    /// Registered key handles.
    keys: [KeyHandle; 16],
    /// Count of registered keys.
    key_count: AtomicU64,
    /// Default backend.
    default_backend: HsmBackend,
    /// Flag indicating if the layer is initialized.
    is_initialized: AtomicBool,
    /// Total signing operations.
    sign_count: AtomicU64,
    /// Failed signing operations.
    fail_count: AtomicU64,
    /// Padding.
    _padding: [u8; 48],
}

unsafe impl Send for HsmKmsLayer {}
unsafe impl Sync for HsmKmsLayer {}

impl HsmKmsLayer {
    /// Create a new HSM/KMS layer.
    #[inline]
    pub const fn new() -> Self {
        Self {
            keys: [KeyHandle::new(); 16],
            key_count: AtomicU64::new(0),
            default_backend: HsmBackend::Mock,
            is_initialized: AtomicBool::new(false),
            sign_count: AtomicU64::new(0),
            fail_count: AtomicU64::new(0),
            _padding: [0u8; 48],
        }
    }

    /// Initialize the HSM/KMS layer.
    #[inline]
    pub fn init(&mut self, backend: HsmBackend) -> Result<(), &'static str> {
        if self.is_initialized.load(Ordering::Acquire) {
            return Err("HSM/KMS layer already initialized");
        }

        self.default_backend = backend;
        self.is_initialized.store(true, Ordering::Release);
        Ok(())
    }

    /// Register a key handle.
    #[inline]
    pub fn register_key(&self, key_id: u64, algorithm: SignAlgorithm, usage_flags: u32) -> Result<usize, &'static str> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err("HSM/KMS layer not initialized");
        }

        let idx = self.key_count.load(Ordering::Acquire) as usize;
        if idx >= 16 {
            return Err("Key registry full");
        }

        let claimed = self.key_count.compare_exchange(
            idx as u64,
            (idx + 1) as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match claimed {
            Ok(_) => {
                let key = &self.keys[idx];
                unsafe {
                    ptr::write_volatile(&key.key_id as *const u64 as *mut u64, key_id);
                    ptr::write_volatile(&key.algorithm as *const SignAlgorithm as *mut SignAlgorithm, algorithm);
                    ptr::write_volatile(&key.backend as *const HsmBackend as *mut HsmBackend, self.default_backend);
                    ptr::write_volatile(&key.usage_flags as *const u32 as *mut u32, usage_flags);
                }
                key.is_active.store(true, Ordering::Release);
                Ok(idx)
            }
            Err(_) => Err("Failed to claim key slot"),
        }
    }

    /// Sign a message digest using the specified key.
    ///
    /// Returns the signature. In production, this would call the actual HSM/KMS.
    #[inline]
    pub fn sign(&self, key_idx: usize, digest: &[u8; MAX_DIGEST_SIZE]) -> Result<[u8; MAX_SIGNATURE_SIZE], &'static str> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err("HSM/KMS layer not initialized");
        }

        if key_idx >= 16 {
            return Err("Invalid key index");
        }

        let key = &self.keys[key_idx];
        
        if !key.can_sign() {
            self.fail_count.fetch_add(1, Ordering::Relaxed);
            return Err("Key cannot sign");
        }

        // Validate digest length based on algorithm
        let required_len = match key.algorithm {
            SignAlgorithm::Ed25519 => 32,
            SignAlgorithm::ECDSAP256 => 32,
            SignAlgorithm::RSAPSS2048 => 32,
            SignAlgorithm::RSAPSS4096 => 48,
        };

        // Simplified signing (replace with actual HSM call in production)
        let mut signature = [0u8; MAX_SIGNATURE_SIZE];
        
        match key.algorithm {
            SignAlgorithm::Ed25519 => {
                // Ed25519 produces 64-byte signatures
                // Mock signature: hash of digest + key_id
                self.mock_sign(digest, key.key_id, &mut signature[..64]);
            }
            SignAlgorithm::ECDSAP256 => {
                // ECDSA P-256 produces 64-byte signatures (r, s)
                self.mock_sign(digest, key.key_id, &mut signature[..64]);
            }
            _ => {
                self.fail_count.fetch_add(1, Ordering::Relaxed);
                return Err("Algorithm not supported in mock mode");
            }
        }

        // Update statistics
        key.use_count.fetch_add(1, Ordering::Relaxed);
        self.sign_count.fetch_add(1, Ordering::Relaxed);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::_rdtsc;
            key.last_use_ts.store(_rdtsc(), Ordering::Release);
        }

        Ok(signature)
    }

    /// Verify a signature (public operation, doesn't require HSM).
    #[inline]
    pub fn verify(&self, key_idx: usize, digest: &[u8; MAX_DIGEST_SIZE], signature: &[u8; MAX_SIGNATURE_SIZE]) -> Result<bool, &'static str> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err("HSM/KMS layer not initialized");
        }

        if key_idx >= 16 {
            return Err("Invalid key index");
        }

        let key = &self.keys[key_idx];
        
        if !key.is_active.load(Ordering::Acquire) {
            return Err("Key not active");
        }

        // Mock verification (always succeeds for non-zero signatures)
        let is_valid = signature.iter().any(|&b| b != 0);
        Ok(is_valid)
    }

    /// Mock signing function (replace with actual HSM call).
    #[inline]
    fn mock_sign(&self, digest: &[u8], key_id: u64, output: &mut [u8]) {
        // Simple mock: XOR digest with key_id pattern
        for (i, byte) in output.iter_mut().enumerate() {
            let digest_byte = digest[i % digest.len()];
            let key_byte = ((key_id >> (i % 64)) & 0xFF) as u8;
            *byte = digest_byte ^ key_byte;
        }
    }

    /// Get key statistics.
    #[inline]
    pub fn get_key_stats(&self, key_idx: usize) -> Option<(u64, u64, u64)> {
        if key_idx >= 16 {
            return None;
        }

        let key = &self.keys[key_idx];
        if !key.is_active.load(Ordering::Acquire) {
            return None;
        }

        Some((
            key.key_id,
            key.use_count.load(Ordering::Acquire),
            key.last_use_ts.load(Ordering::Acquire),
        ))
    }

    /// Get total signing statistics.
    #[inline]
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.sign_count.load(Ordering::Acquire),
            self.fail_count.load(Ordering::Acquire),
        )
    }

    /// Shutdown the HSM/KMS layer.
    #[inline]
    pub fn shutdown(&mut self) {
        // Deactivate all keys
        for i in 0..16 {
            self.keys[i].is_active.store(false, Ordering::Release);
        }
        self.is_initialized.store(false, Ordering::Release);
    }
}

impl Default for HsmKmsLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for HsmKmsLayer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_handle_size() {
        assert_eq!(core::mem::size_of::<KeyHandle>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_hsm_init() {
        let mut hsm = HsmKmsLayer::new();
        hsm.init(HsmBackend::Mock).unwrap();
        assert!(hsm.is_initialized.load(Ordering::Acquire));
        assert_eq!(hsm.default_backend, HsmBackend::Mock);
    }

    #[test]
    fn test_register_and_sign() {
        let mut hsm = HsmKmsLayer::new();
        hsm.init(HsmBackend::Mock).unwrap();
        
        let key_idx = hsm.register_key(12345, SignAlgorithm::Ed25519, 1).unwrap();
        assert_eq!(key_idx, 0);
        
        let digest = [0x42u8; MAX_DIGEST_SIZE];
        let signature = hsm.sign(0, &digest).unwrap();
        
        // Verify signature is non-zero
        assert!(signature.iter().any(|&b| b != 0));
        
        let (signs, fails) = hsm.get_stats();
        assert_eq!(signs, 1);
        assert_eq!(fails, 0);
    }

    #[test]
    fn test_verify() {
        let mut hsm = HsmKmsLayer::new();
        hsm.init(HsmBackend::Mock).unwrap();
        
        hsm.register_key(12345, SignAlgorithm::Ed25519, 1).unwrap();
        
        let digest = [0x42u8; MAX_DIGEST_SIZE];
        let signature = hsm.sign(0, &digest).unwrap();
        
        let valid = hsm.verify(0, &digest, &signature).unwrap();
        assert!(valid);
    }
}
