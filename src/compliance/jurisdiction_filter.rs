//! Real-time compliance filter blocking restricted tokens or geo-fenced features.
//!
//! This module implements jurisdiction-based filtering for tokens and trading
//! features, ensuring compliance with regulatory requirements across different
//! geographic regions.
//!
//! **Latency Target:** < 100ns per filter check.
//! **Memory Limit:** Pre-allocated token lists, no heap allocation.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of blocked tokens.
const MAX_BLOCKED_TOKENS: usize = 1024;

/// Maximum number of jurisdiction rules.
const MAX_JURISDICTIONS: usize = 64;

/// Token hash (first 8 bytes of SHA-256).
pub type TokenHash = u64;

/// Jurisdiction codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jurisdiction {
    None = 0,
    US = 1,
    EU = 2,
    UK = 3,
    JP = 4,
    CN = 5,
    KR = 6,
    SG = 7,
    Global = 255,
}

/// Blocked token entry.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockedToken {
    /// Token hash.
    pub hash: TokenHash,
    /// Symbol string (truncated).
    pub symbol: [u8; 12],
    /// Blocking reason code.
    pub reason: u16,
    /// Jurisdictions where blocked (bitmask).
    pub jurisdictions: u64,
    /// Active flag.
    pub is_active: AtomicBool,
    /// Padding.
    _padding: [u8; 41],
}

impl BlockedToken {
    #[inline]
    pub const fn new() -> Self {
        Self {
            hash: 0,
            symbol: [0u8; 12],
            reason: 0,
            jurisdictions: 0,
            is_active: AtomicBool::new(false),
            _padding: [0u8; 41],
        }
    }
}

const _: () = assert!(core::mem::size_of::<BlockedToken>() == CACHE_LINE_SIZE);

/// Jurisdiction rule entry.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JurisdictionRule {
    /// Jurisdiction code.
    pub jurisdiction: Jurisdiction,
    /// Allowed features bitmask.
    pub allowed_features: u64,
    /// Blocked token count.
    blocked_count: AtomicU64,
    /// Active flag.
    pub is_active: AtomicBool,
    /// Padding.
    _padding: [u8; 47],
}

impl JurisdictionRule {
    #[inline]
    pub const fn new() -> Self {
        Self {
            jurisdiction: Jurisdiction::None,
            allowed_features: 0,
            blocked_count: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            _padding: [0u8; 47],
        }
    }
}

const _: () = assert!(core::mem::size_of::<JurisdictionRule>() == CACHE_LINE_SIZE);

/// Feature flags.
pub mod features {
    pub const SPOT_TRADING: u64 = 1 << 0;
    pub const FUTURES_TRADING: u64 = 1 << 1;
    pub const OPTIONS_TRADING: u64 = 1 << 2;
    pub const MARGIN_TRADING: u64 = 1 << 3;
    pub const STAKING: u64 = 1 << 4;
    pub const LENDING: u64 = 1 << 5;
    pub const WITHDRAWAL: u64 = 1 << 6;
    pub const DEPOSIT: u64 = 1 << 7;
}

/// The main jurisdiction filter.
pub struct JurisdictionFilter {
    /// Blocked tokens list.
    blocked_tokens: [BlockedToken; MAX_BLOCKED_TOKENS],
    /// Jurisdiction rules.
    rules: [JurisdictionRule; MAX_JURISDICTIONS],
    /// Count of blocked tokens.
    token_count: AtomicU64,
    /// Current active jurisdiction.
    current_jurisdiction: AtomicU64,
    /// Total checks performed.
    check_count: AtomicU64,
    /// Total blocks.
    block_count: AtomicU64,
    /// Flag indicating if the filter is active.
    is_active: AtomicBool,
    /// Shadow mode (log but don't block).
    shadow_mode: AtomicBool,
    /// Padding.
    _padding: [u8; 48],
}

unsafe impl Send for JurisdictionFilter {}
unsafe impl Sync for JurisdictionFilter {}

impl JurisdictionFilter {
    /// Create a new jurisdiction filter.
    #[inline]
    pub const fn new() -> Self {
        Self {
            blocked_tokens: [BlockedToken::new(); MAX_BLOCKED_TOKENS],
            rules: [JurisdictionRule::new(); MAX_JURISDICTIONS],
            token_count: AtomicU64::new(0),
            current_jurisdiction: AtomicU64::new(Jurisdiction::Global as u64),
            check_count: AtomicU64::new(0),
            block_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            shadow_mode: AtomicBool::new(false),
            _padding: [0u8; 48],
        }
    }

    /// Add a blocked token.
    #[inline]
    pub fn add_blocked_token(&self, symbol: &[u8], hash: TokenHash, jurisdictions: u64, reason: u16) -> Result<usize, &'static str> {
        let idx = self.token_count.load(Ordering::Acquire) as usize;
        if idx >= MAX_BLOCKED_TOKENS {
            return Err("Blocked token list full");
        }

        let claimed = self.token_count.compare_exchange(
            idx as u64,
            (idx + 1) as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match claimed {
            Ok(_) => {
                let token = &self.blocked_tokens[idx];
                unsafe {
                    ptr::write_volatile(&token.hash as *const TokenHash as *mut TokenHash, hash);
                    ptr::copy_nonoverlapping(symbol.as_ptr(), token.symbol.as_mut_ptr(), symbol.len().min(12));
                    ptr::write_volatile(&token.reason as *const u16 as *mut u16, reason);
                    ptr::write_volatile(&token.jurisdictions as *const u64 as *mut u64, jurisdictions);
                }
                token.is_active.store(true, Ordering::Release);
                Ok(idx)
            }
            Err(_) => Err("Failed to claim slot"),
        }
    }

    /// Check if a token is allowed in the current jurisdiction.
    #[inline]
    pub fn check_token(&self, token_hash: TokenHash) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return true;
        }

        self.check_count.fetch_add(1, Ordering::Relaxed);

        let current_jur = self.current_jurisdiction.load(Ordering::Acquire);
        let mut is_blocked = false;

        // Check blocked tokens (branchless)
        let count = self.token_count.load(Ordering::Acquire) as usize;
        for i in 0..count.min(MAX_BLOCKED_TOKENS) {
            let token = &self.blocked_tokens[i];
            if !token.is_active.load(Ordering::Acquire) {
                continue;
            }

            let stored_hash = unsafe { ptr::read_volatile(&token.hash as *const TokenHash) };
            let hash_match = ((token_hash ^ stored_hash) - 1) >> 63; // 1 if equal, 0 if not
            
            let jur_mask = token.jurisdictions;
            let jur_match = ((jur_mask >> current_jur) & 1) as u64;
            
            let blocked = (hash_match & jur_match) != 0;
            is_blocked = is_blocked || blocked;
        }

        if is_blocked && !self.shadow_mode.load(Ordering::Acquire) {
            self.block_count.fetch_add(1, Ordering::Relaxed);
        }

        !is_blocked || self.shadow_mode.load(Ordering::Acquire)
    }

    /// Check if a feature is allowed.
    #[inline]
    pub fn check_feature(&self, feature: u64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return true;
        }

        let current_jur = self.current_jurisdiction.load(Ordering::Acquire) as usize;
        if current_jur >= MAX_JURISDICTIONS {
            return true;
        }

        let rule = &self.rules[current_jur];
        if !rule.is_active.load(Ordering::Acquire) {
            return true;
        }

        let allowed = unsafe { ptr::read_volatile(&rule.allowed_features as *const u64) };
        (allowed & feature) != 0
    }

    /// Set the current jurisdiction.
    #[inline]
    pub fn set_jurisdiction(&self, jur: Jurisdiction) {
        self.current_jurisdiction.store(jur as u64, Ordering::Release);
    }

    /// Enable/disable shadow mode.
    #[inline]
    pub fn set_shadow_mode(&self, enabled: bool) {
        self.shadow_mode.store(enabled, Ordering::Release);
    }

    /// Get statistics.
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.check_count.load(Ordering::Acquire),
            self.block_count.load(Ordering::Acquire),
            self.token_count.load(Ordering::Acquire),
        )
    }

    /// Hash a symbol string (simplified FNV-1a).
    #[inline]
    pub fn hash_symbol(symbol: &str) -> TokenHash {
        let mut hash: TokenHash = 0xcbf29ce484222325;
        for byte in symbol.as_bytes() {
            hash ^= *byte as TokenHash;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Shutdown the filter.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
        for i in 0..MAX_BLOCKED_TOKENS {
            self.blocked_tokens[i].is_active.store(false, Ordering::Release);
        }
    }
}

impl Default for JurisdictionFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_token_size() {
        assert_eq!(core::mem::size_of::<BlockedToken>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_filter_init() {
        let filter = JurisdictionFilter::new();
        assert!(filter.is_active.load(Ordering::Acquire));
        assert!(!filter.shadow_mode.load(Ordering::Acquire));
    }

    #[test]
    fn test_hash_symbol() {
        let hash1 = JurisdictionFilter::hash_symbol("BTCUSDT");
        let hash2 = JurisdictionFilter::hash_symbol("ETHUSDT");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_block_token() {
        let filter = JurisdictionFilter::new();
        
        let btc_hash = JurisdictionFilter::hash_symbol("BTCUSDT");
        filter.add_blocked_token(b"BTCUSDT", btc_hash, 1 << Jurisdiction::US as u64, 1).unwrap();
        
        filter.set_jurisdiction(Jurisdiction::US);
        assert!(!filter.check_token(btc_hash));
        
        filter.set_jurisdiction(Jurisdiction::EU);
        assert!(filter.check_token(btc_hash)); // Not blocked in EU
    }

    #[test]
    fn test_shadow_mode() {
        let filter = JurisdictionFilter::new();
        
        let btc_hash = JurisdictionFilter::hash_symbol("BTCUSDT");
        filter.add_blocked_token(b"BTCUSDT", btc_hash, 1 << Jurisdiction::US as u64, 1).unwrap();
        filter.set_jurisdiction(Jurisdiction::US);
        
        filter.set_shadow_mode(true);
        assert!(filter.check_token(btc_hash)); // Allowed in shadow mode
        
        let (_, blocks, _) = filter.get_stats();
        assert_eq!(blocks, 0); // No blocks counted in shadow mode
    }
}
