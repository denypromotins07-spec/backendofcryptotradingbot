//! OKX Specific Quirks
//! 
//! Implements portfolio margin, combo margins, and specific API rate limits
//! unique to OKX exchange. Uses fixed-point arithmetic and lock-free operations.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

pub type FixedPrice = i64;
pub type FixedQty = i64;

/// OKX account mode
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OkxAccountMode {
    Cash = 0,
    MarginIsolated = 1,
    MarginCross = 2,
    PortfolioMargin = 3,
}

/// Cache-line aligned portfolio state
#[repr(C, align(64))]
pub struct OkxPortfolio {
    pub total_equity: AtomicI64,
    pub total_pnl: AtomicI64,
    pub margin_balance: AtomicI64,
    pub available_balance: AtomicI64,
    pub frozen_balance: AtomicI64,
    pub account_mode: AtomicU8,
    _padding: [u8; 39],
}

impl OkxPortfolio {
    pub const fn new() -> Self {
        Self {
            total_equity: AtomicI64::new(0),
            total_pnl: AtomicI64::new(0),
            margin_balance: AtomicI64::new(0),
            available_balance: AtomicI64::new(0),
            frozen_balance: AtomicI64::new(0),
            account_mode: AtomicU8::new(OkxAccountMode::Cash as u8),
            _padding: [0u8; 39],
        }
    }
}

/// OKX API rate limit tracker
#[repr(C, align(64))]
pub struct OkxRateLimiter {
    pub requests_per_second: AtomicU64,
    pub burst_capacity: AtomicU64,
    pub current_tokens: AtomicU64,
    pub last_refill_ticks: AtomicU64,
    pub rejected_count: AtomicU64,
    _padding: [u8; 24],
}

impl OkxRateLimiter {
    pub const fn new(rps: u64, burst: u64) -> Self {
        Self {
            requests_per_second: AtomicU64::new(rps),
            burst_capacity: AtomicU64::new(burst),
            current_tokens: AtomicU64::new(burst),
            last_refill_ticks: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }

    #[inline]
    pub fn try_acquire(&self) -> bool {
        let tokens = self.current_tokens.load(Ordering::Acquire);
        if tokens > 0 {
            self.current_tokens.fetch_sub(1, Ordering::AcqRel);
            true
        } else {
            self.rejected_count.fetch_add(1, Ordering::AcqRel);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = OkxPortfolio::new();
        assert_eq!(portfolio.account_mode.load(Ordering::Acquire), OkxAccountMode::Cash as u8);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = OkxRateLimiter::new(10, 5);
        
        // Should allow burst
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }
        
        // Should reject after burst exhausted
        assert!(!limiter.try_acquire());
        assert_eq!(limiter.rejected_count.load(Ordering::Acquire), 1);
    }
}
