//! Order book imbalance (OBI) alpha generator using weighted bid-ask volumes at multiple levels.
//!
//! Implements OBI calculation with lock-free circular buffers and branchless thresholds.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

const CACHE_LINE_SIZE: usize = 64;
const MAX_LEVELS: usize = 10;
const MAX_HISTORY: usize = 8192;

#[repr(C, align(64))]
pub struct ImbalanceAlphaState {
    pub active: AtomicBool,
    pub obs_count: AtomicU64,
    pub bid_volumes: [FixedI64; MAX_LEVELS],
    pub ask_volumes: [FixedI64; MAX_LEVELS],
    pub weights: [FixedI64; MAX_LEVELS],
    pub obi_history: [FixedI64; MAX_HISTORY],
    pub head: usize,
    pub current_obi: FixedI64,
    pub weighted_obi: FixedI64,
    pub alpha_signal: FixedI64,
    _pad: [u8; CACHE_LINE_SIZE - 2 * 8 - MAX_LEVELS * 8 * 3 - MAX_HISTORY * 8 - 8 - 3 * 8 - 8],
}

impl ImbalanceAlphaState {
    #[inline]
    pub const fn new() -> Self {
        let mut weights = [FixedI64::ZERO; MAX_LEVELS];
        let mut i = 0;
        while i < MAX_LEVELS {
            weights[i] = FixedI64::from_i64((10 - i) as i64 * 100000000i64);
            i += 1;
        }
        Self {
            active: AtomicBool::new(false),
            obs_count: AtomicU64::new(0),
            bid_volumes: [FixedI64::ZERO; MAX_LEVELS],
            ask_volumes: [FixedI64::ZERO; MAX_LEVELS],
            weights,
            obi_history: [FixedI64::ZERO; MAX_HISTORY],
            head: 0,
            current_obi: FixedI64::ZERO,
            weighted_obi: FixedI64::ZERO,
            alpha_signal: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE - 2 * 8 - MAX_LEVELS * 8 * 3 - MAX_HISTORY * 8 - 8 - 3 * 8 - 8],
        }
    }

    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }

    #[inline]
    pub fn update(&self, bids: &[FixedI64; MAX_LEVELS], asks: &[FixedI64; MAX_LEVELS]) -> FixedI64 {
        if !self.is_active() { return FixedI64::ZERO; }
        
        let mut total_bid = FixedI64::ZERO;
        let mut total_ask = FixedI64::ZERO;
        let mut weighted_bid = FixedI64::ZERO;
        let mut weighted_ask = FixedI64::ZERO;
        
        for i in 0..MAX_LEVELS {
            total_bid = total_bid + bids[i];
            total_ask = total_ask + asks[i];
            weighted_bid = weighted_bid + bids[i] * self.weights[i];
            weighted_ask = weighted_ask + asks[i] * self.weights[i];
        }
        
        let total = total_bid + total_ask;
        self.current_obi = if total > FixedI64::ZERO {
            (total_bid - total_ask) / total
        } else { FixedI64::ZERO };
        
        let w_total = weighted_bid + weighted_ask;
        self.weighted_obi = if w_total > FixedI64::ZERO {
            (weighted_bid - weighted_ask) / w_total
        } else { FixedI64::ZERO };
        
        // Store in circular buffer
        let head = self.head;
        unsafe { *self.obi_history.get_unchecked_mut(head) = self.weighted_obi; }
        self.head = if head + 1 >= MAX_HISTORY { 0 } else { head + 1 };
        let _ = self.obs_count.fetch_add(1, Ordering::Relaxed);
        
        // Generate alpha signal (branchless threshold)
        let thresh = FixedI64::from_i64(200000000i64);
        let abs_obi = if self.weighted_obi < FixedI64::ZERO { -self.weighted_obi } else { self.weighted_obi };
        self.alpha_signal = if abs_obi > thresh { self.weighted_obi } else { FixedI64::ZERO };
        
        self.alpha_signal
    }
}

const _: () = assert!(core::mem::size_of::<ImbalanceAlphaState>() % 64 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_obi_calculation() {
        let state = ImbalanceAlphaState::new();
        state.activate();
        let mut bids = [FixedI64::ZERO; MAX_LEVELS];
        let mut asks = [FixedI64::ZERO; MAX_LEVELS];
        bids[0] = FixedI64::from_i64(1000000000i64);
        asks[0] = FixedI64::from_i64(500000000i64);
        let obi = state.update(&bids, &asks);
        assert!(obi > FixedI64::ZERO);
    }
}
