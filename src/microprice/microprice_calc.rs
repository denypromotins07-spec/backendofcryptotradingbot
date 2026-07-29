//! High-frequency micro-price and fair value calculator using consolidated L2 depth.
//!
//! Implements micro-price calculation with fixed-point arithmetic and cache-aligned structures.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

const CACHE_LINE_SIZE: usize = 64;
const MAX_LEVELS: usize = 20;

#[repr(C, align(64))]
pub struct MicroPriceState {
    pub active: AtomicBool,
    pub update_count: AtomicU64,
    pub bid_prices: [FixedI64; MAX_LEVELS],
    pub ask_prices: [FixedI64; MAX_LEVELS],
    pub bid_sizes: [FixedI64; MAX_LEVELS],
    pub ask_sizes: [FixedI64; MAX_LEVELS],
    pub mid_price: FixedI64,
    pub micro_price: FixedI64,
    pub fair_value: FixedI64,
    pub spread: FixedI64,
    pub spread_bps: FixedI64,
    _pad: [u8; CACHE_LINE_SIZE - 2 * 8 - MAX_LEVELS * 8 * 4 - 5 * 8 - 8],
}

impl MicroPriceState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            update_count: AtomicU64::new(0),
            bid_prices: [FixedI64::ZERO; MAX_LEVELS],
            ask_prices: [FixedI64::ZERO; MAX_LEVELS],
            bid_sizes: [FixedI64::ZERO; MAX_LEVELS],
            ask_sizes: [FixedI64::ZERO; MAX_LEVELS],
            mid_price: FixedI64::ZERO,
            micro_price: FixedI64::ZERO,
            fair_value: FixedI64::ZERO,
            spread: FixedI64::ZERO,
            spread_bps: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE - 2 * 8 - MAX_LEVELS * 8 * 4 - 5 * 8 - 8],
        }
    }

    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }

    #[inline]
    pub fn update(&self) -> Option<(FixedI64, FixedI64, FixedI64)> {
        if !self.is_active() { return None; }
        
        let best_bid = self.bid_prices[0];
        let best_ask = self.ask_prices[0];
        
        if best_bid <= FixedI64::ZERO || best_ask <= FixedI64::ZERO { return None; }
        
        // Mid price
        self.mid_price = (best_bid + best_ask) / FixedI64::from_i64(2000000000i64);
        
        // Spread
        self.spread = best_ask - best_bid;
        self.spread_bps = if self.mid_price > FixedI64::ZERO {
            self.spread * FixedI64::from_i64(100000000i64) / self.mid_price
        } else { FixedI64::ZERO };
        
        // Micro-price: volume-weighted mid
        let bid_vol = self.bid_sizes[0];
        let ask_vol = self.ask_sizes[0];
        let total_vol = bid_vol + ask_vol;
        
        self.micro_price = if total_vol > FixedI64::ZERO {
            (best_bid * ask_vol + best_ask * bid_vol) / total_vol
        } else { self.mid_price };
        
        // Fair value: multi-level weighted average
        let mut fw_bid = FixedI64::ZERO;
        let mut fw_ask = FixedI64::ZERO;
        let mut w_sum = FixedI64::ZERO;
        
        for i in 0..MAX_LEVELS.min(5) {
            let w = FixedI64::from_i64((5 - i) as i64);
            fw_bid = fw_bid + self.bid_prices[i] * self.bid_sizes[i] * w;
            fw_ask = fw_ask + self.ask_prices[i] * self.ask_sizes[i] * w;
            w_sum = w_sum + self.bid_sizes[i] * w + self.ask_sizes[i] * w;
        }
        
        self.fair_value = if w_sum > FixedI64::ZERO {
            (fw_bid + fw_ask) / (w_sum * FixedI64::from_i64(2000000000i64))
        } else { self.mid_price };
        
        let _ = self.update_count.fetch_add(1, Ordering::Relaxed);
        Some((self.micro_price, self.fair_value, self.mid_price))
    }
}

const _: () = assert!(core::mem::size_of::<MicroPriceState>() % 64 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_microprice() {
        let state = MicroPriceState::new();
        state.activate();
        state.bid_prices[0] = FixedI64::from_i64(99_900_000_000i64);
        state.ask_prices[0] = FixedI64::from_i64(100_100_000_000i64);
        state.bid_sizes[0] = FixedI64::from_i64(1000000000i64);
        state.ask_sizes[0] = FixedI64::from_i64(1000000000i64);
        let result = state.update();
        assert!(result.is_some());
    }
}
