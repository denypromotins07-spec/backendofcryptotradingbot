//! Lee-Ready and Tick Rule trade sign classifier using zero-copy trade arrays and SIMD.
//!
//! Implements high-frequency trade sign classification for micro-price calculation
//! using fixed-point arithmetic and AVX2 vectorization.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

const CACHE_LINE_SIZE: usize = 64;
const MAX_TRADES: usize = 65536;

#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSign {
    Unknown = 0,
    Buy = 1,
    Sell = -1,
}

#[repr(C, align(64))]
pub struct TradeSignState {
    pub active: AtomicBool,
    pub trade_count: AtomicU64,
    pub prev_price: FixedI64,
    pub prev_mid: FixedI64,
    pub trades: [FixedI64; MAX_TRADES],
    pub signs: [i8; MAX_TRADES],
    pub head: usize,
    pub buy_volume: FixedI64,
    pub sell_volume: FixedI64,
    _pad: [u8; CACHE_LINE_SIZE - 2 * 8 - 2 * 8 - MAX_TRADES * 8 - MAX_TRADES - 8 - 2 * 8 - 8],
}

impl TradeSignState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            trade_count: AtomicU64::new(0),
            prev_price: FixedI64::ZERO,
            prev_mid: FixedI64::ZERO,
            trades: [FixedI64::ZERO; MAX_TRADES],
            signs: [0i8; MAX_TRADES],
            head: 0,
            buy_volume: FixedI64::ZERO,
            sell_volume: FixedI64::ZERO,
            _pad: [0u8; CACHE_LINE_SIZE - 2 * 8 - 2 * 8 - MAX_TRADES * 8 - MAX_TRADES - 8 - 2 * 8 - 8],
        }
    }

    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }

    #[inline]
    pub fn classify_tick_rule(&self, price: FixedI64, volume: FixedI64) -> TradeSign {
        if !self.is_active() { return TradeSign::Unknown; }
        
        let sign = if price > self.prev_price {
            TradeSign::Buy
        } else if price < self.prev_price {
            TradeSign::Sell
        } else {
            TradeSign::Unknown
        };

        self.prev_price = price;
        self._update_volume(sign, volume);
        sign
    }

    #[inline]
    pub fn classify_lee_ready(&self, price: FixedI64, mid: FixedI64, volume: FixedI64) -> TradeSign {
        if !self.is_active() { return TradeSign::Unknown; }
        
        let sign = if price > mid {
            TradeSign::Buy
        } else if price < mid {
            TradeSign::Sell
        } else {
            self.classify_tick_rule(price, volume)
        };

        self.prev_mid = mid;
        self._update_volume(sign, volume);
        sign
    }

    #[inline]
    fn _update_volume(&self, sign: TradeSign, volume: FixedI64) {
        match sign {
            TradeSign::Buy => self.buy_volume = self.buy_volume + volume,
            TradeSign::Sell => self.sell_volume = self.sell_volume + volume,
            _ => {}
        }
    }

    #[inline]
    pub fn classify_batch_simd(&self, prices: &[FixedI64], mids: &[FixedI64], output: &mut [i8]) {
        assert!(prices.len() == mids.len());
        unsafe {
            for i in (0..prices.len()).step_by(4) {
                let p = _mm256_loadu_si256(prices[i..].as_ptr() as *const __m256i);
                let m = _mm256_loadu_si256(mids[i..].as_ptr() as *const __m256i);
                let gt = _mm256_cmpgt_epi64(p, m);
                let lt = _mm256_cmpgt_epi64(m, p);
                let mut signs = [0i8; 4];
                let gt_arr: [i64; 4] = core::mem::transmute(gt);
                let lt_arr: [i64; 4] = core::mem::transmute(lt);
                for j in 0..4 {
                    signs[j] = if gt_arr[j] != 0 { 1 } else if lt_arr[j] != 0 { -1 } else { 0 };
                    if i + j < output.len() { output[i + j] = signs[j]; }
                }
            }
        }
    }
}

const _: () = assert!(core::mem::size_of::<TradeSignState>() % 64 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tick_rule() {
        let state = TradeSignState::new();
        state.activate();
        let p1 = FixedI64::from_i64(100_000_000_000i64);
        let p2 = FixedI64::from_i64(100_000_000_100i64);
        let v = FixedI64::from_i64(1000000000i64);
        let _ = state.classify_tick_rule(p1, v);
        let sign = state.classify_tick_rule(p2, v);
        assert_eq!(sign, TradeSign::Buy);
    }
}
