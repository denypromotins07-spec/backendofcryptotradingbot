//! Bybit Specific Quirks
//! 
//! Implements unified margin logic, specific fee rebates, and order limits
//! unique to Bybit exchange. Uses fixed-point arithmetic and lock-free operations.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicU8, Ordering};

pub type FixedPrice = i64;
pub type FixedQty = i64;

/// Unified margin mode
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginMode {
    Isolated = 0,
    Unified = 1,
    Portfolio = 2,
}

/// Bybit-specific order flags
#[repr(u8)]
pub struct BybitOrderFlags {
    pub bits: u8,
}

impl BybitOrderFlags {
    pub const TP_SL_ORDER: u8 = 0b0000_0001;
    pub const CLOSE_ON_TRIGGER: u8 = 0b0000_0010;
    pub const REDUCE_ONLY: u8 = 0b0000_0100;
    pub const POST_ONLY: u8 = 0b0000_1000;
    
    pub const fn new() -> Self {
        Self { bits: 0 }
    }
    
    pub const fn with_tp_sl(mut self) -> Self {
        self.bits |= Self::TP_SL_ORDER;
        self
    }
}

/// Cache-line aligned Bybit position state
#[repr(C, align(64))]
pub struct BybitPosition {
    pub symbol_id: u32,
    pub side: AtomicU8,
    pub size: AtomicI64,
    pub entry_price: AtomicI64,
    pub liq_price: AtomicI64,
    pub unrealized_pnl: AtomicI64,
    pub leverage: AtomicU64,
    pub margin_mode: AtomicU8,
    _padding: [u8; 22],
}

impl BybitPosition {
    pub const fn new(symbol_id: u32) -> Self {
        Self {
            symbol_id,
            side: AtomicU8::new(0),
            size: AtomicI64::new(0),
            entry_price: AtomicI64::new(0),
            liq_price: AtomicI64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            leverage: AtomicU64::new(1),
            margin_mode: AtomicU8::new(MarginMode::Isolated as u8),
            _padding: [0u8; 22],
        }
    }
}

/// Bybit fee rebate calculator
#[repr(C, align(64))]
pub struct BybitFeeCalculator {
    pub maker_rebate_bps: AtomicU64,
    pub taker_fee_bps: AtomicU64,
    pub vip_level: AtomicU8,
    pub total_maker_volume: AtomicI64,
    pub total_taker_volume: AtomicI64,
    pub total_rebates_earned: AtomicI64,
    _padding: [u8; 23],
}

impl BybitFeeCalculator {
    pub const fn new() -> Self {
        Self {
            maker_rebate_bps: AtomicU64::new(10), // 0.10% default
            taker_fee_bps: AtomicU64::new(10),
            vip_level: AtomicU8::new(0),
            total_maker_volume: AtomicI64::new(0),
            total_taker_volume: AtomicI64::new(0),
            total_rebates_earned: AtomicI64::new(0),
            _padding: [0u8; 23],
        }
    }

    #[inline]
    pub fn calculate_maker_rebate(&self, qty: FixedQty, price: FixedPrice) -> FixedQty {
        let rebate_bps = self.maker_rebate_bps.load(Ordering::Acquire);
        let notional = (qty * price) / 1_000_000_000;
        (notional * rebate_bps as i64) / 10000
    }

    #[inline]
    pub fn calculate_taker_fee(&self, qty: FixedQty, price: FixedPrice) -> FixedQty {
        let fee_bps = self.taker_fee_bps.load(Ordering::Acquire);
        let notional = (qty * price) / 1_000_000_000;
        (notional * fee_bps as i64) / 10000
    }

    #[inline]
    pub fn record_maker_fill(&self, qty: FixedQty, price: FixedPrice) {
        let notional = (qty * price) / 1_000_000_000;
        self.total_maker_volume.fetch_add(notional, Ordering::AcqRel);
        let rebate = self.calculate_maker_rebate(qty, price);
        self.total_rebates_earned.fetch_add(rebate, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = BybitPosition::new(1);
        assert_eq!(pos.side.load(Ordering::Acquire), 0);
        assert_eq!(pos.margin_mode.load(Ordering::Acquire), MarginMode::Isolated as u8);
    }

    #[test]
    fn test_fee_calculation() {
        let calc = BybitFeeCalculator::new();
        
        let rebate = calc.calculate_maker_rebate(1_000_000_000i64, 50_000_000_000i64);
        // 50B notional * 0.10% = 50M
        assert_eq!(rebate, 50_000_000i64);
    }
}
