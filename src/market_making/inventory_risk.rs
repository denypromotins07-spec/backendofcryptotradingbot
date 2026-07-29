//! Real-time inventory risk penalty scaling based on multi-asset portfolio variance.

#![allow(clippy::missing_docs_in_private_items)]
#![forbid(clippy::vec_init_then_push, clippy::useless_vec)]

use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use crate::common::fixed_point::FixedI64;

const CACHE_LINE_SIZE: usize = 64;
const MAX_ASSETS: usize = 64;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskState { Normal = 0, Elevated = 1, High = 2, Critical = 3, Halted = 4 }

#[repr(C, align(64))]
pub struct InventoryRiskParams {
    pub base_gamma: FixedI64,
    pub position_limit: FixedI64,
    pub portfolio_limit: FixedI64,
    pub var_confidence: FixedI64,
    pub max_var: FixedI64,
    pub corr_window: u32,
    pub circuit_breaker_threshold: FixedI64,
    _pad: [u8; CACHE_LINE_SIZE - 6 * 8 - 4],
}

impl InventoryRiskParams {
    #[inline]
    pub const fn new() -> Self {
        Self {
            base_gamma: FixedI64::from_i64(100000000),
            position_limit: FixedI64::from_i64(10000000000),
            portfolio_limit: FixedI64::from_i64(100000000000),
            var_confidence: FixedI64::from_i64(990000000),
            max_var: FixedI64::from_i64(5000000000),
            corr_window: 1000,
            circuit_breaker_threshold: FixedI64::from_i64(2000000000),
            _pad: [0u8; CACHE_LINE_SIZE - 6 * 8 - 4],
        }
    }
}

#[repr(C, align(64))]
pub struct InventoryRiskState {
    pub active: AtomicBool,
    pub halted: AtomicBool,
    pub risk_state: AtomicI64,
    pub num_assets: AtomicU64,
    pub asset_ids: [u64; MAX_ASSETS],
    pub inventories: [FixedI64; MAX_ASSETS],
    pub volatilities: [FixedI64; MAX_ASSETS],
    pub portfolio_variance: FixedI64,
    pub portfolio_var: FixedI64,
    pub risk_penalty: FixedI64,
    pub breach_counter: AtomicU64,
    _pad: [u8; CACHE_LINE_SIZE - 2*8 - 8 - 8 - MAX_ASSETS*8*2 - 3*8 - 8],
}

impl InventoryRiskState {
    #[inline]
    pub const fn new(_params: &InventoryRiskParams) -> Self {
        Self {
            active: AtomicBool::new(false),
            halted: AtomicBool::new(false),
            risk_state: AtomicI64::new(RiskState::Normal as i64),
            num_assets: AtomicU64::new(0),
            asset_ids: [0; MAX_ASSETS],
            inventories: [FixedI64::ZERO; MAX_ASSETS],
            volatilities: [FixedI64::ZERO; MAX_ASSETS],
            portfolio_variance: FixedI64::ZERO,
            portfolio_var: FixedI64::ZERO,
            risk_penalty: FixedI64::ONE,
            breach_counter: AtomicU64::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - 2*8 - 8 - 8 - MAX_ASSETS*8*2 - 3*8 - 8],
        }
    }
    #[inline] pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); self.halted.store(false, Ordering::Relaxed); }
    #[inline] pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }
    #[inline] pub fn halt(&self) { self.halted.store(true, Ordering::SeqCst); self.risk_state.store(RiskState::Halted as i64, Ordering::SeqCst); }
    #[inline] pub fn can_trade(&self) -> bool { self.active.load(Ordering::Relaxed) && !self.halted.load(Ordering::SeqCst) }
    #[inline] pub fn get_risk_state(&self) -> RiskState { match self.risk_state.load(Ordering::Relaxed) { 0=>RiskState::Normal,1=>RiskState::Elevated,2=>RiskState::High,3=>RiskState::Critical,_=>RiskState::Halted } }
    #[inline] pub fn get_risk_penalty(&self) -> FixedI64 { self.risk_penalty }
    
    #[inline]
    pub fn register_asset(&self, id: u64, vol: FixedI64) -> Option<usize> {
        let idx = self.num_assets.load(Ordering::Relaxed) as usize;
        if idx >= MAX_ASSETS { return None; }
        unsafe {
            *self.asset_ids.get_unchecked_mut(idx) = id;
            *self.volatilities.get_unchecked_mut(idx) = vol;
        }
        self.num_assets.fetch_add(1, Ordering::Relaxed);
        Some(idx)
    }
    
    #[inline]
    pub fn update_inventory(&self, idx: usize, inv: FixedI64) -> Option<FixedI64> {
        if !self.can_trade() || idx >= self.num_assets.load(Ordering::Relaxed) as usize { return None; }
        unsafe { *self.inventories.get_unchecked_mut(idx) = inv; }
        self.recalc_risk();
        Some(self.calc_penalty(idx))
    }
    
    #[inline]
    fn recalc_risk(&self) {
        let n = self.num_assets.load(Ordering::Relaxed) as usize;
        if n == 0 { return; }
        let mut var = FixedI64::ZERO;
        unsafe {
            for i in 0..n {
                let vi = *self.inventories.get_unchecked(i);
                let voli = *self.volatilities.get_unchecked(i);
                for j in 0..n {
                    let vj = *self.inventories.get_unchecked(j);
                    let volj = *self.volatilities.get_unchecked(j);
                    var = var + vi * vj * voli * volj / FixedI64::from_i64(1000000000);
                }
            }
        }
        self.portfolio_variance = var;
        let std = if var > FixedI64::ZERO { var.sqrt_approx() } else { FixedI64::ZERO };
        self.portfolio_var = FixedI64::from_i64(2330000000) * std / FixedI64::ONE;
        self.update_state();
    }
    
    #[inline]
    fn update_state(&self) {
        let p = InventoryRiskParams::new();
        let var = self.portfolio_var;
        let state = if var > p.max_var * FixedI64::from_i64(2000000000) { RiskState::Critical }
            else if var > p.max_var { RiskState::High }
            else if var > p.max_var / FixedI64::from_i64(2000000000) { RiskState::Elevated }
            else { RiskState::Normal };
        self.risk_state.store(state as i64, Ordering::Relaxed);
        let penalty = match state { RiskState::Normal=>FixedI64::ONE, RiskState::Elevated=>FixedI64::from_i64(1250000000), RiskState::High=>FixedI64::from_i64(1500000000), RiskState::Critical=>FixedI64::from_i64(2000000000), RiskState::Halted=>FixedI64::MAX };
        self.risk_penalty = penalty;
        if penalty >= p.circuit_breaker_threshold {
            if self.breach_counter.fetch_add(1, Ordering::Relaxed) >= 3 { self.halt(); }
        } else { self.breach_counter.store(0, Ordering::Relaxed); }
    }
    
    #[inline]
    fn calc_penalty(&self, idx: usize) -> FixedI64 {
        let params = InventoryRiskParams::new();
        let inv = unsafe { *self.inventories.get_unchecked(idx) }.abs();
        let vol = unsafe { *self.volatilities.get_unchecked(idx) };
        let ratio = if params.position_limit > FixedI64::ZERO { inv / params.position_limit } else { FixedI64::ZERO };
        let vol_adj = if vol > FixedI64::from_i64(1000000000) { vol / FixedI64::from_i64(1000000000) } else { FixedI64::ONE };
        (FixedI64::ONE + ratio) * vol_adj * self.risk_penalty
    }
    
    #[inline]
    pub fn would_breach(&self, idx: usize, add: FixedI64) -> bool {
        let params = InventoryRiskParams::new();
        let cur = unsafe { *self.inventories.get_unchecked(idx) };
        (cur + add).abs() > params.position_limit
    }
}

const _: () = assert!(core::mem::size_of::<InventoryRiskParams>() % 64 == 0);
const _: () = assert!(core::mem::size_of::<InventoryRiskState>() % 64 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_risk_params() { let p = InventoryRiskParams::new(); assert!(p.base_gamma > FixedI64::ZERO); }
    #[test]
    fn test_activation() { let p = InventoryRiskParams::new(); let s = InventoryRiskState::new(&p); s.activate(); assert!(s.can_trade()); }
    #[test]
    fn test_circuit_breaker() { let p = InventoryRiskParams::new(); let s = InventoryRiskState::new(&p); s.activate(); s.halt(); assert!(!s.can_trade()); }
}
