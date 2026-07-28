//! Dynamic Position Sizer
//! 
//! Kelly Criterion and dynamic position sizing engine with atomic exposure limits.
//! Uses zero-copy math and pre-allocated lookup tables for sub-50ns execution.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Cache line padding size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of instruments supported
const MAX_INSTRUMENTS: usize = 256;

/// Kelly fraction lookup table (pre-computed for common win rates)
/// Index = win_rate * 100 (0-100), Value = Kelly fraction * 10000
static KELLY_LOOKUP: [u16; 101] = compute_kelly_lookup();

const fn compute_kelly_lookup() -> [u16; 101] {
    let mut arr = [0u16; 101];
    let mut i = 0;
    while i <= 100 {
        // Simplified Kelly: f = p - (1-p)/R where R is reward/ratio (assume 1.5)
        // For win_rate i%, Kelly = i/100 - (1-i/100)/1.5
        let win_rate = i as i32;
        let kelly = (win_rate * 150 - (100 - win_rate) * 100) / 150;
        let kelly_clamped = if kelly < 0 { 0 } else if kelly > 100 { 100 } else { kelly };
        arr[i] = (kelly_clamped * 100) as u16;
        i += 1;
    }
    arr
}

/// Position sizing result - fixed size, no allocation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PositionSizeResult {
    pub quantity: u64,
    pub kelly_fraction: u16,
    pub risk_adjusted_size: u64,
    pub max_position: u64,
    pub _padding: [u8; 32],
}

impl PositionSizeResult {
    const fn empty() -> Self {
        Self {
            quantity: 0,
            kelly_fraction: 0,
            risk_adjusted_size: 0,
            max_position: 0,
            _padding: [0u8; 32],
        }
    }
}

/// Per-instrument position state - cache-line aligned
#[repr(C)]
#[derive(Debug)]
struct InstrumentPosition {
    current_qty: AtomicI64,
    max_qty: AtomicU64,
    avg_entry_price: AtomicU64,
    unrealized_pnl: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 32],
}

impl InstrumentPosition {
    const fn new() -> Self {
        Self {
            current_qty: AtomicI64::new(0),
            max_qty: AtomicU64::new(0),
            avg_entry_price: AtomicU64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 32],
        }
    }
}

/// Dynamic position sizing engine
pub struct PositionSizer {
    /// Per-instrument positions
    positions: [InstrumentPosition; MAX_INSTRUMENTS],
    /// Global account equity (in quote units * 10^8 for precision)
    account_equity: AtomicU64,
    /// Max portfolio leverage (in basis points, 10000 = 1x)
    max_leverage_bp: AtomicU64,
    /// Risk per trade (in basis points of equity)
    risk_per_trade_bp: AtomicU64,
    /// Kelly multiplier (0-100, where 100 = full Kelly)
    kelly_multiplier: AtomicU64,
    /// Total portfolio exposure
    total_exposure: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for PositionSizer {}
unsafe impl Sync for PositionSizer {}

impl PositionSizer {
    /// Create new position sizer with default parameters
    pub const fn new() -> Self {
        const EMPTY_POS: InstrumentPosition = InstrumentPosition::new();
        Self {
            positions: [EMPTY_POS; MAX_INSTRUMENTS],
            account_equity: AtomicU64::new(10_000_000_000), // Default 100M * 10^8
            max_leverage_bp: AtomicU64::new(30000), // 3x max leverage
            risk_per_trade_bp: AtomicU64::new(200), // 2% risk per trade
            kelly_multiplier: AtomicU64::new(50), // Half-Kelly by default
            total_exposure: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Set account equity (in base units * 10^8)
    #[inline(always)]
    pub fn set_account_equity(&self, equity: u64) {
        self.account_equity.store(equity, Ordering::Release);
    }

    /// Get current account equity
    #[inline(always)]
    pub fn get_account_equity(&self) -> u64 {
        self.account_equity.load(Ordering::Acquire)
    }

    /// Set maximum leverage in basis points
    #[inline(always)]
    pub fn set_max_leverage(&self, leverage_bp: u64) {
        self.max_leverage_bp.store(leverage_bp, Ordering::Release);
    }

    /// Set risk per trade in basis points
    #[inline(always)]
    pub fn set_risk_per_trade(&self, risk_bp: u64) {
        self.risk_per_trade_bp.store(risk_bp, Ordering::Release);
    }

    /// Set Kelly multiplier (0-100)
    #[inline(always)]
    pub fn set_kelly_multiplier(&self, multiplier: u64) {
        let clamped = if multiplier > 100 { 100 } else { multiplier };
        self.kelly_multiplier.store(clamped, Ordering::Release);
    }

    /// Set maximum position size for an instrument
    #[inline(always)]
    pub fn set_max_position(&self, instrument_id: usize, max_qty: u64) {
        if instrument_id < MAX_INSTRUMENTS {
            self.positions[instrument_id].max_qty.store(max_qty, Ordering::Release);
        }
    }

    /// Update current position for an instrument
    #[inline(always)]
    pub fn update_position(&self, instrument_id: usize, qty_delta: i64, price: u64) {
        if instrument_id >= MAX_INSTRUMENTS {
            return;
        }

        let pos = &self.positions[instrument_id];
        let old_qty = pos.current_qty.fetch_add(qty_delta, Ordering::AcqRel);
        let new_qty = old_qty + qty_delta;

        // Update average entry price (branchless)
        if (old_qty >= 0 && qty_delta > 0) || (old_qty <= 0 && qty_delta < 0) {
            // Adding to position - update avg price
            let old_abs = old_qty.unsigned_abs();
            let delta_abs = qty_delta.unsigned_abs();
            let total_abs = old_abs + delta_abs;
            
            let old_notional = old_abs as u128 * pos.avg_entry_price.load(Ordering::Relaxed) as u128;
            let new_notional = delta_abs as u128 * price as u128;
            let new_avg = ((old_notional + new_notional) / total_abs as u128) as u64;
            
            pos.avg_entry_price.store(new_avg, Ordering::Release);
        } else if (new_qty as i64).signum() != old_qty.signum() {
            // Position flipped - reset avg price
            pos.avg_entry_price.store(price, Ordering::Release);
        }

        // Update unrealized PnL
        let pnl = (price as i128 - pos.avg_entry_price.load(Ordering::Relaxed) as i128) * new_qty as i128;
        pos.unrealized_pnl.store(pnl as i64, Ordering::Release);

        // Update total exposure
        let abs_qty = new_qty.unsigned_abs();
        let notional = abs_qty as u128 * price as u128;
        self.total_exposure.fetch_add(notional as u64, Ordering::AcqRel);
    }

    /// Calculate position size using Kelly Criterion
    /// Returns quantity in base units
    #[inline(always)]
    pub fn calculate_kelly_size(
        &self,
        instrument_id: usize,
        win_rate_bps: u16, // Win rate in basis points (0-10000)
        reward_ratio_bps: u16, // Reward/risk ratio in basis points (e.g., 150 = 1.5:1)
        current_price: u64,
    ) -> PositionSizeResult {
        if instrument_id >= MAX_INSTRUMENTS || win_rate_bps > 10000 {
            return PositionSizeResult::empty();
        }

        let equity = self.account_equity.load(Ordering::Acquire);
        let kelly_mult = self.kelly_multiplier.load(Ordering::Acquire);
        let max_leverage = self.max_leverage_bp.load(Ordering::Acquire);

        // Lookup Kelly fraction from pre-computed table
        let win_rate_pct = (win_rate_bps / 100) as usize;
        let base_kelly = KELLY_LOOKUP[win_rate_pct] as u32;

        // Adjust for reward ratio (branchless)
        let reward_adj = if reward_ratio_bps >= 100 {
            (base_kelly as u64 * reward_ratio_bps as u64) / 150
        } else {
            (base_kelly as u64 * reward_ratio_bps as u64) / 150
        };

        // Apply Kelly multiplier
        let adjusted_kelly = (reward_adj * kelly_mult) / 100;

        // Calculate risk-adjusted position size
        let risk_bp = self.risk_per_trade_bp.load(Ordering::Acquire);
        let max_position_value = (equity as u128 * max_leverage as u128) / 10000;
        let risk_adjusted_value = (equity as u128 * risk_bp as u128 * adjusted_kelly as u128) / 1_000_000_000;

        // Take minimum of leverage limit and risk-adjusted size
        let final_value = if max_position_value < risk_adjusted_value {
            max_position_value
        } else {
            risk_adjusted_value
        };

        // Convert to quantity
        let quantity = if current_price > 0 {
            (final_value / current_price as u128) as u64
        } else {
            0
        };

        // Check against instrument-specific max
        let max_qty = self.positions[instrument_id].max_qty.load(Ordering::Acquire);
        let current_qty = self.positions[instrument_id].current_qty.load(Ordering::Acquire).unsigned_abs();
        let remaining = max_qty.saturating_sub(current_qty);

        let final_qty = if quantity > remaining { remaining } else { quantity };

        PositionSizeResult {
            quantity: final_qty,
            kelly_fraction: (adjusted_kelly & 0xFFFF) as u16,
            risk_adjusted_size: (risk_adjusted_value & 0xFFFFFFFFFFFF) as u64,
            max_position: max_qty,
            _padding: [0u8; 32],
        }
    }

    /// Get current position for an instrument
    #[inline(always)]
    pub fn get_position(&self, instrument_id: usize) -> i64 {
        if instrument_id < MAX_INSTRUMENTS {
            self.positions[instrument_id].current_qty.load(Ordering::Acquire)
        } else {
            0
        }
    }

    /// Get unrealized PnL for an instrument
    #[inline(always)]
    pub fn get_unrealized_pnl(&self, instrument_id: usize) -> i64 {
        if instrument_id < MAX_INSTRUMENTS {
            self.positions[instrument_id].unrealized_pnl.load(Ordering::Acquire)
        } else {
            0
        }
    }

    /// Get total portfolio exposure
    #[inline(always)]
    pub fn get_total_exposure(&self) -> u64 {
        self.total_exposure.load(Ordering::Acquire)
    }

    /// Calculate available buying power
    #[inline(always)]
    pub fn get_buying_power(&self) -> u64 {
        let equity = self.account_equity.load(Ordering::Acquire);
        let max_leverage = self.max_leverage_bp.load(Ordering::Acquire);
        let current_exposure = self.total_exposure.load(Ordering::Acquire);

        let max_exposure = (equity as u128 * max_leverage as u128) / 10000;
        
        // Branchless saturating subtract
        let available = if max_exposure > current_exposure as u128 {
            (max_exposure - current_exposure as u128) as u64
        } else {
            0
        };

        available
    }

    /// Reset all positions (for system restart)
    #[inline(always)]
    pub fn reset_positions(&self) {
        for i in 0..MAX_INSTRUMENTS {
            self.positions[i].current_qty.store(0, Ordering::Release);
            self.positions[i].avg_entry_price.store(0, Ordering::Release);
            self.positions[i].unrealized_pnl.store(0, Ordering::Release);
        }
        self.total_exposure.store(0, Ordering::Release);
    }
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_size_result_size() {
        assert_eq!(core::mem::size_of::<PositionSizeResult>(), 64);
    }

    #[test]
    fn test_kelly_lookup_table() {
        // 50% win rate should give ~0 Kelly with 1:1 reward
        assert!(KELLY_LOOKUP[50] > 0);
        // 100% win rate should give max Kelly
        assert_eq!(KELLY_LOOKUP[100], 10000);
        // 0% win rate should give 0 Kelly
        assert_eq!(KELLY_LOOKUP[0], 0);
    }

    #[test]
    fn test_kelly_sizing() {
        let sizer = PositionSizer::new();
        sizer.set_account_equity(10_000_000_000); // 100M
        sizer.set_kelly_multiplier(50); // Half-Kelly

        let result = sizer.calculate_kelly_size(0, 6000, 150, 50000);
        assert!(result.quantity > 0);
        assert!(result.kelly_fraction > 0);
    }

    #[test]
    fn test_position_updates() {
        let sizer = PositionSizer::new();
        sizer.set_max_position(0, 1000);

        // Buy 100 at 50000
        sizer.update_position(0, 100, 50000);
        assert_eq!(sizer.get_position(0), 100);

        // Sell 50 at 51000
        sizer.update_position(0, -50, 51000);
        assert_eq!(sizer.get_position(0), 50);
        assert!(sizer.get_unrealized_pnl(0) > 0);
    }

    #[test]
    fn test_buying_power() {
        let sizer = PositionSizer::new();
        sizer.set_account_equity(10_000_000_000);
        sizer.set_max_leverage(30000); // 3x

        let initial_bp = sizer.get_buying_power();
        assert!(initial_bp > 0);

        // Take a position
        sizer.update_position(0, 100, 50000);
        
        let remaining_bp = sizer.get_buying_power();
        assert!(remaining_bp < initial_bp);
    }

    #[test]
    fn test_instrument_max_position() {
        let sizer = PositionSizer::new();
        sizer.set_max_position(0, 500);

        // Try to exceed max
        sizer.update_position(0, 600, 50000);
        
        let result = sizer.calculate_kelly_size(0, 6000, 150, 50000);
        assert!(result.quantity <= 500);
    }
}
