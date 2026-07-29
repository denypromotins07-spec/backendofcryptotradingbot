//! Chapter 1: Margin, Leverage, and Open Interest Dynamics
//! Cross-margin and isolated margin calculator with liquidation price estimator.

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Fixed-point scaling factor (1e9)
const FP_SCALE: i64 = 1_000_000_000;

#[repr(C, align(64))]
pub struct MarginPosition {
    /// Entry price in fixed-point
    entry_price: AtomicI64,
    /// Position size (signed: positive=long, negative=short)
    position_size: AtomicI64,
    /// Initial margin in fixed-point
    initial_margin: AtomicI64,
    /// Maintenance margin ratio (scaled by 1e9)
    maintenance_margin_ratio: AtomicI64,
    /// Leverage multiplier (scaled by 1e9)
    leverage: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 6 * 8],
}

#[repr(C, align(64))]
pub struct LiquidationCalculator {
    /// Current mark price in fixed-point
    mark_price: AtomicI64,
    /// Last liquidation threshold
    liq_threshold: AtomicI64,
    /// Margin balance in fixed-point
    margin_balance: AtomicI64,
    /// Unrealized PnL in fixed-point
    unrealized_pnl: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub enum MarginType {
    Isolated,
    Cross,
}

impl Default for MarginPosition {
    fn default() -> Self {
        Self {
            entry_price: AtomicI64::new(0),
            position_size: AtomicI64::new(0),
            initial_margin: AtomicI64::new(0),
            maintenance_margin_ratio: AtomicI64::new(5_000_000_000), // 0.5%
            leverage: AtomicI64::new(1_000_000_000), // 1x
            _padding: [0u8; CACHE_LINE_SIZE - 6 * 8],
        }
    }
}

impl Default for LiquidationCalculator {
    fn default() -> Self {
        Self {
            mark_price: AtomicI64::new(0),
            liq_threshold: AtomicI64::new(0),
            margin_balance: AtomicI64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl MarginPosition {
    /// Create new margin position with specified leverage
    #[inline]
    pub fn new(entry_price: i64, size: i64, leverage: i64) -> Self {
        let initial_margin = (entry_price * size).abs() / leverage;
        Self {
            entry_price: AtomicI64::new(entry_price),
            position_size: AtomicI64::new(size),
            initial_margin: AtomicI64::new(initial_margin),
            maintenance_margin_ratio: AtomicI64::new(5_000_000_000), // 0.5% default
            leverage: AtomicI64::new(leverage),
            _padding: [0u8; CACHE_LINE_SIZE - 6 * 8],
        }
    }

    /// Calculate liquidation price for long position (fixed-point arithmetic)
    #[inline]
    pub fn calc_liquidation_price_long(&self) -> i64 {
        let entry = self.entry_price.load(Ordering::Relaxed);
        let size = self.position_size.load(Ordering::Relaxed);
        let init_margin = self.initial_margin.load(Ordering::Relaxed);
        let mmr = self.maintenance_margin_ratio.load(Ordering::Relaxed);

        if size <= 0 {
            return 0;
        }

        // Liq Price = (Entry Price * Size - Initial Margin) / (Size - (MMR * Entry Price * Size / FP_SCALE))
        // Simplified: Liq Price = Entry * (1 - IM/Notional) / (1 - MMR)
        let notional = entry * size / FP_SCALE;
        if notional == 0 {
            return 0;
        }

        let im_ratio = init_margin * FP_SCALE / notional;
        let denominator = FP_SCALE - mmr;
        
        if denominator <= 0 {
            return i64::MAX;
        }

        entry * (FP_SCALE - im_ratio) / denominator
    }

    /// Calculate liquidation price for short position (fixed-point arithmetic)
    #[inline]
    pub fn calc_liquidation_price_short(&self) -> i64 {
        let entry = self.entry_price.load(Ordering::Relaxed);
        let size = self.position_size.load(Ordering::Relaxed);
        let init_margin = self.initial_margin.load(Ordering::Relaxed);
        let mmr = self.maintenance_margin_ratio.load(Ordering::Relaxed);

        if size >= 0 {
            return 0;
        }

        let abs_size = size.abs();
        let notional = entry * abs_size / FP_SCALE;
        if notional == 0 {
            return 0;
        }

        let im_ratio = init_margin * FP_SCALE / notional;
        let denominator = FP_SCALE + mmr;

        entry * (FP_SCALE + im_ratio) / denominator
    }

    /// Update mark price and calculate unrealized PnL
    #[inline]
    pub fn update_mark_price(&self, mark_price: i64, calc: &LiquidationCalculator) {
        calc.mark_price.store(mark_price, Ordering::Relaxed);
        
        let entry = self.entry_price.load(Ordering::Relaxed);
        let size = self.position_size.load(Ordering::Relaxed);
        
        // Unrealized PnL = (Mark Price - Entry Price) * Size for long
        // Unrealized PnL = (Entry Price - Mark Price) * Size for short
        let pnl = if size > 0 {
            (mark_price - entry) * size / FP_SCALE
        } else {
            (entry - mark_price) * size.abs() / FP_SCALE
        };
        
        calc.unrealized_pnl.store(pnl, Ordering::Relaxed);
    }

    /// Check if position should be liquidated
    #[inline]
    pub fn is_liquidated(&self, calc: &LiquidationCalculator) -> bool {
        let margin_balance = calc.margin_balance.load(Ordering::Relaxed);
        let unrealized_pnl = calc.unrealized_pnl.load(Ordering::Relaxed);
        let init_margin = self.initial_margin.load(Ordering::Relaxed);
        let mmr = self.maintenance_margin_ratio.load(Ordering::Relaxed);

        // Available margin = balance + unrealized PnL
        let available_margin = margin_balance + unrealized_pnl;
        
        // Maintenance margin required = Notional * MMR
        let entry = self.entry_price.load(Ordering::Relaxed);
        let size = self.position_size.load(Ordering::Relaxed).abs();
        let notional = entry * size / FP_SCALE;
        let maintenance_margin = notional * mmr / FP_SCALE;

        available_margin < maintenance_margin && available_margin < init_margin
    }

    /// Set leverage (branchless validation)
    #[inline]
    pub fn set_leverage(&self, new_leverage: i64) {
        // Validate leverage is between 1x and 125x (scaled)
        let valid = (new_leverage >= 1_000_000_000) & (new_leverage <= 125_000_000_000);
        let clamped = if valid != 0 { new_leverage } else { self.leverage.load(Ordering::Relaxed) };
        self.leverage.store(clamped, Ordering::Relaxed);
    }
}

impl LiquidationCalculator {
    /// Update margin balance
    #[inline]
    pub fn update_balance(&self, balance: i64) {
        self.margin_balance.store(balance, Ordering::Relaxed);
    }

    /// Get current equity (balance + unrealized PnL)
    #[inline]
    pub fn get_equity(&self) -> i64 {
        let balance = self.margin_balance.load(Ordering::Relaxed);
        let pnl = self.unrealized_pnl.load(Ordering::Relaxed);
        balance + pnl
    }

    /// Calculate margin ratio (used margin / equity)
    #[inline]
    pub fn margin_ratio(&self, used_margin: i64) -> i64 {
        let equity = self.get_equity();
        if equity <= 0 {
            return i64::MAX;
        }
        used_margin * FP_SCALE / equity
    }

    /// Branchless check for margin call threshold
    #[inline]
    pub fn check_margin_call(&self, threshold: i64) -> i64 {
        let ratio = self.margin_ratio(self.margin_balance.load(Ordering::Relaxed));
        // Returns 1 if margin call needed, 0 otherwise (branchless)
        ((ratio > threshold) as i64)
    }
}

/// Cross-margin calculator for portfolio-level margin optimization
#[repr(C, align(64))]
pub struct CrossMarginEngine {
    /// Total portfolio value in fixed-point
    portfolio_value: AtomicI64,
    /// Total used margin across positions
    total_used_margin: AtomicI64,
    /// Net PnL across all positions
    net_pnl: AtomicI64,
    /// Number of active positions
    position_count: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 4 * 8],
}

impl Default for CrossMarginEngine {
    fn default() -> Self {
        Self {
            portfolio_value: AtomicI64::new(0),
            total_used_margin: AtomicI64::new(0),
            net_pnl: AtomicI64::new(0),
            position_count: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 4 * 8],
        }
    }
}

impl CrossMarginEngine {
    /// Add position to cross-margin pool
    #[inline]
    pub fn add_position(&self, margin_used: i64, pnl: i64) {
        self.total_used_margin.fetch_add(margin_used, Ordering::Relaxed);
        self.net_pnl.fetch_add(pnl, Ordering::Relaxed);
        self.position_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove position from cross-margin pool
    #[inline]
    pub fn remove_position(&self, margin_used: i64, pnl: i64) {
        self.total_used_margin.fetch_sub(margin_used, Ordering::Relaxed);
        self.net_pnl.fetch_sub(pnl, Ordering::Relaxed);
        self.position_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Calculate portfolio margin ratio
    #[inline]
    pub fn portfolio_margin_ratio(&self) -> i64 {
        let value = self.portfolio_value.load(Ordering::Relaxed);
        let used = self.total_used_margin.load(Ordering::Relaxed);
        if value <= 0 {
            return i64::MAX;
        }
        used * FP_SCALE / value
    }

    /// Check if portfolio is undercollateralized
    #[inline]
    pub fn is_undercollateralized(&self, threshold: i64) -> bool {
        let ratio = self.portfolio_margin_ratio();
        ratio > threshold
    }

    /// Update portfolio value
    #[inline]
    pub fn update_portfolio_value(&self, value: i64) {
        self.portfolio_value.store(value, Ordering::Relaxed);
    }

    /// Get net PnL
    #[inline]
    pub fn get_net_pnl(&self) -> i64 {
        self.net_pnl.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_liquidation_price() {
        let pos = MarginPosition::new(50_000 * FP_SCALE, 1_000_000_000, 10_000_000_000); // 10x leverage
        let liq_price = pos.calc_liquidation_price_long();
        
        // With 10x leverage, liquidation should occur around 10% below entry
        assert!(liq_price > 45_000 * FP_SCALE);
        assert!(liq_price < 50_000 * FP_SCALE);
    }

    #[test]
    fn test_short_liquidation_price() {
        let pos = MarginPosition::new(50_000 * FP_SCALE, -1_000_000_000, 10_000_000_000); // 10x short
        let liq_price = pos.calc_liquidation_price_short();
        
        // With 10x leverage, liquidation should occur around 10% above entry
        assert!(liq_price > 50_000 * FP_SCALE);
        assert!(liq_price < 55_000 * FP_SCALE);
    }

    #[test]
    fn test_cross_margin_engine() {
        let engine = CrossMarginEngine::default();
        engine.update_portfolio_value(1_000_000 * FP_SCALE);
        engine.add_position(100_000 * FP_SCALE, 5_000 * FP_SCALE);
        engine.add_position(50_000 * FP_SCALE, -2_000 * FP_SCALE);
        
        assert_eq!(engine.get_net_pnl(), 3_000 * FP_SCALE);
        assert_eq!(engine.position_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_extreme_leverage_wicks() {
        // Test 100x leverage position
        let pos = MarginPosition::new(50_000 * FP_SCALE, 1_000_000_000, 100_000_000_000);
        let liq_price = pos.calc_liquidation_price_long();
        
        // With 100x leverage, liquidation should occur very close to entry (~1%)
        assert!(liq_price > 49_000 * FP_SCALE);
        assert!(liq_price < 50_000 * FP_SCALE);
    }

    #[test]
    fn test_margin_call_detection() {
        let calc = LiquidationCalculator::default();
        calc.update_balance(10_000 * FP_SCALE);
        
        // Threshold at 80% margin ratio
        let needs_call = calc.check_margin_call(800_000_000);
        assert_eq!(needs_call, 0); // Should not trigger initially
    }
}
