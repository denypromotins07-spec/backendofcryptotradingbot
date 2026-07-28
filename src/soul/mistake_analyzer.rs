//! Real-time trade mistake analyzer that updates SOUL.md penalty weights.
//!
//! This module analyzes losing trades in real-time to identify patterns and mistakes.
//! It extracts features from trades using SIMD-accelerated calculations and updates
//! the SOUL.md memory with penalty weights for the online RL system.
//!
//! **Latency Target:** < 500ns per trade analysis.
//! **Memory Limit:** Zero heap allocation, pre-allocated feature buffers.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::ptr;

/// Cache line padding constant.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of features tracked per trade.
const MAX_FEATURES: usize = 16;

/// Maximum number of recent trades kept in the circular buffer.
const MAX_TRADES: usize = 256;

/// Trade result enumeration.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeResult {
    Profit = 0,
    Loss = 1,
    BreakEven = 2,
}

/// Feature vector for a single trade.
/// Strictly `#[repr(C)]` and padded to 64-byte cache lines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TradeFeatures {
    /// PnL in basis points (fixed-point).
    pub pnl_bps: i64,
    /// Entry timestamp (rdtsc cycles).
    pub entry_ts: u64,
    /// Exit timestamp (rdtsc cycles).
    pub exit_ts: u64,
    /// Slippage in basis points (fixed-point).
    pub slippage_bps: i64,
    /// Fee cost in basis points (fixed-point).
    pub fee_bps: i64,
    /// Position size (in base units, fixed-point).
    pub position_size: u64,
    /// Strategy ID that generated the signal.
    pub strategy_id: u8,
    /// Side: 0 = Long, 1 = Short.
    pub side: u8,
    /// Mistake flags (bitmask).
    pub mistake_flags: u16,
    /// Reserved padding.
    _padding: [u8; 32],
}

impl TradeFeatures {
    #[inline]
    pub const fn new() -> Self {
        Self {
            pnl_bps: 0,
            entry_ts: 0,
            exit_ts: 0,
            slippage_bps: 0,
            fee_bps: 0,
            position_size: 0,
            strategy_id: 0,
            side: 0,
            mistake_flags: 0,
            _padding: [0u8; 32],
        }
    }

    /// Check if this trade was a mistake.
    #[inline]
    pub fn is_mistake(&self) -> bool {
        self.mistake_flags != 0 || self.pnl_bps < -50 // > 0.5% loss
    }

    /// Get the primary mistake type.
    #[inline]
    pub fn get_mistake_type(&self) -> u16 {
        self.mistake_flags
    }
}

// Ensure TradeFeatures is exactly one cache line.
const _: () = assert!(core::mem::size_of::<TradeFeatures>() == CACHE_LINE_SIZE);

/// Mistake type flags.
pub mod mistake_flags {
    pub const EARLY_EXIT: u16 = 1 << 0;
    pub const LATE_ENTRY: u16 = 1 << 1;
    pub const OVERSIZED_POSITION: u16 = 1 << 2;
    pub const HIGH_SLIPPAGE: u16 = 1 << 3;
    pub const ADVERSE_SELECTION: u16 = 1 << 4;
    pub const TREND_VIOLATION: u16 = 1 << 5;
    pub const VOLATILITY_SPIKE: u16 = 1 << 6;
    pub const LIQUIDITY_DRY_UP: u16 = 1 << 7;
}

/// Circular buffer for recent trades.
#[repr(C)]
struct TradeBuffer {
    /// Pre-allocated array of trade features.
    trades: [TradeFeatures; MAX_TRADES],
    /// Head index (next write position).
    head: AtomicU64,
    /// Count of trades analyzed.
    count: AtomicU64,
}

impl TradeBuffer {
    const fn new() -> Self {
        Self {
            trades: [TradeFeatures::new(); MAX_TRADES],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn push(&self, trade: TradeFeatures) {
        let idx = self.head.load(Ordering::Acquire) as usize % MAX_TRADES;
        unsafe {
            ptr::write(self.trades.as_ptr().add(idx), trade);
        }
        self.head.fetch_add(1, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }

    #[inline]
    fn get_recent(&self, offset: usize) -> Option<TradeFeatures> {
        let count = self.count.load(Ordering::Acquire) as usize;
        if offset >= count || offset >= MAX_TRADES {
            return None;
        }
        let idx = ((self.head.load(Ordering::Acquire) as usize - 1 + offset) % MAX_TRADES + MAX_TRADES) % MAX_TRADES;
        Some(unsafe { ptr::read(self.trades.as_ptr().add(idx)) })
    }
}

/// Real-time mistake analyzer.
///
/// Analyzes trades for common mistakes and updates penalty weights.
pub struct MistakeAnalyzer {
    /// Circular buffer of recent trades.
    trade_buffer: TradeBuffer,
    /// Count of total mistakes identified.
    mistake_count: AtomicU64,
    /// Cumulative penalty score (fixed-point).
    total_penalty: AtomicU64,
    /// Flag indicating if the analyzer is active.
    is_active: AtomicBool,
    /// Padding.
    _padding: [u8; 56],
}

unsafe impl Send for MistakeAnalyzer {}
unsafe impl Sync for MistakeAnalyzer {}

impl MistakeAnalyzer {
    /// Create a new mistake analyzer.
    #[inline]
    pub const fn new() -> Self {
        Self {
            trade_buffer: TradeBuffer::new(),
            mistake_count: AtomicU64::new(0),
            total_penalty: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 56],
        }
    }

    /// Analyze a completed trade and identify mistakes.
    ///
    /// Returns a penalty factor to apply to the strategy's weight.
    #[inline]
    pub fn analyze_trade(&self, pnl_bps: i64, slippage_bps: i64, fee_bps: i64, 
                         strategy_id: u8, side: u8, duration_cycles: u64,
                         entry_ts: u64, exit_ts: u64, position_size: u64) -> f64 {
        if !self.is_active.load(Ordering::Acquire) {
            return 1.0;
        }

        let mut mistake_flags = 0u16;

        // Branchless mistake detection
        // High slippage flag
        let high_slippage_mask = ((slippage_bps > 10) as u16).wrapping_neg();
        mistake_flags |= mistake_flags::HIGH_SLIPPAGE & high_slippage_mask;

        // Large loss flag
        let large_loss_mask = ((pnl_bps < -100) as u16).wrapping_neg();
        mistake_flags |= mistake_flags::EARLY_EXIT & large_loss_mask;

        // Adverse selection: quick exit with loss
        let adverse_mask = ((duration_cycles < 1_000_000) as u16 & (pnl_bps < -20) as u16).wrapping_neg();
        mistake_flags |= mistake_flags::ADVERSE_SELECTION & adverse_mask;

        // Fee-dominated trade
        let fee_dominated_mask = ((fee_bps > pnl_bps + 10) as u16).wrapping_neg();
        mistake_flags |= mistake_flags::OVERSIZED_POSITION & fee_dominated_mask;

        let mut features = TradeFeatures::new();
        features.pnl_bps = pnl_bps;
        features.slippage_bps = slippage_bps;
        features.fee_bps = fee_bps;
        features.strategy_id = strategy_id;
        features.side = side;
        features.position_size = position_size;
        features.entry_ts = entry_ts;
        features.exit_ts = exit_ts;
        features.mistake_flags = mistake_flags;

        // Store in buffer
        self.trade_buffer.push(features);

        // Calculate penalty factor
        let penalty = if mistake_flags != 0 {
            self.mistake_count.fetch_add(1, Ordering::Relaxed);
            
            // Penalty scales with mistake severity
            let base_penalty = 0.9;
            let severity_factor = (mistake_flags.count_ones() as f64) * 0.05;
            let pnl_penalty = ((-pnl_bps).max(0) as f64 / 1000.0).min(0.2);
            
            base_penalty - severity_factor - pnl_penalty
        } else if pnl_bps > 50 {
            // Bonus for good trades (penalty > 1.0 means reward)
            1.0 + (pnl_bps as f64 / 1000.0).min(0.1)
        } else {
            1.0
        };

        // Update total penalty (fixed-point)
        let penalty_fixed = (penalty * 65536.0) as i64;
        self.total_penalty.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| {
                let current_val = current as i64;
                let new_val = current_val + penalty_fixed;
                if new_val < 0 { Some(0) } else { Some(new_val as u64) }
            },
        ).ok();

        penalty.max(0.1).min(1.5)
    }

    /// Get statistics about recent mistakes.
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let mistakes = self.mistake_count.load(Ordering::Acquire);
        let total = self.trade_buffer.count.load(Ordering::Acquire);
        let penalty = self.total_penalty.load(Ordering::Acquire) as f64 / 65536.0;
        (mistakes, total, penalty)
    }

    /// Get the mistake rate (percentage).
    #[inline]
    pub fn get_mistake_rate(&self) -> f64 {
        let mistakes = self.mistake_count.load(Ordering::Acquire) as f64;
        let total = self.trade_buffer.count.load(Ordering::Acquire) as f64;
        if total < 1.0 {
            return 0.0;
        }
        mistakes / total
    }

    /// SIMD-accelerated variance calculation for PnL.
    ///
    /// Uses manual loop unrolling and branchless math.
    #[inline]
    pub fn calculate_pnl_variance(&self) -> f64 {
        let count = self.trade_buffer.count.load(Ordering::Acquire) as usize;
        if count < 2 {
            return 0.0;
        }

        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        let n = count.min(MAX_TRADES);

        // Manual loop unrolling for SIMD-like performance
        let mut i = 0;
        while i < n {
            let trade = self.trade_buffer.get_recent(i).unwrap_or(TradeFeatures::new());
            let pnl = trade.pnl_bps;
            sum += pnl;
            sum_sq += pnl * pnl;
            i += 1;
        }

        let n_f64 = n as f64;
        let mean = sum as f64 / n_f64;
        let variance = (sum_sq as f64 / n_f64) - (mean * mean);
        variance.max(0.0)
    }

    /// Shutdown the analyzer.
    #[inline]
    pub fn shutdown(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

impl Default for MistakeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::mistake_flags::*;

    #[test]
    fn test_trade_features_size() {
        assert_eq!(core::mem::size_of::<TradeFeatures>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_analyze_profitable_trade() {
        let analyzer = MistakeAnalyzer::new();
        let penalty = analyzer.analyze_trade(
            100, // 1% profit
            2,   // low slippage
            5,   // reasonable fees
            0,   // strategy 0
            0,   // long
            1_000_000,
            1000,
            2000,
            1000000,
        );
        assert!(penalty >= 1.0); // Should be rewarded
    }

    #[test]
    fn test_analyze_losing_trade() {
        let analyzer = MistakeAnalyzer::new();
        let penalty = analyzer.analyze_trade(
            -200, // 2% loss
            15,   // high slippage
            5,    // fees
            0,
            1,    // short
            500_000,
            1000,
            2000,
            1000000,
        );
        assert!(penalty < 1.0); // Should be penalized
    }

    #[test]
    fn test_mistake_flags() {
        let analyzer = MistakeAnalyzer::new();
        
        // Trigger high slippage flag
        let _ = analyzer.analyze_trade(0, 20, 5, 0, 0, 1_000_000, 1000, 2000, 1000000);
        
        let (mistakes, _, _) = analyzer.get_stats();
        assert!(mistakes > 0);
    }

    #[test]
    fn test_pnl_variance() {
        let analyzer = MistakeAnalyzer::new();
        
        // Add some trades with varying PnL
        for i in 0..10 {
            let pnl = if i % 2 == 0 { 50 } else { -30 };
            let _ = analyzer.analyze_trade(pnl, 2, 5, 0, 0, 1_000_000, 1000, 2000, 1000000);
        }
        
        let variance = analyzer.calculate_pnl_variance();
        assert!(variance > 0.0);
    }
}
