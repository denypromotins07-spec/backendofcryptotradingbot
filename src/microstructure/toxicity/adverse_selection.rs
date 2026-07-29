//! Real-time adverse selection cost estimator using markouts.
//! 
//! Measures the cost of adverse selection by tracking price markouts
//! after our trades execute. Negative markouts indicate we traded
//! against informed flow.
//! 
//! Uses fixed-point arithmetic and lock-free circular buffers.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Number of markout samples to track
const MAX_MARKOUTS: usize = 256;

/// Markout horizons in microseconds
const MARKOUT_HORIZONS: [u64; 4] = [100, 500, 1000, 5000]; // 100us, 500us, 1ms, 5ms

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Trade record for markout calculation
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TradeRecord {
    pub price: i64,                  // Execution price (fixed-point)
    pub size: i64,                   // Trade size
    pub is_buy: bool,                // Was this a buy?
    pub timestamp_cycles: u64,       // rdtsc timestamp
    _padding: [u8; 47],              // Pad to 64 bytes
}

impl Default for TradeRecord {
    fn default() -> Self {
        Self {
            price: 0,
            size: 0,
            is_buy: false,
            timestamp_cycles: 0,
            _padding: [0; 47],
        }
    }
}

/// Lock-free adverse selection calculator
#[repr(C)]
pub struct AdverseSelection {
    /// Recent trades for markout calculation
    trades: [TradeRecord; MAX_MARKOUTS],
    trade_head: AtomicU64,
    trade_tail: AtomicU64,
    
    /// Markout results per horizon (in basis points)
    markout_sum: [AtomicI64; 4],     // Sum of markouts per horizon
    markout_count: [AtomicU64; 4],   // Count per horizon
    
    /// Adverse selection metrics
    total_adverse_cost: AtomicI64,   // Total adverse selection cost (bps)
    trade_count: AtomicU64,          // Total trades analyzed
    
    /// Risk metrics
    max_adverse_cost: AtomicI64,     // Worst adverse selection observed
    consecutive_adverse: AtomicU64,  // Consecutive adverse trades
    
    /// Kill switches
    toxicity_flag: AtomicBool,       // High adverse selection detected
    trading_halted: AtomicBool,      // Halt trading flag
    
    /// Thresholds
    adverse_threshold_bps: AtomicI64, // Threshold for toxicity (bps)
    
    _padding: [u8; 16],              // Pad to cache line
}

impl AdverseSelection {
    /// Create new adverse selection calculator
    pub const fn new() -> Self {
        Self {
            trades: [TradeRecord::default(); MAX_MARKOUTS],
            trade_head: AtomicU64::new(0),
            trade_tail: AtomicU64::new(0),
            markout_sum: [
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
            ],
            markout_count: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            total_adverse_cost: AtomicI64::new(0),
            trade_count: AtomicU64::new(0),
            max_adverse_cost: AtomicI64::new(0),
            consecutive_adverse: AtomicU64::new(0),
            toxicity_flag: AtomicBool::new(false),
            trading_halted: AtomicBool::new(false),
            adverse_threshold_bps: AtomicI64::new(500), // 5 bps threshold
            _padding: [0; 16],
        }
    }
    
    /// Record a trade execution
    #[inline(always)]
    pub fn record_trade(&self, price: i64, size: i64, is_buy: bool) {
        if self.trading_halted.load(Ordering::Acquire) {
            return;
        }
        
        let cycles = unsafe { x86_64::_rdtsc() };
        
        // Get next slot
        let head = self.trade_head.load(Ordering::Relaxed);
        let next_head = (head + 1) % MAX_MARKOUTS as u64;
        
        // If buffer full, advance tail
        if next_head == self.trade_tail.load(Ordering::Relaxed) {
            self.trade_tail.store((self.trade_tail.load(Ordering::Relaxed) + 1) % MAX_MARKOUTS as u64, Ordering::Relaxed);
        }
        
        // Store trade
        unsafe {
            let trade = self.trades.get_unchecked_mut(head as usize);
            trade.price = price;
            trade.size = size;
            trade.is_buy = is_buy;
            trade.timestamp_cycles = cycles;
        }
        
        self.trade_head.store(next_head, Ordering::Release);
        
        // Calculate markouts for completed trades
        self.calculate_markouts(cycles);
    }
    
    /// Calculate markouts for trades at each horizon
    #[inline(always)]
    fn calculate_markouts(&self, current_cycles: u64) {
        let tail = self.trade_tail.load(Ordering::Acquire);
        let head = self.trade_head.load(Ordering::Acquire);
        
        let mut idx = tail;
        while idx != head {
            let trade = unsafe { self.trades.get_unchecked(idx as usize) };
            let elapsed_us = ((current_cycles.wrapping_sub(trade.timestamp_cycles)) as f64 / 3000.0) as u64;
            
            // Check each horizon
            for (i, &horizon) in MARKOUT_HORIZONS.iter().enumerate() {
                // Only calculate if we've passed the horizon
                if elapsed_us >= horizon && elapsed_us < horizon * 2 {
                    // Get current mid price (would come from market data feed)
                    // For now, this is a placeholder - in production, look up current price
                    let current_mid = trade.price; // Placeholder
                    
                    // Calculate markout: (mid_after - exec_price) for buys, opposite for sells
                    let markout_bps = if trade.is_buy {
                        ((current_mid - trade.price) * 10_000) / trade.price.abs().max(1)
                    } else {
                        ((trade.price - current_mid) * 10_000) / trade.price.abs().max(1)
                    };
                    
                    // Negative markout = adverse selection
                    let is_adverse = markout_bps < 0;
                    
                    // Update statistics
                    self.markout_sum[i].fetch_add(markout_bps, Ordering::Relaxed);
                    self.markout_count[i].fetch_add(1, Ordering::Relaxed);
                    
                    if is_adverse {
                        self.total_adverse_cost.fetch_add(-markout_bps, Ordering::Relaxed);
                        self.consecutive_adverse.fetch_add(1, Ordering::Relaxed);
                        
                        // Track maximum
                        let max = self.max_adverse_cost.load(Ordering::Relaxed);
                        if -markout_bps > max {
                            self.max_adverse_cost.store(-markout_bps, Ordering::Relaxed);
                        }
                    } else {
                        self.consecutive_adverse.store(0, Ordering::Relaxed);
                    }
                    
                    // Check toxicity threshold
                    let avg_adverse = if self.trade_count.load(Ordering::Relaxed) > 0 {
                        self.total_adverse_cost.load(Ordering::Relaxed) / self.trade_count.load(Ordering::Relaxed) as i64
                    } else {
                        0
                    };
                    
                    if avg_adverse > self.adverse_threshold_bps.load(Ordering::Acquire) {
                        self.toxicity_flag.store(true, Ordering::Release);
                        
                        // Halt if consecutive adverse trades exceed limit
                        if self.consecutive_adverse.load(Ordering::Relaxed) > 10 {
                            self.trading_halted.store(true, Ordering::Release);
                        }
                    }
                }
            }
            
            idx = (idx + 1) % MAX_MARKOUTS as u64;
        }
        
        self.trade_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get average markout at a specific horizon
    #[inline(always)]
    pub fn avg_markout_bps(&self, horizon_idx: usize) -> i64 {
        if horizon_idx >= 4 {
            return 0;
        }
        
        let sum = self.markout_sum[horizon_idx].load(Ordering::Acquire);
        let count = self.markout_count[horizon_idx].load(Ordering::Acquire);
        
        if count == 0 {
            return 0;
        }
        
        sum / count as i64
    }
    
    /// Check if flow is toxic
    #[inline(always)]
    pub fn is_toxic(&self) -> bool {
        self.toxicity_flag.load(Ordering::Acquire)
    }
    
    /// Check if trading should halt
    #[inline(always)]
    pub fn should_halt(&self) -> bool {
        self.trading_halted.load(Ordering::Acquire)
    }
    
    /// Reset halt flag
    #[inline(always)]
    pub fn reset_halt(&self) {
        self.trading_halted.store(false, Ordering::Release);
    }
    
    /// Get total adverse selection cost
    #[inline(always)]
    pub fn total_adverse_cost(&self) -> i64 {
        self.total_adverse_cost.load(Ordering::Acquire)
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<TradeRecord>() == 64);
        assert!(core::mem::size_of::<AdverseSelection>() % 64 == 0);
    }
    
    #[test]
    fn test_adverse_selection_basic() {
        let adv = AdverseSelection::new();
        
        // Record some trades
        adv.record_trade(100_000_000, 1000, true);
        adv.record_trade(100_100_000, 1000, false);
        
        // Should have recorded trades
        assert!(adv.trade_count.load(Ordering::Acquire) > 0);
    }
}
