//! Transaction Cost Analysis (TCA) Engine
//! 
//! Measures fees, spread, slippage, and market impact.
//! Uses lock-free circular buffer for slippage metrics without impacting critical path.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Circular buffer size for slippage tracking
const SLIPPAGE_BUFFER_SIZE: usize = 4096;

/// TCA measurement result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TcaResult {
    /// Total fees paid (quote units * 10^8)
    pub total_fees: u64,
    /// Average spread cost (basis points)
    pub avg_spread_bps: u16,
    /// Slippage from arrival price (basis points)
    pub slippage_bps: i16,
    /// Market impact estimate (basis points)
    pub market_impact_bps: u16,
    /// Total transaction cost (basis points)
    pub total_cost_bps: u16,
    /// Number of trades analyzed
    pub trade_count: u32,
    _padding: [u8; 16],
}

impl TcaResult {
    const fn empty() -> Self {
        Self {
            total_fees: 0,
            avg_spread_bps: 0,
            slippage_bps: 0,
            market_impact_bps: 0,
            total_cost_bps: 0,
            trade_count: 0,
            _padding: [0u8; 16],
        }
    }
}

/// Slippage sample for circular buffer
#[repr(C)]
struct SlippageSample {
    timestamp: AtomicU64,
    slippage_bps: AtomicI64,
    quantity: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 24],
}

impl SlippageSample {
    const fn new() -> Self {
        Self {
            timestamp: AtomicU64::new(0),
            slippage_bps: AtomicI64::new(0),
            quantity: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 24],
        }
    }
}

/// TCA Engine with lock-free circular buffer
pub struct TcaEngine {
    /// Slippage samples circular buffer
    slippage_buffer: [SlippageSample; SLIPPAGE_BUFFER_SIZE],
    /// Write index for circular buffer
    write_idx: AtomicU64,
    /// Total fees accumulated
    total_fees: AtomicU64,
    /// Total notional traded
    total_notional: AtomicU64,
    /// Total spread cost
    total_spread_cost: AtomicU64,
    /// Trade count
    trade_count: AtomicU64,
    /// Arrival prices for slippage calculation
    arrival_prices: [AtomicU64; 256],
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for TcaEngine {}
unsafe impl Sync for TcaEngine {}

impl TcaEngine {
    /// Create new TCA engine
    pub const fn new() -> Self {
        const EMPTY_SAMPLE: SlippageSample = SlippageSample::new();
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        
        Self {
            slippage_buffer: [EMPTY_SAMPLE; SLIPPAGE_BUFFER_SIZE],
            write_idx: AtomicU64::new(0),
            total_fees: AtomicU64::new(0),
            total_notional: AtomicU64::new(0),
            total_spread_cost: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            arrival_prices: [ZERO_U64; 256],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Record a trade for TCA analysis
    #[inline(always)]
    pub fn record_trade(
        &self,
        trade_id: u64,
        quantity: u64,
        fill_price: u64,
        fee: u64,
        spread_bps: u16,
        arrival_price: u64,
    ) {
        let now = unsafe { _rdtsc() };
        
        // Update totals
        self.total_fees.fetch_add(fee, Ordering::Relaxed);
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        
        let notional = quantity * fill_price;
        self.total_notional.fetch_add(notional, Ordering::Relaxed);
        
        let spread_cost = (notional * spread_bps as u64) / 10000;
        self.total_spread_cost.fetch_add(spread_cost, Ordering::Relaxed);

        // Calculate slippage
        let slippage_bps = if arrival_price > 0 {
            ((fill_price as i64 - arrival_price as i64) * 10000) / arrival_price as i64
        } else {
            0
        };

        // Store in circular buffer (lock-free)
        let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) % SLIPPAGE_BUFFER_SIZE as u64;
        let sample = &self.slippage_buffer[idx as usize];
        sample.timestamp.store(now, Ordering::Release);
        sample.slippage_bps.store(slippage_bps, Ordering::Release);
        sample.quantity.store(quantity, Ordering::Release);

        // Store arrival price for later reference
        let price_idx = trade_id % 256;
        self.arrival_prices[price_idx as usize].store(arrival_price, Ordering::Release);
    }

    /// Get average slippage from circular buffer
    #[inline(always)]
    pub fn get_avg_slippage_bps(&self) -> i64 {
        let count = self.write_idx.load(Ordering::Relaxed).min(SLIPPAGE_BUFFER_SIZE as u64);
        if count == 0 {
            return 0;
        }

        let mut sum = 0i64;
        let start = if self.write_idx.load(Ordering::Relaxed) > SLIPPAGE_BUFFER_SIZE as u64 {
            self.write_idx.load(Ordering::Relaxed) - SLIPPAGE_BUFFER_SIZE as u64
        } else {
            0
        };

        for i in start..self.write_idx.load(Ordering::Relaxed) {
            let idx = i % SLIPPAGE_BUFFER_SIZE as u64;
            sum += self.slippage_buffer[idx as usize].slippage_bps.load(Ordering::Relaxed);
        }

        sum / count as i64
    }

    /// Get TCA summary
    #[inline(always)]
    pub fn get_summary(&self) -> TcaResult {
        let trade_count = self.trade_count.load(Ordering::Relaxed);
        if trade_count == 0 {
            return TcaResult::empty();
        }

        let total_fees = self.total_fees.load(Ordering::Relaxed);
        let total_notional = self.total_notional.load(Ordering::Relaxed);
        let total_spread = self.total_spread_cost.load(Ordering::Relaxed);
        let avg_slippage = self.get_avg_slippage_bps();

        // Calculate costs in basis points
        let fee_bps = if total_notional > 0 {
            ((total_fees * 10000) / total_notional) as u16
        } else {
            0
        };

        let spread_bps = if total_notional > 0 {
            ((total_spread * 10000) / total_notional) as u16
        } else {
            0
        };

        let impact_bps = spread_bps / 2; // Simplified impact estimate
        let total_cost = (fee_bps as i32 + spread_bps as i32 + avg_slippage.abs() as i32) as u16;

        TcaResult {
            total_fees,
            avg_spread_bps: spread_bps,
            slippage_bps: avg_slippage as i16,
            market_impact_bps: impact_bps,
            total_cost_bps: total_cost,
            trade_count: trade_count as u32,
            _padding: [0u8; 16],
        }
    }

    /// SIMD-accelerated slippage calculation for multiple trades
    #[inline(always)]
    pub fn calculate_slippage_simd(
        &self,
        fill_prices: &[u64; 4],
        arrival_prices: &[u64; 4],
        quantities: &[u64; 4],
    ) -> [i64; 4] {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.calculate_slippage_avx2(fill_prices, arrival_prices, quantities)
            } else {
                let mut results = [0i64; 4];
                for i in 0..4 {
                    if arrival_prices[i] > 0 {
                        results[i] = ((fill_prices[i] as i64 - arrival_prices[i] as i64) * 10000) 
                            / arrival_prices[i] as i64;
                    }
                }
                results
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(always)]
    unsafe fn calculate_slippage_avx2(
        &self,
        fill_prices: &[u64; 4],
        arrival_prices: &[u64; 4],
        _quantities: &[u64; 4],
    ) -> [i64; 4] {
        use core::arch::x86_64::*;

        let fill_vec = _mm256_loadu_si256(fill_prices.as_ptr() as *const __m256i);
        let arrival_vec = _mm256_loadu_si256(arrival_prices.as_ptr() as *const __m256i);

        // Convert to double for division (simplified)
        let fill_lo = _mm256_cvtepi64_pd(_mm256_castsi256_si128(fill_vec));
        let arrival_lo = _mm256_cvtepi64_pd(_mm256_castsi256_si128(arrival_vec));

        // Calculate difference
        let diff = _mm256_sub_pd(fill_lo, arrival_lo);
        
        // Multiply by 10000 and divide by arrival
        let scale = _mm256_set1_pd(10000.0);
        let scaled = _mm256_mul_pd(diff, scale);
        let result = _mm256_div_pd(scaled, arrival_lo);

        // Convert back to integer
        let result_lo = _mm256_cvttpd_epi64(result);
        
        let mut results = [0i64; 4];
        _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, result_lo);
        
        results
    }

    /// Reset all TCA statistics
    #[inline(always)]
    pub fn reset(&self) {
        self.write_idx.store(0, Ordering::Release);
        self.total_fees.store(0, Ordering::Release);
        self.total_notional.store(0, Ordering::Release);
        self.total_spread_cost.store(0, Ordering::Release);
        self.trade_count.store(0, Ordering::Release);
        
        for i in 0..SLIPPAGE_BUFFER_SIZE {
            self.slippage_buffer[i] = SlippageSample::new();
        }
    }

    /// Get total fees
    #[inline(always)]
    pub fn get_total_fees(&self) -> u64 {
        self.total_fees.load(Ordering::Relaxed)
    }

    /// Get trade count
    #[inline(always)]
    pub fn get_trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }
}

impl Default for TcaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tca_result_size() {
        assert_eq!(core::mem::size_of::<TcaResult>(), 48);
    }

    #[test]
    fn test_record_trade() {
        let tca = TcaEngine::new();
        
        tca.record_trade(1, 100, 50000, 100, 5, 49950);
        
        assert_eq!(tca.get_trade_count(), 1);
        assert_eq!(tca.get_total_fees(), 100);
    }

    #[test]
    fn test_slippage_calculation() {
        let tca = TcaEngine::new();
        
        // Fill at higher price than arrival (negative slippage for buy)
        tca.record_trade(1, 100, 50100, 100, 5, 50000);
        
        let summary = tca.get_summary();
        assert!(summary.slippage_bps > 0); // Positive means worse price
    }

    #[test]
    fn test_tca_summary() {
        let tca = TcaEngine::new();
        
        for i in 0..10 {
            tca.record_trade(
                i,
                100,
                50000,
                50,
                5,
                50000,
            );
        }
        
        let summary = tca.get_summary();
        assert_eq!(summary.trade_count, 10);
        assert_eq!(summary.total_fees, 500);
    }

    #[test]
    fn test_circular_buffer() {
        let tca = TcaEngine::new();
        
        // Fill more than buffer size
        for i in 0..5000 {
            tca.record_trade(i, 100, 50000, 10, 5, 50000);
        }
        
        // Should still work (circular)
        let avg = tca.get_avg_slippage_bps();
        assert!(avg.is_finite());
    }

    #[test]
    fn test_slippage_simd() {
        let tca = TcaEngine::new();
        
        let fills = [50100u64, 50200, 49900, 50050];
        let arrivals = [50000u64, 50000, 50000, 50000];
        let quantities = [100u64, 100, 100, 100];
        
        let slippages = tca.calculate_slippage_simd(&fills, &arrivals, &quantities);
        
        assert!(slippages[0] > 0); // 50100 vs 50000 = +2 bps
        assert!(slippages[1] > slippages[0]); // 50200 vs 50000 = +4 bps
        assert!(slippages[2] < 0); // 49900 vs 50000 = -2 bps
    }
}
