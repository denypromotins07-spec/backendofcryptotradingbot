//! Chapter 3: Layer 2, Cross-Chain, and Bridging Analytics
//! Cross-chain bridge liquidity, lock/unlock events, and arbitrage spread calculator.

use core::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line padding for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum bridge pairs tracked
const MAX_BRIDGES: usize = 16;

/// Fixed-point scaling factor (1e9)
const FP_SCALE: i64 = 1_000_000_000;

#[repr(C, align(64))]
pub struct BridgeLiquidity {
    /// Available liquidity on source chain (fixed-point)
    source_liquidity: [AtomicI64; MAX_BRIDGES],
    /// Available liquidity on destination chain (fixed-point)
    dest_liquidity: [AtomicI64; MAX_BRIDGES],
    /// Locked amount pending transfer (fixed-point)
    locked_amount: [AtomicI64; MAX_BRIDGES],
    /// Last update timestamp (TSC cycles)
    last_update: [AtomicU64; MAX_BRIDGES],
    _padding: [u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
}

#[repr(C, align(64))]
pub struct BridgeEventTracker {
    /// Total lock events per bridge
    lock_count: [AtomicU64; MAX_BRIDGES],
    /// Total unlock events per bridge
    unlock_count: [AtomicU64; MAX_BRIDGES],
    /// Total value locked (cumulative, fixed-point)
    total_value_locked: [AtomicI64; MAX_BRIDGES],
    /// Pending unlocks (fixed-point)
    pending_unlocks: [AtomicI64; MAX_BRIDGES],
    _padding: [u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
}

#[repr(C, align(64))]
pub struct ArbitrageSpreadCalculator {
    /// Price on source chain (fixed-point)
    source_price: [AtomicI64; MAX_BRIDGES],
    /// Price on destination chain (fixed-point)
    dest_price: [AtomicI64; MAX_BRIDGES],
    /// Bridge fee (fixed-point)
    bridge_fee: [AtomicI64; MAX_BRIDGES],
    /// Last calculated spread (fixed-point, basis points)
    last_spread_bp: [AtomicI64; MAX_BRIDGES],
    _padding: [u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
}

// Compile-time assertions
const _: () = {
    assert!(MAX_BRIDGES <= 16, "MAX_BRIDGES exceeds limit");
};

impl Default for BridgeLiquidity {
    fn default() -> Self {
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        
        Self {
            source_liquidity: [INIT_I64; MAX_BRIDGES],
            dest_liquidity: [INIT_I64; MAX_BRIDGES],
            locked_amount: [INIT_I64; MAX_BRIDGES],
            last_update: [INIT_U64; MAX_BRIDGES],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
        }
    }
}

impl Default for BridgeEventTracker {
    fn default() -> Self {
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        const INIT_U64: AtomicU64 = AtomicU64::new(0);
        
        Self {
            lock_count: [INIT_U64; MAX_BRIDGES],
            unlock_count: [INIT_U64; MAX_BRIDGES],
            total_value_locked: [INIT_I64; MAX_BRIDGES],
            pending_unlocks: [INIT_I64; MAX_BRIDGES],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
        }
    }
}

impl Default for ArbitrageSpreadCalculator {
    fn default() -> Self {
        const INIT_I64: AtomicI64 = AtomicI64::new(0);
        
        Self {
            source_price: [INIT_I64; MAX_BRIDGES],
            dest_price: [INIT_I64; MAX_BRIDGES],
            bridge_fee: [INIT_I64; MAX_BRIDGES],
            last_spread_bp: [INIT_I64; MAX_BRIDGES],
            _padding: [0u8; CACHE_LINE_SIZE - MAX_BRIDGES * 4 * 8],
        }
    }
}

impl BridgeLiquidity {
    /// Initialize bridge liquidity tracking
    #[inline]
    pub fn init_bridge(&self, bridge_id: usize, initial_liq: i64) {
        if bridge_id >= MAX_BRIDGES {
            return;
        }
        
        self.source_liquidity[bridge_id].store(initial_liq, Ordering::Relaxed);
        self.dest_liquidity[bridge_id].store(initial_liq, Ordering::Relaxed);
        self.last_update[bridge_id].store(unsafe { _rdtsc() }, Ordering::Relaxed);
    }

    /// Record lock event (assets locked on source chain)
    #[inline]
    pub fn lock_assets(&self, bridge_id: usize, amount: i64) -> bool {
        if bridge_id >= MAX_BRIDGES {
            return false;
        }
        
        let source_liq = self.source_liquidity[bridge_id].load(Ordering::Relaxed);
        if amount > source_liq {
            return false; // Insufficient liquidity
        }
        
        self.source_liquidity[bridge_id].fetch_sub(amount, Ordering::Relaxed);
        self.locked_amount[bridge_id].fetch_add(amount, Ordering::Relaxed);
        self.last_update[bridge_id].store(unsafe { _rdtsc() }, Ordering::Relaxed);
        
        true
    }

    /// Record unlock event (assets released on destination chain)
    #[inline]
    pub fn unlock_assets(&self, bridge_id: usize, amount: i64) -> bool {
        if bridge_id >= MAX_BRIDGES {
            return false;
        }
        
        let locked = self.locked_amount[bridge_id].load(Ordering::Relaxed);
        if amount > locked {
            return false; // Cannot unlock more than locked
        }
        
        self.locked_amount[bridge_id].fetch_sub(amount, Ordering::Relaxed);
        self.dest_liquidity[bridge_id].fetch_add(amount, Ordering::Relaxed);
        self.last_update[bridge_id].store(unsafe { _rdtsc() }, Ordering::Relaxed);
        
        true
    }

    /// Get available liquidity on source chain
    #[inline]
    pub fn get_source_liquidity(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.source_liquidity[bridge_id].load(Ordering::Relaxed)
    }

    /// Get available liquidity on destination chain
    #[inline]
    pub fn get_dest_liquidity(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.dest_liquidity[bridge_id].load(Ordering::Relaxed)
    }

    /// Get locked amount
    #[inline]
    pub fn get_locked_amount(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.locked_amount[bridge_id].load(Ordering::Relaxed)
    }

    /// SIMD-accelerated liquidity check across multiple bridges
    #[inline]
    pub fn simd_check_liquidity<const N: usize>(&self, bridge_ids: [usize; N], min_amount: i64) -> u32
    where [usize; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut has_liquidity_mask = 0u32;
        
        unsafe {
            if N == 4 {
                let mut liqudities = [0i64; 4];
                for i in 0..4 {
                    liqudities[i] = self.source_liquidity[bridge_ids[i]].load(Ordering::Relaxed);
                }
                
                let liq_vec = _mm256_load_si256(liqudities.as_ptr() as *const __m256i);
                let min_vec = _mm256_set1_epi64x(min_amount);
                
                // Compare: liq >= min_amount
                let cmp_vec = _mm256_cmpgt_epi64(liq_vec, min_vec);
                
                let mask = _mm256_movemask_epi8(cmp_vec);
                has_liquidity_mask = ((mask & 0x1) | ((mask >> 4) & 0x2) | ((mask >> 7) & 0x4) | ((mask >> 11) & 0x8)) as u32;
                
                _mm256_zeroupper();
            } else {
                for i in 0..N {
                    let liq = self.source_liquidity[bridge_ids[i]].load(Ordering::Relaxed);
                    let bit = ((liq >= min_amount) as u32) << i;
                    has_liquidity_mask |= bit;
                }
            }
        }
        
        has_liquidity_mask
    }

    /// Calculate total liquidity across all bridges
    #[inline]
    pub fn total_liquidity(&self) -> i64 {
        let mut total = 0i64;
        for i in 0..MAX_BRIDGES {
            total += self.source_liquidity[i].load(Ordering::Relaxed);
            total += self.dest_liquidity[i].load(Ordering::Relaxed);
        }
        total
    }
}

impl BridgeEventTracker {
    /// Record a lock event
    #[inline]
    pub fn record_lock(&self, bridge_id: usize, amount: i64) {
        if bridge_id >= MAX_BRIDGES {
            return;
        }
        
        self.lock_count[bridge_id].fetch_add(1, Ordering::Relaxed);
        self.total_value_locked[bridge_id].fetch_add(amount, Ordering::Relaxed);
        self.pending_unlocks[bridge_id].fetch_add(amount, Ordering::Relaxed);
    }

    /// Record an unlock event
    #[inline]
    pub fn record_unlock(&self, bridge_id: usize, amount: i64) {
        if bridge_id >= MAX_BRIDGES {
            return;
        }
        
        self.unlock_count[bridge_id].fetch_add(1, Ordering::Relaxed);
        let pending = self.pending_unlocks[bridge_id].fetch_sub(amount, Ordering::Relaxed);
        
        // Ensure we don't go negative
        if pending < amount {
            self.pending_unlocks[bridge_id].store(0, Ordering::Relaxed);
        }
    }

    /// Get lock count
    #[inline]
    pub fn get_lock_count(&self, bridge_id: usize) -> u64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.lock_count[bridge_id].load(Ordering::Relaxed)
    }

    /// Get unlock count
    #[inline]
    pub fn get_unlock_count(&self, bridge_id: usize) -> u64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.unlock_count[bridge_id].load(Ordering::Relaxed)
    }

    /// Get TVL
    #[inline]
    pub fn get_tvl(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.total_value_locked[bridge_id].load(Ordering::Relaxed)
    }

    /// Get pending unlocks
    #[inline]
    pub fn get_pending_unlocks(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.pending_unlocks[bridge_id].load(Ordering::Relaxed)
    }

    /// Get lock/unlock ratio (scaled by 1e9)
    #[inline]
    pub fn get_lock_unlock_ratio(&self, bridge_id: usize) -> i64 {
        let locks = self.lock_count[bridge_id].load(Ordering::Relaxed);
        let unlocks = self.unlock_count[bridge_id].load(Ordering::Relaxed);
        
        if unlocks == 0 {
            return if locks > 0 { i64::MAX } else { 0 };
        }
        
        (locks * FP_SCALE) / unlocks
    }
}

impl ArbitrageSpreadCalculator {
    /// Update prices for a bridge pair
    #[inline]
    pub fn update_prices(&self, bridge_id: usize, source: i64, dest: i64, fee: i64) {
        if bridge_id >= MAX_BRIDGES {
            return;
        }
        
        self.source_price[bridge_id].store(source, Ordering::Relaxed);
        self.dest_price[bridge_id].store(dest, Ordering::Relaxed);
        self.bridge_fee[bridge_id].store(fee, Ordering::Relaxed);
        
        // Calculate spread in basis points
        let spread = if source > 0 {
            ((dest - source) * 10_000) / source
        } else {
            0
        };
        
        self.last_spread_bp[bridge_id].store(spread, Ordering::Relaxed);
    }

    /// Get current spread in basis points
    #[inline]
    pub fn get_spread_bp(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        self.last_spread_bp[bridge_id].load(Ordering::Relaxed)
    }

    /// Check if arbitrage opportunity exists (spread > fees)
    #[inline]
    pub fn is_arb_opportunity(&self, bridge_id: usize) -> bool {
        if bridge_id >= MAX_BRIDGES {
            return false;
        }
        
        let spread = self.last_spread_bp[bridge_id].load(Ordering::Relaxed).abs();
        let fee = self.bridge_fee[bridge_id].load(Ordering::Relaxed);
        
        spread > fee
    }

    /// Get profitable arb direction (positive = source->dest, negative = dest->source)
    #[inline]
    pub fn get_arb_direction(&self, bridge_id: usize) -> i64 {
        if bridge_id >= MAX_BRIDGES {
            return 0;
        }
        
        let source = self.source_price[bridge_id].load(Ordering::Relaxed);
        let dest = self.dest_price[bridge_id].load(Ordering::Relaxed);
        let fee = self.bridge_fee[bridge_id].load(Ordering::Relaxed);
        
        let gross_spread = dest - source;
        let net_spread = gross_spread - fee;
        
        if net_spread.abs() > fee {
            net_spread
        } else {
            0
        }
    }

    /// SIMD-accelerated spread calculation for multiple bridges
    #[inline]
    pub fn simd_calc_spreads<const N: usize>(&self, bridge_ids: [usize; N]) -> [i64; N]
    where [usize; N]: Copy
    {
        assert!(N <= 4, "SIMD batch size must be <= 4 for AVX2");
        let mut spreads = [0i64; N];
        
        unsafe {
            if N == 4 {
                let mut sources = [0i64; 4];
                let mut dests = [0i64; 4];
                
                for i in 0..4 {
                    sources[i] = self.source_price[bridge_ids[i]].load(Ordering::Relaxed);
                    dests[i] = self.dest_price[bridge_ids[i]].load(Ordering::Relaxed);
                }
                
                let src_vec = _mm256_load_si256(sources.as_ptr() as *const __m256i);
                let dst_vec = _mm256_load_si256(dests.as_ptr() as *const __m256i);
                
                // Calculate differences
                let diff_vec = _mm256_sub_epi64(dst_vec, src_vec);
                
                _mm256_storeu_si256(spreads.as_mut_ptr() as *mut __m256i, diff_vec);
                
                // Convert to basis points (simplified, assumes similar price levels)
                for i in 0..4 {
                    if sources[i] != 0 {
                        spreads[i] = (spreads[i] * 10_000) / sources[i];
                    }
                }
                
                _mm256_zeroupper();
            } else {
                for i in 0..N {
                    let src = self.source_price[bridge_ids[i]].load(Ordering::Relaxed);
                    let dst = self.dest_price[bridge_ids[i]].load(Ordering::Relaxed);
                    if src != 0 {
                        spreads[i] = ((dst - src) * 10_000) / src;
                    }
                }
            }
        }
        
        spreads
    }

    /// Get max arb opportunity across all bridges
    #[inline]
    pub fn find_best_arb(&self) -> Option<(usize, i64)> {
        let mut best_idx = None;
        let mut best_spread = 0i64;
        
        for i in 0..MAX_BRIDGES {
            if self.is_arb_opportunity(i) {
                let direction = self.get_arb_direction(i).abs();
                if direction > best_spread {
                    best_spread = direction;
                    best_idx = Some(i);
                }
            }
        }
        
        best_idx.map(|idx| (idx, best_spread))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_lock_unlock() {
        let liq = BridgeLiquidity::default();
        liq.init_bridge(0, 1_000_000 * FP_SCALE);
        
        // Lock assets
        assert!(liq.lock_assets(0, 100_000 * FP_SCALE));
        assert_eq!(liq.get_source_liquidity(0), 900_000 * FP_SCALE);
        assert_eq!(liq.get_locked_amount(0), 100_000 * FP_SCALE);
        
        // Unlock assets
        assert!(liq.unlock_assets(0, 100_000 * FP_SCALE));
        assert_eq!(liq.get_locked_amount(0), 0);
        assert_eq!(liq.get_dest_liquidity(0), 1_100_000 * FP_SCALE);
    }

    #[test]
    fn test_insufficient_liquidity() {
        let liq = BridgeLiquidity::default();
        liq.init_bridge(0, 100_000 * FP_SCALE);
        
        // Try to lock more than available
        assert!(!liq.lock_assets(0, 200_000 * FP_SCALE));
    }

    #[test]
    fn test_event_tracking() {
        let tracker = BridgeEventTracker::default();
        
        tracker.record_lock(0, 100_000 * FP_SCALE);
        tracker.record_lock(0, 50_000 * FP_SCALE);
        tracker.record_unlock(0, 75_000 * FP_SCALE);
        
        assert_eq!(tracker.get_lock_count(0), 2);
        assert_eq!(tracker.get_unlock_count(0), 1);
        assert_eq!(tracker.get_tvl(0), 150_000 * FP_SCALE);
        assert_eq!(tracker.get_pending_unlocks(0), 75_000 * FP_SCALE);
    }

    #[test]
    fn test_arbitrage_detection() {
        let calc = ArbitrageSpreadCalculator::default();
        
        // No arb initially
        calc.update_prices(0, 100_000, 100_000, 100); // Same price, 0.01% fee
        assert!(!calc.is_arb_opportunity(0));
        
        // Create arb opportunity
        calc.update_prices(0, 100_000, 100_500, 100); // 0.5% price difference
        assert!(calc.is_arb_opportunity(0));
        
        let direction = calc.get_arb_direction(0);
        assert!(direction > 0); // Should be positive (source -> dest)
    }

    #[test]
    fn test_simd_liquidity_check() {
        let liq = BridgeLiquidity::default();
        liq.init_bridge(0, 1_000_000);
        liq.init_bridge(1, 500_000);
        liq.init_bridge(2, 100_000);
        liq.init_bridge(3, 2_000_000);
        
        let bridges = [0, 1, 2, 3];
        let mask = liq.simd_check_liquidity(bridges, 200_000);
        
        // Bridges 0, 1, 3 have enough liquidity; 2 does not
        assert_eq!(mask, 0b1011);
    }

    #[test]
    fn test_best_arb_finder() {
        let calc = ArbitrageSpreadCalculator::default();
        
        calc.update_prices(0, 100_000, 100_100, 50); // Small spread
        calc.update_prices(1, 100_000, 101_000, 50); // Large spread
        
        let best = calc.find_best_arb();
        assert_eq!(best, Some((1, 950))); // Bridge 1 with ~950 bps net spread
    }

    #[test]
    fn test_lock_unlock_ratio() {
        let tracker = BridgeEventTracker::default();
        
        tracker.record_lock(0, 100);
        tracker.record_lock(0, 100);
        tracker.record_lock(0, 100);
        tracker.record_unlock(0, 100);
        
        let ratio = tracker.get_lock_unlock_ratio(0);
        assert_eq!(ratio, 3_000_000_000); // 3:1 ratio scaled
    }
}
