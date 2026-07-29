//! Real-Time Gamma Exposure (GEX) Tracker
//! Zero-copy strike mapper for dealer hedging flow estimation.
//! Lock-free aggregation across all expirations.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

/// Maximum strikes tracked
const MAX_STRIKES: usize = 512;
/// Maximum expirations tracked
const MAX_EXPIRIES: usize = 64;
/// Memory tracker
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);
const MEMORY_LIMIT_BYTES: u64 = 6_500_000_000;
/// GEX valid flag
static GEX_VALID: AtomicBool = AtomicBool::new(false);

/// Strike-level gamma data - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StrikeGamma {
    /// Strike price (scaled by 10^8)
    pub strike: i64,
    /// Call open interest
    pub call_oi: i64,
    /// Put open interest
    pub put_oi: i64,
    /// Call gamma (scaled by 10^10)
    pub call_gamma: i64,
    /// Put gamma (scaled by 10^10)
    pub put_gamma: i64,
    /// Net dealer gamma (scaled by 10^10)
    pub net_gamma: i64,
    /// Volume at strike
    pub volume: i64,
    _pad: [u8; 16],
}

impl Default for StrikeGamma {
    fn default() -> Self {
        Self {
            strike: 0,
            call_oi: 0,
            put_oi: 0,
            call_gamma: 0,
            put_gamma: 0,
            net_gamma: 0,
            volume: 0,
            _pad: [0u8; 16],
        }
    }
}

const _: () = assert!(core::mem::size_of::<StrikeGamma>() == 64);

/// Expiration bucket - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpiryBucket {
    /// Time to expiry in days
    pub days_to_expiry: u32,
    /// Number of active strikes
    pub num_strikes: u32,
    /// Total call gamma
    pub total_call_gamma: i64,
    /// Total put gamma
    pub total_put_gamma: i64,
    /// Net gamma
    pub net_gamma: i64,
    /// Zero-gamma level (flip point)
    pub zero_gamma_strike: i64,
    _pad: [u8; 32],
}

impl Default for ExpiryBucket {
    fn default() -> Self {
        Self {
            days_to_expiry: 0,
            num_strikes: 0,
            total_call_gamma: 0,
            total_put_gamma: 0,
            net_gamma: 0,
            zero_gamma_strike: 0,
            _pad: [0u8; 32],
        }
    }
}

const _: () = assert!(core::mem::size_of::<ExpiryBucket>() == 64);

/// Gamma Exposure Tracker - pre-allocated, lock-free
#[repr(C)]
pub struct GexTracker {
    /// Strike-level gamma data: [expiry][strike]
    pub strikes: [[StrikeGamma; MAX_STRIKES]; MAX_EXPIRIES],
    /// Expiration buckets
    pub expiries: [ExpiryBucket; MAX_EXPIRIES],
    /// Number of active expiries
    pub num_expiries: usize,
    /// Spot price (scaled)
    pub spot_price: AtomicI64,
    /// Total net gamma (scaled)
    pub total_net_gamma: AtomicI64,
    /// Gamma flip level (where net gamma = 0)
    pub gamma_flip: AtomicI64,
    /// Dealer position estimate (long/short gamma)
    pub dealer_position: AtomicI64,
    /// Last update timestamp
    pub last_update: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    _pad: [u8; 32],
}

impl Default for GexTracker {
    fn default() -> Self {
        Self {
            strikes: [[StrikeGamma::default(); MAX_STRIKES]; MAX_EXPIRIES],
            expiries: [ExpiryBucket::default(); MAX_EXPIRIES],
            num_expiries: 0,
            spot_price: AtomicI64::new(0),
            total_net_gamma: AtomicI64::new(0),
            gamma_flip: AtomicI64::new(0),
            dealer_position: AtomicI64::new(0),
            last_update: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _pad: [0u8; 32],
        }
    }
}

impl GexTracker {
    /// Create new GEX tracker
    pub const fn new() -> Self {
        Self::default()
    }
    
    /// Set strike gamma data (lock-free, zero-copy)
    #[inline(always)]
    pub fn set_strike_gamma(
        &self,
        expiry_idx: usize,
        strike_idx: usize,
        strike: i64,
        call_oi: i64,
        put_oi: i64,
        call_gamma: i64,
        put_gamma: i64,
    ) {
        if !GEX_VALID.load(Ordering::Relaxed) {
            return;
        }
        
        if expiry_idx >= MAX_EXPIRIES || strike_idx >= MAX_STRIKES {
            return;
        }
        
        // Net gamma = call_gamma * call_oi + put_gamma * put_oi
        // Note: dealer is short options, so their gamma is negative of long gamma
        let net = -(call_gamma.saturating_mul(call_oi) + put_gamma.saturating_mul(put_oi));
        
        unsafe {
            let s_ptr = self.strikes.as_ptr() as *mut StrikeGamma;
            let idx = expiry_idx * MAX_STRIKES + strike_idx;
            let sg = &mut *s_ptr.add(idx);
            
            sg.strike = strike;
            sg.call_oi = call_oi;
            sg.put_oi = put_oi;
            sg.call_gamma = call_gamma;
            sg.put_gamma = put_gamma;
            sg.net_gamma = net;
        }
        
        // Update expiry bucket count
        unsafe {
            let e_ptr = self.expiries.as_ptr() as *mut ExpiryBucket;
            let eb = &mut *e_ptr.add(expiry_idx);
            if strike_idx >= eb.num_strikes as usize {
                eb.num_strikes = (strike_idx + 1) as u32;
            }
        }
    }
    
    /// Aggregate gamma for an expiry
    pub fn aggregate_expiry(&self, expiry_idx: usize) {
        if expiry_idx >= MAX_EXPIRIES {
            return;
        }
        
        let mut total_call = 0i64;
        let mut total_put = 0i64;
        let mut net = 0i64;
        let mut zero_gamma = 0i64;
        let mut found_zero = false;
        
        let num_strikes = self.expiries[expiry_idx].num_strikes as usize;
        
        for i in 0..num_strikes.min(MAX_STRIKES) {
            let sg = &self.strikes[expiry_idx][i];
            total_call = total_call.saturating_add(sg.call_gamma.saturating_mul(sg.call_oi));
            total_put = total_put.saturating_add(sg.put_gamma.saturating_mul(sg.put_oi));
            net = net.saturating_add(sg.net_gamma);
            
            // Detect gamma flip (sign change)
            if !found_zero && i > 0 {
                let prev = &self.strikes[expiry_idx][i - 1];
                if (prev.net_gamma > 0 && sg.net_gamma < 0) || 
                   (prev.net_gamma < 0 && sg.net_gamma > 0) {
                    // Interpolate zero-gamma level
                    zero_gamma = (prev.strike + sg.strike) / 2;
                    found_zero = true;
                }
            }
        }
        
        unsafe {
            let e_ptr = self.expiries.as_ptr() as *mut ExpiryBucket;
            let eb = &mut *e_ptr.add(expiry_idx);
            eb.total_call_gamma = total_call;
            eb.total_put_gamma = total_put;
            eb.net_gamma = net;
            if found_zero {
                eb.zero_gamma_strike = zero_gamma;
            }
        }
    }
    
    /// Compute total market-wide GEX
    pub fn compute_total_gex(&self) {
        if !self.valid.load(Ordering::Acquire) {
            return;
        }
        
        let mut total = 0i64;
        let spot = self.spot_price.load(Ordering::Relaxed);
        let mut flip = spot;
        
        for i in 0..self.num_expiries {
            total = total.saturating_add(self.expiries[i].net_gamma);
            
            // Track nearest-to-spot flip level
            if self.expiries[i].zero_gamma_strike != 0 {
                flip = self.expiries[i].zero_gamma_strike;
            }
        }
        
        self.total_net_gamma.store(total, Ordering::Release);
        self.gamma_flip.store(flip, Ordering::Release);
        
        // Estimate dealer position
        // Positive GEX = dealers are long gamma (stable market)
        // Negative GEX = dealers are short gamma (volatile market)
        self.dealer_position.store(-total, Ordering::Release);
    }
    
    /// Get estimated dealer hedging flow
    #[inline(always)]
    pub fn get_hedging_flow(&self, price_change: i64) -> i64 {
        let gex = self.total_net_gamma.load(Ordering::Acquire);
        
        if gex == 0 {
            return 0;
        }
        
        // Dealer hedging = -GEX * delta_spot
        // Positive GEX: dealers buy dips, sell rallies (stabilizing)
        // Negative GEX: dealers sell dips, buy rallies (destabilizing)
        -gex.saturating_mul(price_change) / 1_000_000_000
    }
    
    /// Set spot price
    pub fn set_spot(&self, spot: i64) {
        self.spot_price.store(spot, Ordering::Release);
    }
    
    /// Mark tracker as valid
    pub fn mark_valid(&self) {
        #[cfg(target_arch = "x86_64")]
        let timestamp = unsafe { core::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        let timestamp = 0;
        
        self.last_update.store(timestamp, Ordering::Release);
        self.valid.store(true, Ordering::Release);
        GEX_VALID.store(true, Ordering::Relaxed);
    }
    
    /// Invalidate tracker
    pub fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
        GEX_VALID.store(false, Ordering::Relaxed);
    }
    
    /// Check if in negative gamma regime (high volatility expected)
    pub fn is_negative_gamma_regime(&self) -> bool {
        self.total_net_gamma.load(Ordering::Acquire) < 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_gex_aggregation(
            call_oi in 0i64..1_000_000i64,
            put_oi in 0i64..1_000_000i64,
            gamma in -1000i64..1000i64,
        ) {
            let tracker = GexTracker::new();
            tracker.mark_valid();
            
            // Setup single strike
            tracker.set_strike_gamma(0, 0, 5000000000, call_oi, put_oi, gamma, gamma);
            tracker.aggregate_expiry(0);
            tracker.num_expiries = 1;
            tracker.compute_total_gex();
            
            let total = tracker.total_net_gamma.load(Ordering::Relaxed);
            
            // GEX should be proportional to OI and gamma
            let expected_sign = if gamma > 0 { -1 } else { 1 };
            assert!(total.signum() == expected_sign || total == 0);
        }
        
        #[test]
        fn test_hedging_flow_direction(
            price_change in -10000i64..10000i64,
            gex in -1_000_000i64..1_000_000i64,
        ) {
            let tracker = GexTracker::new();
            tracker.total_net_gamma.store(gex, Ordering::Relaxed);
            
            let flow = tracker.get_hedging_flow(price_change);
            
            // Flow direction depends on GEX sign
            if gex > 0 && price_change < 0 {
                assert!(flow >= 0, "Positive GEX should buy dips");
            } else if gex > 0 && price_change > 0 {
                assert!(flow <= 0, "Positive GEX should sell rallies");
            }
        }
    }
    
    #[test]
    fn test_strike_gamma_size() {
        assert_eq!(core::mem::size_of::<StrikeGamma>(), 64);
    }
    
    #[test]
    fn test_expiry_bucket_size() {
        assert_eq!(core::mem::size_of::<ExpiryBucket>(), 64);
    }
    
    #[test]
    fn test_negative_gamma_detection() {
        let tracker = GexTracker::new();
        tracker.total_net_gamma.store(-1_000_000, Ordering::Relaxed);
        assert!(tracker.is_negative_gamma_regime());
        
        tracker.total_net_gamma.store(1_000_000, Ordering::Relaxed);
        assert!(!tracker.is_negative_gamma_regime());
    }
}
