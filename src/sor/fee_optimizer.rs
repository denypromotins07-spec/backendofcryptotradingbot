//! Fee Optimizer
//! 
//! Real-time fee tier tracker and routing optimizer for maker rebates.
//! Tracks exchange fee schedules, volume discounts, and optimizes order routing
//! to maximize maker rebates while minimizing taker fees.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::arch::x86_64::*;

use crate::common::fixed_point::FixedPoint;
use crate::execution::types::VenueId;

/// Fee tier structure for a venue
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FeeTier {
    pub min_volume: FixedPoint,          // Minimum 30-day volume for this tier
    pub maker_fee_bps: FixedPoint,       // Maker fee (negative = rebate)
    pub taker_fee_bps: FixedPoint,       // Taker fee
    _padding: [u8; 40],                  // Pad to 64 bytes
}

impl Default for FeeTier {
    fn default() -> Self {
        Self {
            min_volume: FixedPoint::ZERO,
            maker_fee_bps: FixedPoint::from_raw(10),  // 0.10% default
            taker_fee_bps: FixedPoint::from_raw(30),  // 0.30% default
            _padding: [0u8; 40],
        }
    }
}

/// Venue fee state with current tier
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VenueFeeState {
    pub venue_id: VenueId,
    pub current_tier_idx: u8,
    pub num_tiers: u8,
    pub tiers: [FeeTier; 8],             // Up to 8 fee tiers per venue
    pub volume_30d: FixedPoint,          // 30-day rolling volume
    pub last_update_cycle: u64,
    _padding: [u8; 24],                  // Pad to 128 bytes (2 cache lines)
}

impl Default for VenueFeeState {
    fn default() -> Self {
        Self {
            venue_id: 0,
            current_tier_idx: 0,
            num_tiers: 1,
            tiers: [FeeTier::default(); 8],
            volume_30d: FixedPoint::ZERO,
            last_update_cycle: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Optimal fee calculation result
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FeeOptimizationResult {
    pub best_venue: VenueId,
    pub best_maker_fee_bps: FixedPoint,
    pub best_taker_fee_bps: FixedPoint,
    pub expected_rebate_bps: FixedPoint,
    pub volume_required_for_next_tier: FixedPoint,
    _padding: [u8; 32],                  // Pad to 64 bytes
}

impl Default for FeeOptimizationResult {
    fn default() -> Self {
        Self {
            best_venue: 0,
            best_maker_fee_bps: FixedPoint::ZERO,
            best_taker_fee_bps: FixedPoint::ZERO,
            expected_rebate_bps: FixedPoint::ZERO,
            volume_required_for_next_tier: FixedPoint::MAX,
            _padding: [0u8; 32],
        }
    }
}

/// Shadow log entry for fee optimization decisions
#[repr(C)]
pub struct ShadowFeeLog {
    pub timestamp_cycles: u64,
    pub venue_id: VenueId,
    pub side: u8,                        // 0=maker, 1=taker
    pub applied_fee_bps: FixedPoint,
    pub optimal_fee_bps: FixedPoint,
    pub savings_bps: FixedPoint,
    _padding: [u8; 38],                  // Pad to 64 bytes
}

/// Fee optimizer with real-time tier tracking
#[repr(C)]
pub struct FeeOptimizer {
    /// Pre-allocated venue fee states (max 16 venues)
    venues: [VenueFeeState; 16],
    num_venues: AtomicUsize,
    
    /// Global volume tracking for fee calculations
    global_volume_30d: AtomicU64,        // Stored as fixed-point raw
    
    /// Shadow mode logging
    shadow_log_buffer: [ShadowFeeLog; 2048],
    shadow_log_head: AtomicUsize,
    shadow_enabled: AtomicU64,
    
    /// Statistics
    total_fees_saved_accum: AtomicU64,   // Accumulated savings in basis points * volume
    total_maker_volume: AtomicU64,
    total_taker_volume: AtomicU64,
    
    /// Circuit breaker
    halted: AtomicU64,
    
    _padding: [u8; 32],                  // Pad to cache line
}

impl FeeOptimizer {
    pub const fn new() -> Self {
        Self {
            venues: [VenueFeeState::default(); 16],
            num_venues: AtomicUsize::new(0),
            global_volume_30d: AtomicU64::new(0),
            shadow_log_buffer: unsafe { core::mem::zeroed() },
            shadow_log_head: AtomicUsize::new(0),
            shadow_enabled: AtomicU64::new(0),
            total_fees_saved_accum: AtomicU64::new(0),
            total_maker_volume: AtomicU64::new(0),
            total_taker_volume: AtomicU64::new(0),
            halted: AtomicU64::new(0),
            _padding: [0u8; 32],
        }
    }
    
    /// Register or update venue fee schedule
    #[inline]
    pub fn update_venue_fees(&self, state: VenueFeeState) {
        let idx = state.venue_id as usize;
        if idx >= 16 {
            return;
        }
        
        unsafe {
            core::ptr::write_volatile(&mut self.venues[idx] as *mut VenueFeeState, state);
        }
        
        let current = self.num_venues.load(Ordering::Relaxed);
        if idx >= current {
            self.num_venues.store(idx + 1, Ordering::Release);
        }
    }
    
    /// Update 30-day volume for a venue (triggers tier recalculation)
    #[inline]
    pub fn update_volume(&self, venue_id: VenueId, volume_30d: FixedPoint) {
        let idx = venue_id as usize;
        if idx >= 16 {
            return;
        }
        
        unsafe {
            let venue = &mut self.venues[idx];
            venue.volume_30d = volume_30d;
            
            // Find appropriate tier (branchless binary search for small arrays)
            let mut tier_idx = 0u8;
            for i in 1..venue.num_tiers as usize {
                if volume_30d >= venue.tiers[i].min_volume {
                    tier_idx = i as u8;
                }
            }
            venue.current_tier_idx = tier_idx;
            venue.last_update_cycle = self.read_rdtsc();
        }
    }
    
    /// Get current fee for a venue and order type
    #[inline]
    pub fn get_current_fee(&self, venue_id: VenueId, is_maker: bool) -> FixedPoint {
        let idx = venue_id as usize;
        if idx >= 16 {
            return FixedPoint::from_raw(30); // Default taker fee
        }
        
        unsafe {
            let venue = &self.venues[idx];
            let tier = &venue.tiers[venue.current_tier_idx as usize];
            
            if is_maker {
                tier.maker_fee_bps
            } else {
                tier.taker_fee_bps
            }
        }
    }
    
    /// Compute optimal venue for fee minimization using SIMD
    #[inline]
    pub fn compute_optimal_venue(&self, is_maker: bool, order_size: FixedPoint) -> FeeOptimizationResult {
        if self.halted.load(Ordering::Acquire) != 0 {
            return FeeOptimizationResult::default();
        }
        
        let num_venues = self.num_venues.load(Ordering::Acquire);
        if num_venues == 0 {
            return FeeOptimizationResult::default();
        }
        
        let mut result = FeeOptimizationResult::default();
        let mut best_fee = if is_maker { FixedPoint::MAX } else { FixedPoint::MAX };
        
        unsafe {
            for i in 0..num_venues.min(16) {
                let venue = &self.venues[i];
                let tier = &venue.tiers[venue.current_tier_idx as usize];
                
                let fee = if is_maker { tier.maker_fee_bps } else { tier.taker_fee_bps };
                
                // Branchless comparison
                let is_better = ((fee < best_fee) as u64).wrapping_neg(); // All 1s if true, 0s if false
                best_fee = fee ^ ((best_fee ^ fee) & is_better);
                
                if is_better != 0 {
                    result.best_venue = venue.venue_id;
                    if is_maker {
                        result.best_maker_fee_bps = fee;
                    } else {
                        result.best_taker_fee_bps = fee;
                    }
                    
                    // Calculate potential rebate (negative fee)
                    if is_maker && fee.to_raw() < 0 {
                        result.expected_rebate_bps = FixedPoint::from_raw(-fee.to_raw());
                    }
                    
                    // Volume needed for next tier
                    if (venue.current_tier_idx as usize) < (venue.num_tiers as usize - 1) {
                        let next_tier = &venue.tiers[(venue.current_tier_idx + 1) as usize];
                        result.volume_required_for_next_tier = next_tier.min_volume - venue.volume_30d;
                    }
                }
            }
        }
        
        result
    }
    
    /// Log fee optimization decision (shadow mode)
    #[inline]
    pub fn log_decision(&self, log_entry: ShadowFeeLog) {
        if self.shadow_enabled.load(Ordering::Relaxed) == 0 {
            return;
        }
        
        let head = self.shadow_log_head.fetch_add(1, Ordering::Relaxed);
        let idx = head % 2048;
        
        unsafe {
            core::ptr::write_volatile(&mut self.shadow_log_buffer[idx], log_entry);
        }
    }
    
    /// Record executed trade for statistics
    #[inline]
    pub fn record_trade(&self, venue_id: VenueId, volume: FixedPoint, is_maker: bool, fee_bps: FixedPoint) {
        if is_maker {
            let vol_raw = volume.to_raw() as u64;
            self.total_maker_volume.fetch_add(vol_raw, Ordering::Relaxed);
        } else {
            let vol_raw = volume.to_raw() as u64;
            self.total_taker_volume.fetch_add(vol_raw, Ordering::Relaxed);
        }
        
        // Track fee savings vs baseline
        let baseline_fee = FixedPoint::from_raw(30); // Baseline 0.30%
        let savings = baseline_fee - fee_bps;
        if savings > FixedPoint::ZERO {
            let savings_value = (savings.to_raw() as u64 * volume.to_raw() as u64) / 10000;
            self.total_fees_saved_accum.fetch_add(savings_value, Ordering::Relaxed);
        }
    }
    
    /// Halt fee optimization
    #[inline]
    pub fn halt(&self) {
        self.halted.store(1, Ordering::SeqCst);
    }
    
    /// Resume fee optimization
    #[inline]
    pub fn resume(&self) {
        self.halted.store(0, Ordering::SeqCst);
    }
    
    /// Enable shadow mode
    #[inline]
    pub fn enable_shadow_mode(&self) {
        self.shadow_enabled.store(1, Ordering::Relaxed);
    }
    
    /// Disable shadow mode
    #[inline]
    pub fn disable_shadow_mode(&self) {
        self.shadow_enabled.store(0, Ordering::Relaxed);
    }
    
    /// Get total maker volume
    #[inline]
    pub fn get_total_maker_volume(&self) -> FixedPoint {
        FixedPoint::from_raw(self.total_maker_volume.load(Ordering::Relaxed) as i64)
    }
    
    /// Get total taker volume
    #[inline]
    pub fn get_total_taker_volume(&self) -> FixedPoint {
        FixedPoint::from_raw(self.total_taker_volume.load(Ordering::Relaxed) as i64)
    }
    
    /// Read timestamp counter
    #[inline]
    fn read_rdtsc(&self) -> u64 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::x86_64::_rdtsc()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer_initialization() {
        let optimizer = FeeOptimizer::new();
        assert_eq!(optimizer.num_venues.load(Ordering::Relaxed), 0);
        assert_eq!(optimizer.halted.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_fee_tier_update() {
        let optimizer = FeeOptimizer::new();
        
        let mut venue = VenueFeeState::default();
        venue.venue_id = 0;
        venue.num_tiers = 3;
        venue.tiers[0] = FeeTier {
            min_volume: FixedPoint::ZERO,
            maker_fee_bps: FixedPoint::from_raw(-5), // -0.05% rebate
            taker_fee_bps: FixedPoint::from_raw(30),
            _padding: [0u8; 40],
        };
        venue.tiers[1] = FeeTier {
            min_volume: FixedPoint::from_raw(1000000),
            maker_fee_bps: FixedPoint::from_raw(-10), // -0.10% rebate
            taker_fee_bps: FixedPoint::from_raw(25),
            _padding: [0u8; 40],
        };
        venue.tiers[2] = FeeTier {
            min_volume: FixedPoint::from_raw(10000000),
            maker_fee_bps: FixedPoint::from_raw(-15), // -0.15% rebate
            taker_fee_bps: FixedPoint::from_raw(20),
            _padding: [0u8; 40],
        };
        
        optimizer.update_venue_fees(venue);
        
        // Test with low volume (tier 0)
        optimizer.update_volume(0, FixedPoint::from_raw(500000));
        assert_eq!(optimizer.get_current_fee(0, true).to_raw(), -5);
        
        // Test with high volume (tier 2)
        optimizer.update_volume(0, FixedPoint::from_raw(15000000));
        assert_eq!(optimizer.get_current_fee(0, true).to_raw(), -15);
    }
    
    #[test]
    fn test_optimal_venue_selection() {
        let optimizer = FeeOptimizer::new();
        
        // Venue 0: -0.05% maker rebate
        let mut venue0 = VenueFeeState::default();
        venue0.venue_id = 0;
        venue0.tiers[0].maker_fee_bps = FixedPoint::from_raw(-5);
        optimizer.update_venue_fees(venue0);
        
        // Venue 1: -0.10% maker rebate (better)
        let mut venue1 = VenueFeeState::default();
        venue1.venue_id = 1;
        venue1.tiers[0].maker_fee_bps = FixedPoint::from_raw(-10);
        optimizer.update_venue_fees(venue1);
        
        let result = optimizer.compute_optimal_venue(true, FixedPoint::from_raw(1000));
        assert_eq!(result.best_venue, 1);
        assert_eq!(result.best_maker_fee_bps.to_raw(), -10);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let optimizer = FeeOptimizer::new();
        
        optimizer.halt();
        assert_eq!(optimizer.halted.load(Ordering::Acquire), 1);
        
        optimizer.resume();
        assert_eq!(optimizer.halted.load(Ordering::Acquire), 0);
    }
}
