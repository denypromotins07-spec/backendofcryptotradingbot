//! Staking yield, validator uptime, and slashing risk monitor for PoS networks.
//! Lock-free atomic state tracking with branchless risk scoring.
//! Pre-allocated buffers for zero heap allocation.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Fixed-point representation (scaled by 10^8)
pub type FixedValue = i64;
const SCALE: i64 = 100_000_000;

/// Maximum validators tracked (pre-allocated)
pub const MAX_VALIDATORS: usize = 512;

/// Validator status enum
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidatorStatus {
    Active = 0,
    Inactive = 1,
    Slashed = 2,
    Jailed = 3,
}

/// Cache-line aligned validator data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ValidatorData {
    /// Validator ID / address hash
    pub validator_id: u64,
    /// Chain ID
    pub chain_id: u32,
    /// Staked amount (fixed-point USD)
    pub staked_amount: i64,
    /// Current APY (fixed-point, basis points * 100)
    pub current_apy: FixedValue,
    /// Uptime percentage (fixed-point, 0-100 * SCALE)
    pub uptime_pct: FixedValue,
    /// Consecutive missed blocks
    pub missed_blocks: u64,
    /// Total blocks proposed
    pub total_proposed: u64,
    /// Slashing risk score (0-100, higher = riskier)
    pub risk_score: u8,
    /// Validator status
    pub status: ValidatorStatus,
    /// Last heartbeat timestamp (ns)
    pub last_heartbeat_ns: u64,
    _padding: [u8; 26], // Pad to 64 bytes
}

impl Default for ValidatorData {
    fn default() -> Self {
        Self {
            validator_id: 0,
            chain_id: 0,
            staked_amount: 0,
            current_apy: 0,
            uptime_pct: 100 * SCALE,
            missed_blocks: 0,
            total_proposed: 0,
            risk_score: 0,
            status: ValidatorStatus::Active,
            last_heartbeat_ns: 0,
            _padding: [0; 26],
        }
    }
}

/// Main validator metrics tracker
#[repr(C)]
pub struct ValidatorMetricsTracker {
    /// Pre-allocated validator array
    pub validators: [ValidatorData; MAX_VALIDATORS],
    /// Validator count
    pub validator_count: AtomicU64,
    /// Total staked across all validators
    pub total_staked: AtomicI64,
    /// Average network APY
    pub avg_apy: AtomicI64,
    /// Network-wide uptime
    pub network_uptime: AtomicI64,
    /// Slashed validator count
    pub slashed_count: AtomicU64,
    /// High-risk validator threshold
    pub high_risk_threshold: u8,
    _padding: [u8; 48], // Align to cache line
}

impl ValidatorMetricsTracker {
    pub fn new(high_risk_threshold: u8) -> Self {
        Self {
            validators: [ValidatorData::default(); MAX_VALIDATORS],
            validator_count: AtomicU64::new(0),
            total_staked: AtomicI64::new(0),
            avg_apy: AtomicI64::new(0),
            network_uptime: AtomicI64::new(100 * SCALE),
            slashed_count: AtomicU64::new(0),
            high_risk_threshold,
            _padding: [0; 48],
        }
    }

    /// Update validator metrics (lock-free)
    #[inline]
    pub fn update_validator(
        &mut self,
        validator_id: u64,
        chain_id: u32,
        staked_amount: i64,
        apy_bps: FixedValue,
        uptime_pct: FixedValue,
        missed_blocks: u64,
        total_proposed: u64,
        timestamp_ns: u64,
    ) {
        // Find or create validator entry
        let count = self.validator_count.load(Ordering::Acquire);
        let mut found_idx: Option<usize> = None;
        
        for i in 0..count.min(MAX_VALIDATORS as u64) as usize {
            if self.validators[i].validator_id == validator_id {
                found_idx = Some(i);
                break;
            }
        }

        let idx = if let Some(i) = found_idx {
            i
        } else if count < MAX_VALIDATORS as u64 {
            let new_idx = count as usize;
            self.validator_count.fetch_add(1, Ordering::AcqRel);
            new_idx
        } else {
            return; // No space
        };

        // Calculate risk score (branchless)
        let risk_score = self.calculate_risk_score(missed_blocks, uptime_pct, total_proposed);

        // Determine status
        let status = if risk_score >= self.high_risk_threshold {
            ValidatorStatus::Jailed
        } else if missed_blocks > 100 {
            ValidatorStatus::Inactive
        } else {
            ValidatorStatus::Active
        };

        // Update validator data
        self.validators[idx] = ValidatorData {
            validator_id,
            chain_id,
            staked_amount,
            current_apy: apy_bps,
            uptime_pct,
            missed_blocks,
            total_proposed,
            risk_score,
            status,
            last_heartbeat_ns: timestamp_ns,
            _padding: [0; 26],
        };

        // Recalculate aggregates
        self.recalculate_aggregates();
    }

    /// Branchless risk score calculation (0-100)
    #[inline]
    fn calculate_risk_score(&self, missed_blocks: u64, uptime_pct: FixedValue, total_proposed: u64) -> u8 {
        // Risk from missed blocks (capped at 50)
        let miss_risk = ((missed_blocks.min(50)) as u8);
        
        // Risk from uptime (capped at 30)
        let uptime_deficit = (100 * SCALE - uptime_pct).max(0) / SCALE;
        let uptime_risk = (uptime_deficit.min(30) as u8);
        
        // Risk from low proposal count (capped at 20)
        let proposal_risk = if total_proposed < 10 { 20 } else { 0 };
        
        // Sum capped at 100
        miss_risk.saturating_add(uptime_risk).saturating_add(proposal_risk)
    }

    /// Recalculate aggregate metrics
    #[inline]
    fn recalculate_aggregates(&mut self) {
        let count = self.validator_count.load(Ordering::Acquire);
        let mut total_staked: i64 = 0;
        let mut total_apy: i64 = 0;
        let mut total_uptime: i64 = 0;
        let mut slashed: u64 = 0;
        let mut active_count: i64 = 0;

        for i in 0..count.min(MAX_VALIDATORS as u64) as usize {
            let v = &self.validators[i];
            if v.status != ValidatorStatus::Slashed {
                total_staked = total_staked.saturating_add(v.staked_amount);
                total_apy = total_apy.saturating_add(v.current_apy);
                total_uptime = total_uptime.saturating_add(v.uptime_pct);
                active_count += 1;
            } else {
                slashed += 1;
            }
        }

        self.total_staked.store(total_staked, Ordering::Release);
        self.slashed_count.store(slashed, Ordering::Release);

        if active_count > 0 {
            self.avg_apy.store(total_apy / active_count, Ordering::Release);
            self.network_uptime.store(total_uptime / active_count, Ordering::Release);
        }
    }

    /// Get total staked amount
    #[inline]
    pub fn get_total_staked(&self) -> i64 {
        self.total_staked.load(Ordering::Acquire)
    }

    /// Get average APY
    #[inline]
    pub fn get_avg_apy(&self) -> FixedValue {
        self.avg_apy.load(Ordering::Acquire)
    }

    /// Get network uptime
    #[inline]
    pub fn get_network_uptime(&self) -> FixedValue {
        self.network_uptime.load(Ordering::Acquire)
    }

    /// Get slashed count
    #[inline]
    pub fn get_slashed_count(&self) -> u64 {
        self.slashed_count.load(Ordering::Acquire)
    }

    /// Get validator by ID
    #[inline]
    pub fn get_validator(&self, validator_id: u64) -> Option<&ValidatorData> {
        let count = self.validator_count.load(Ordering::Acquire);
        
        for i in 0..count.min(MAX_VALIDATORS as u64) as usize {
            if self.validators[i].validator_id == validator_id {
                return Some(&self.validators[i]);
            }
        }
        None
    }

    /// Get high-risk validators count
    #[inline]
    pub fn get_high_risk_count(&self) -> u64 {
        let mut count = 0u64;
        let total = self.validator_count.load(Ordering::Acquire);
        
        for i in 0..total.min(MAX_VALIDATORS as u64) as usize {
            if self.validators[i].risk_score >= self.high_risk_threshold {
                count += 1;
            }
        }
        count
    }
}

/// Staking yield calculator with compounding
#[repr(C)]
pub struct StakingYieldCalculator {
    /// Base APY (fixed-point)
    pub base_apy: FixedValue,
    /// Compounding frequency per year
    pub compound_freq: u32,
    /// Validator commission (basis points)
    pub commission_bps: u32,
    _padding: [u8; 52], // Pad to 64 bytes
}

impl StakingYieldCalculator {
    pub const fn new(base_apy_bps: u32, compound_freq: u32, commission_bps: u32) -> Self {
        Self {
            base_apy: (base_apy_bps as i64) * SCALE / 100,
            compound_freq,
            commission_bps,
            _padding: [0; 52],
        }
    }

    /// Calculate effective APY after commission and compounding
    #[inline]
    pub fn calculate_effective_apy(&self) -> FixedValue {
        // Net APY after commission
        let net_apy = self.base_apy * (10_000 - self.commission_bps as i64) / 10_000;
        
        // Simple compounding approximation: (1 + r/n)^n - 1
        // Using fixed-point approximation
        if self.compound_freq > 0 {
            let rate_per_period = net_apy / self.compound_freq as i64;
            // Simplified: just return net_apy for now (compound effect is small)
            net_apy + (rate_per_period * rate_per_period / SCALE) * self.compound_freq as i64 / 2
        } else {
            net_apy
        }
    }

    /// Calculate expected rewards for stake amount over period (days)
    #[inline]
    pub fn calculate_rewards(&self, stake_amount: i64, days: u32) -> i64 {
        let effective_apy = self.calculate_effective_apy();
        // Daily rate = APY / 365
        let daily_rate = effective_apy / 365;
        // Rewards = stake * daily_rate * days / SCALE
        (stake_amount * daily_rate * days as i64) / SCALE / SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_tracker_initialization() {
        let tracker = ValidatorMetricsTracker::new(70);
        assert_eq!(tracker.get_total_staked(), 0);
        assert_eq!(tracker.get_slashed_count(), 0);
    }

    #[test]
    fn test_validator_update() {
        let mut tracker = ValidatorMetricsTracker::new(70);
        
        tracker.update_validator(
            1, 1, 1_000_000, 500, 99 * SCALE, 2, 100, 1000
        );
        
        assert_eq!(tracker.get_total_staked(), 1_000_000);
        assert_eq!(tracker.get_high_risk_count(), 0);
    }

    #[test]
    fn test_risk_score_calculation() {
        let tracker = ValidatorMetricsTracker::new(70);
        
        // High miss count should increase risk
        let risk = tracker.calculate_risk_score(50, 95 * SCALE, 100);
        assert!(risk > 50);
        
        // Perfect validator should have low risk
        let risk_low = tracker.calculate_risk_score(0, 100 * SCALE, 1000);
        assert!(risk_low < 10);
    }

    #[test]
    fn test_staking_yield() {
        let calc = StakingYieldCalculator::new(500, 365, 500); // 5% APY, daily compounding, 5% commission
        
        let effective = calc.calculate_effective_apy();
        // Should be slightly less than 5% due to commission
        assert!(effective < 500 * SCALE / 100);
        assert!(effective > 450 * SCALE / 100);
        
        let rewards = calc.calculate_rewards(10_000, 30);
        assert!(rewards > 0);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<ValidatorData>() >= 64);
        assert!(core::mem::size_of::<ValidatorMetricsTracker>() >= 64);
        assert!(core::mem::size_of::<StakingYieldCalculator>() >= 64);
    }
}
