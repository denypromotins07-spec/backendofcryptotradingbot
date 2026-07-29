//! Vesting schedule and token unlock event calendar with pre-event risk scaling.
//! 
//! Uses fixed-point arithmetic, circular buffers for rolling calculations,
//! and cache-line aligned structs to prevent false sharing in hot paths.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of tracked vesting schedules
const MAX_VESTING_SCHEDULES: usize = 256;

/// Maximum number of upcoming unlock events tracked
const MAX_UPCOMING_UNLOCKS: usize = 128;

/// Fixed-point scale (6 decimal precision)
const FIXED_SCALE: u64 = 1_000_000;

/// Risk multiplier scale (4 decimal precision for multipliers)
const RISK_SCALE: u64 = 10_000;

/// Padded atomic u64 for cache-line alignment
#[repr(C)]
struct PaddedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 8],
}

impl PaddedAtomicU64 {
    const fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; CACHE_LINE_SIZE - 8],
        }
    }
    
    #[inline]
    fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline]
    fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed);
    }
    
    #[inline]
    fn fetch_add(&self, delta: u64) -> u64 {
        self.value.fetch_add(delta, Ordering::Relaxed)
    }
}

/// Vesting schedule entry - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct VestingSchedule {
    /// Schedule ID
    schedule_id: u64,
    /// Token identifier
    token_id: u32,
    /// Total locked amount (fixed-point)
    total_locked_fixed: u64,
    /// Unlocked amount so far (fixed-point)
    unlocked_fixed: u64,
    /// Start timestamp (Unix epoch seconds)
    start_timestamp: u64,
    /// End timestamp (Unix epoch seconds)
    end_timestamp: u64,
    /// Cliff timestamp (no unlocks before this)
    cliff_timestamp: u64,
    /// Unlock interval in seconds
    unlock_interval_sec: u32,
    /// Amount per unlock (fixed-point)
    amount_per_unlock_fixed: u64,
    /// Number of total unlocks
    total_unlocks: u32,
    /// Number of completed unlocks
    completed_unlocks: u32,
    /// Is active
    is_active: bool,
    /// Is linear vesting (vs graded)
    is_linear: bool,
    /// Padding to reach 64 bytes
    _padding: [u8; 34],
}

const _: () = assert!(core::mem::size_of::<VestingSchedule>() == 64);

/// Upcoming unlock event - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct UnlockEvent {
    /// Event ID
    event_id: u64,
    /// Schedule ID reference
    schedule_id: u64,
    /// Token ID
    token_id: u32,
    /// Unlock amount (fixed-point)
    unlock_amount_fixed: u64,
    /// Timestamp (Unix epoch seconds)
    timestamp: u64,
    /// Recipient address hash
    recipient_hash: u64,
    /// Estimated market impact (fixed-point, scaled by 1e6)
    estimated_impact_fixed: u64,
    /// Risk score (0-10000, scaled by RISK_SCALE)
    risk_score: u32,
    /// Is processed
    is_processed: bool,
    /// Padding
    _padding: [u8; 35],
}

const _: () = assert!(core::mem::size_of::<UnlockEvent>() == 64);

/// Circular buffer for rolling unlock volume calculations
#[repr(C)]
struct UnlockVolumeBuffer {
    /// Pre-allocated buffer (24-hour windows)
    buffer: [u64; 168], // 168 hours = 1 week
    /// Head index
    head: AtomicU64,
    /// Current sum
    sum_fixed: AtomicU64,
    /// Window size
    window_size: usize,
    /// Padding
    _padding: [u8; CACHE_LINE_SIZE - 20],
}

impl UnlockVolumeBuffer {
    const fn new(window_size: usize) -> Self {
        Self {
            buffer: [0u64; 168],
            head: AtomicU64::new(0),
            sum_fixed: AtomicU64::new(0),
            window_size,
            _padding: [0u8; CACHE_LINE_SIZE - 20],
        }
    }
    
    /// Add unlock volume to rolling window
    #[inline]
    pub fn push(&self, volume_fixed: u64) -> u64 {
        let head = self.head.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = head % self.window_size;
        
        let old_value = unsafe { *self.buffer.get_unchecked(idx) };
        let delta = if volume_fixed >= old_value {
            volume_fixed - old_value
        } else {
            0 // Saturating subtraction
        };
        
        let new_sum = self.sum_fixed.fetch_add(delta, Ordering::Relaxed) + delta;
        
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = volume_fixed;
        }
        
        new_sum
    }
    
    #[inline]
    pub fn get_sum(&self) -> u64 {
        self.sum_fixed.load(Ordering::Relaxed)
    }
    
    #[inline]
    pub fn get_average(&self) -> u64 {
        let sum = self.get_sum();
        let count = core::cmp::min(self.head.load(Ordering::Relaxed) as usize, self.window_size);
        if count == 0 { return 0; }
        sum / count as u64
    }
}

/// Pre-event risk scaling parameters
#[repr(C)]
struct RiskScalingParams {
    /// Base risk multiplier (scaled by RISK_SCALE)
    base_multiplier: u64,
    /// Hours-before-event threshold for increased risk
    threshold_hours: u32,
    /// Max risk multiplier near event (scaled)
    max_multiplier: u64,
    /// Padding
    _padding: [u8; CACHE_LINE_SIZE - 20],
}

impl RiskScalingParams {
    const fn new() -> Self {
        Self {
            base_multiplier: 10_000, // 1.0x
            threshold_hours: 24,
            max_multiplier: 50_000, // 5.0x
            _padding: [0u8; CACHE_LINE_SIZE - 20],
        }
    }
}

/// Main token unlocks tracker
#[repr(C)]
pub struct TokenUnlocksTracker {
    /// Vesting schedules (pre-allocated)
    schedules: [VestingSchedule; MAX_VESTING_SCHEDULES],
    /// Upcoming unlock events (sorted by timestamp)
    upcoming_events: [UnlockEvent; MAX_UPCOMING_UNLOCKS],
    /// Active schedule count
    active_schedules: AtomicU64,
    /// Upcoming event count
    upcoming_count: AtomicU64,
    /// Rolling volume buffer (24h windows)
    volume_rolling: UnlockVolumeBuffer,
    /// Total unlocked volume (all-time, fixed-point)
    total_unlocked_fixed: PaddedAtomicU64,
    /// Next unlock timestamp
    next_unlock_timestamp: AtomicU64,
    /// Risk scaling parameters
    risk_params: RiskScalingParams,
    /// Circuit breaker for extreme unlock events
    circuit_breaker: AtomicBool,
    /// Max single unlock threshold (fixed-point)
    max_unlock_threshold_fixed: AtomicU64,
}

impl TokenUnlocksTracker {
    /// Create a new token unlocks tracker
    pub const fn new() -> Self {
        Self {
            schedules: [VestingSchedule {
                schedule_id: 0,
                token_id: 0,
                total_locked_fixed: 0,
                unlocked_fixed: 0,
                start_timestamp: 0,
                end_timestamp: 0,
                cliff_timestamp: 0,
                unlock_interval_sec: 0,
                amount_per_unlock_fixed: 0,
                total_unlocks: 0,
                completed_unlocks: 0,
                is_active: false,
                is_linear: false,
                _padding: [0u8; 34],
            }; MAX_VESTING_SCHEDULES],
            upcoming_events: [UnlockEvent {
                event_id: 0,
                schedule_id: 0,
                token_id: 0,
                unlock_amount_fixed: 0,
                timestamp: 0,
                recipient_hash: 0,
                estimated_impact_fixed: 0,
                risk_score: 0,
                is_processed: false,
                _padding: [0u8; 35],
            }; MAX_UPCOMING_UNLOCKS],
            active_schedules: AtomicU64::new(0),
            upcoming_count: AtomicU64::new(0),
            volume_rolling: UnlockVolumeBuffer::new(168), // 1 week of hourly data
            total_unlocked_fixed: PaddedAtomicU64::new(0),
            next_unlock_timestamp: AtomicU64::new(u64::MAX),
            risk_params: RiskScalingParams::new(),
            circuit_breaker: AtomicBool::new(false),
            max_unlock_threshold_fixed: AtomicU64::new(10_000_000_000_000), // 10M default
        }
    }
    
    /// Register a new vesting schedule
    #[inline]
    pub fn register_schedule(&self, schedule: VestingSchedule) -> bool {
        let idx = self.active_schedules.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= MAX_VESTING_SCHEDULES {
            self.active_schedules.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            *self.schedules.get_unchecked_mut(idx) = schedule;
        }
        
        // Calculate and register upcoming unlocks
        self.calculate_upcoming_unlocks(idx as u64);
        
        true
    }
    
    /// Calculate upcoming unlocks for a schedule
    #[inline]
    fn calculate_upcoming_unlocks(&self, schedule_idx: u64) {
        unsafe {
            let schedule = self.schedules.get_unchecked(schedule_idx as usize);
            
            if !schedule.is_active || schedule.unlock_interval_sec == 0 {
                return;
            }
            
            let current_time = get_current_timestamp();
            let mut next_unlock_time = schedule.start_timestamp.max(schedule.cliff_timestamp);
            
            // Find first future unlock
            while next_unlock_time < current_time && next_unlock_time < schedule.end_timestamp {
                next_unlock_time += schedule.unlock_interval_sec as u64;
            }
            
            // Register upcoming events
            let event_idx = self.upcoming_count.load(Ordering::Relaxed) as usize;
            if event_idx < MAX_UPCOMING_UNLOCKS && next_unlock_time < schedule.end_timestamp {
                let event = UnlockEvent {
                    event_id: event_idx as u64,
                    schedule_id: schedule.schedule_id,
                    token_id: schedule.token_id,
                    unlock_amount_fixed: schedule.amount_per_unlock_fixed,
                    timestamp: next_unlock_time,
                    recipient_hash: schedule.schedule_id, // Simplified
                    estimated_impact_fixed: self.estimate_market_impact(schedule.amount_per_unlock_fixed),
                    risk_score: self.calculate_risk_score(next_unlock_time, schedule.amount_per_unlock_fixed),
                    is_processed: false,
                    _padding: [0u8; 35],
                };
                
                *self.upcoming_events.get_unchecked_mut(event_idx) = event;
                self.upcoming_count.fetch_add(1, Ordering::Relaxed);
                
                // Update next unlock timestamp
                let current_next = self.next_unlock_timestamp.load(Ordering::Relaxed);
                if next_unlock_time < current_next {
                    self.next_unlock_timestamp.store(next_unlock_time, Ordering::Relaxed);
                }
            }
        }
    }
    
    /// Estimate market impact based on unlock size (fixed-point)
    #[inline]
    fn estimate_market_impact(&self, amount_fixed: u64) -> u64 {
        // Simple linear model: impact = amount / 1M * 0.01 (1% per 1M)
        // All in fixed-point arithmetic
        let impact_bps = (amount_fixed / 1_000_000_000_000) * 100; // Basis points
        impact_bps * 100 // Scale to match FIXED_SCALE
    }
    
    /// Calculate risk score for an upcoming unlock
    #[inline]
    fn calculate_risk_score(&self, unlock_timestamp: u64, amount_fixed: u64) -> u32 {
        let current_time = get_current_timestamp();
        let hours_until_unlock = if unlock_timestamp > current_time {
            (unlock_timestamp - current_time) / 3600
        } else {
            0
        };
        
        // Base risk from amount
        let base_risk = ((amount_fixed / FIXED_SCALE) as u32).min(1000);
        
        // Time-based risk multiplier
        let threshold_sec = self.risk_params.threshold_hours as u64 * 3600;
        let time_multiplier = if hours_until_unlock < self.risk_params.threshold_hours as u64 {
            // Linear interpolation between max and base multiplier
            let ratio = hours_until_unlock as u64 * RISK_SCALE / threshold_sec;
            self.risk_params.base_multiplier + 
                (self.risk_params.max_multiplier - self.risk_params.base_multiplier) * (RISK_SCALE - ratio) / RISK_SCALE
        } else {
            self.risk_params.base_multiplier
        };
        
        // Combined risk score
        ((base_risk as u64 * time_multiplier / RISK_SCALE) as u32).min(10000)
    }
    
    /// Process an unlock event (called when unlock occurs)
    #[inline]
    pub fn process_unlock(&self, event_id: u64) -> bool {
        // Find and mark event as processed
        let count = self.upcoming_count.load(Ordering::Relaxed) as usize;
        for i in 0..count {
            unsafe {
                let event = self.upcoming_events.get_unchecked(i);
                if event.event_id == event_id && !event.is_processed {
                    // Check circuit breaker
                    if event.unlock_amount_fixed > self.max_unlock_threshold_fixed.load(Ordering::Relaxed) {
                        if self.circuit_breaker.load(Ordering::Relaxed) {
                            return false;
                        }
                    }
                    
                    // Mark as processed
                    (*self.upcoming_events.get_unchecked_mut(i)).is_processed = true;
                    
                    // Update totals
                    self.total_unlocked_fixed.fetch_add(event.unlock_amount_fixed);
                    self.volume_rolling.push(event.unlock_amount_fixed);
                    
                    // Update schedule
                    self.update_schedule_unlocked(event.schedule_id, event.unlock_amount_fixed);
                    
                    return true;
                }
            }
        }
        false
    }
    
    /// Update schedule's unlocked amount
    #[inline]
    fn update_schedule_unlocked(&self, schedule_id: u64, amount_fixed: u64) {
        for i in 0..MAX_VESTING_SCHEDULES {
            unsafe {
                let schedule = self.schedules.get_unchecked(i);
                if schedule.schedule_id == schedule_id {
                    (*self.schedules.get_unchecked_mut(i)).unlocked_fixed += amount_fixed;
                    (*self.schedules.get_unchecked_mut(i)).completed_unlocks += 1;
                    break;
                }
            }
        }
    }
    
    /// Get total upcoming unlock volume in next N hours
    #[inline]
    pub fn get_upcoming_volume(&self, hours: u32) -> u64 {
        let current_time = get_current_timestamp();
        let cutoff = current_time + (hours as u64 * 3600);
        let mut total = 0u64;
        
        let count = self.upcoming_count.load(Ordering::Relaxed) as usize;
        for i in 0..count {
            unsafe {
                let event = self.upcoming_events.get_unchecked(i);
                if !event.is_processed && event.timestamp <= cutoff {
                    total += event.unlock_amount_fixed;
                }
            }
        }
        
        total
    }
    
    /// Get average risk score for upcoming events
    #[inline]
    pub fn get_average_risk_score(&self) -> u32 {
        let count = self.upcoming_count.load(Ordering::Relaxed) as usize;
        if count == 0 { return 0; }
        
        let mut total_risk = 0u64;
        for i in 0..count {
            unsafe {
                let event = self.upcoming_events.get_unchecked(i);
                if !event.is_processed {
                    total_risk += event.risk_score as u64;
                }
            }
        }
        
        (total_risk / count as u64) as u32
    }
    
    /// Get rolling average unlock volume
    #[inline]
    pub fn get_rolling_avg_volume(&self) -> u64 {
        self.volume_rolling.get_average()
    }
    
    /// Enable circuit breaker
    #[inline]
    pub fn enable_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Relaxed);
    }
    
    /// Disable circuit breaker
    #[inline]
    pub fn disable_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
    }
    
    /// Set max unlock threshold
    #[inline]
    pub fn set_max_unlock_threshold(&self, threshold_fixed: u64) {
        self.max_unlock_threshold_fixed.store(threshold_fixed, Ordering::Relaxed);
    }
}

impl Default for TokenUnlocksTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current Unix timestamp (mock implementation - would use rdtsc in production)
#[inline]
fn get_current_timestamp() -> u64 {
    // In production: use rdtsc or system clock
    1700000000 // Mock timestamp
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_register_schedule() {
        let tracker = TokenUnlocksTracker::new();
        
        let schedule = VestingSchedule {
            schedule_id: 1,
            token_id: 1,
            total_locked_fixed: 100_000_000_000_000, // 100M
            unlocked_fixed: 0,
            start_timestamp: 1700000000,
            end_timestamp: 1731536000,
            cliff_timestamp: 1700000000,
            unlock_interval_sec: 86400, // Daily
            amount_per_unlock_fixed: 1_000_000_000_000, // 1M per day
            total_unlocks: 100,
            completed_unlocks: 0,
            is_active: true,
            is_linear: true,
            _padding: [0u8; 34],
        };
        
        assert!(tracker.register_schedule(schedule));
        assert_eq!(tracker.active_schedules.load(), 1);
    }
    
    #[test]
    fn test_upcoming_volume() {
        let tracker = TokenUnlocksTracker::new();
        
        // Manually add an upcoming event
        let event = UnlockEvent {
            event_id: 0,
            schedule_id: 1,
            token_id: 1,
            unlock_amount_fixed: 5_000_000_000_000, // 5M
            timestamp: 1700100000,
            recipient_hash: 0,
            estimated_impact_fixed: 50000,
            risk_score: 5000,
            is_processed: false,
            _padding: [0u8; 35],
        };
        
        unsafe {
            *tracker.upcoming_events.get_unchecked_mut(0) = event;
        }
        tracker.upcoming_count.store(1, Ordering::Relaxed);
        
        let volume = tracker.get_upcoming_volume(48); // Next 48 hours
        assert_eq!(volume, 5_000_000_000_000);
    }
    
    #[test]
    fn test_risk_scoring() {
        let tracker = TokenUnlocksTracker::new();
        
        // Near-term unlock (high risk)
        let risk_near = tracker.calculate_risk_score(1700010000, 10_000_000_000_000);
        
        // Far-term unlock (lower risk)
        let risk_far = tracker.calculate_risk_score(1700500000, 10_000_000_000_000);
        
        assert!(risk_near >= risk_far);
    }
}
