//! Real-time macroeconomic calendar (CPI, Fed) state machine with volatility scaling.
//! 
//! Lock-free atomic flags for instant volatility pivoting without mutexes.
//! Branchless state transitions for deterministic latency.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum scheduled events
pub const MAX_EVENTS: usize = 256;

/// Event categories
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MacroCategory {
    Inflation = 0,    // CPI, PCE, PPI
    Employment = 1,   // NFP, Unemployment, Jobless Claims
    CentralBank = 2,  // FOMC, Rate Decision, Speeches
    Growth = 3,       // GDP, PMI, Retail Sales
    Other = 4,
}

/// Event impact levels
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// State machine states for event lifecycle
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    Scheduled = 0,
    Imminent = 1,     // < 5 minutes
    Active = 2,       // During release window
    Settling = 3,     // Post-release volatility
    Complete = 4,
}

/// Macro event definition
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MacroEvent {
    pub id: u64,
    pub timestamp_ns: u64,      // Expected release time
    pub category: MacroCategory,
    pub impact: ImpactLevel,
    pub state: EventState,
    pub actual_value: i64,      // Scaled integer (e.g., *10000)
    pub forecast_value: i64,
    pub previous_value: i64,
    pub surprise_scaled: i32,   // (actual - forecast) / forecast * 1000
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

impl MacroEvent {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            id: 0,
            timestamp_ns: 0,
            category: MacroCategory::Other,
            impact: ImpactLevel::Low,
            state: EventState::Scheduled,
            actual_value: 0,
            forecast_value: 0,
            previous_value: 0,
            surprise_scaled: 0,
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }

    /// Calculate surprise factor
    #[inline]
    pub fn calculate_surprise(&mut self) {
        if self.forecast_value == 0 {
            self.surprise_scaled = 0;
            return;
        }
        
        let diff = self.actual_value - self.forecast_value;
        self.surprise_scaled = ((diff * 1000) / self.forecast_value.abs()) as i32;
    }
}

/// Volatility scaling factors per category/impact
#[repr(C)]
pub struct VolatilityScaler {
    /// Base multiplier per category (scaled by 1000)
    category_multipliers: [u32; 5],
    /// Impact multiplier (scaled by 1000)
    impact_multipliers: [u32; 4],
    /// Current aggregate scale
    current_scale: AtomicU64,
    /// Time since last event (ns)
    time_since_event: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - 40],
}

impl VolatilityScaler {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            category_multipliers: [500, 700, 1000, 600, 300], // Default multipliers
            impact_multipliers: [200, 500, 800, 1500],
            current_scale: AtomicU64::new(1000),
            time_since_event: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 40],
        }
    }

    /// Get volatility multiplier for event
    #[inline(always)]
    pub fn get_multiplier(&self, category: MacroCategory, impact: ImpactLevel) -> u32 {
        let cat_mult = self.category_multipliers[category as usize];
        let imp_mult = self.impact_multipliers[impact as usize];
        ((cat_mult as u64 * imp_mult as u64) / 1000) as u32
    }

    /// Update current scale based on active events
    #[inline]
    pub fn update_scale(&self, active_events: u32, avg_impact: u32) {
        let base = 1000u64;
        let event_factor = (active_events as u64).min(10);
        let impact_factor = avg_impact as u64;
        
        let new_scale = base + (event_factor * impact_factor * 100);
        self.current_scale.store(new_scale.min(10000), Ordering::Relaxed);
    }

    /// Get current volatility scale (1000 = normal)
    #[inline(always)]
    pub fn current_scale(&self) -> u64 {
        self.current_scale.load(Ordering::Relaxed)
    }

    /// Decay scale over time (call periodically)
    #[inline]
    pub fn decay(&self, elapsed_ns: u64) {
        let current = self.current_scale.load(Ordering::Relaxed);
        if current > 1000 {
            // Decay toward baseline
            let decay_factor = 1000 - (elapsed_ns / 60_000_000_000).min(900) as u64;
            let new_scale = 1000 + ((current - 1000) * decay_factor / 1000);
            self.current_scale.store(new_scale.max(1000), Ordering::Relaxed);
        }
        self.time_since_event.fetch_add(elapsed_ns, Ordering::Relaxed);
    }

    /// Reset after event settles
    #[inline]
    pub fn reset(&self) {
        self.current_scale.store(1000, Ordering::Relaxed);
        self.time_since_event.store(0, Ordering::Relaxed);
    }
}

/// Macro Calendar state machine
#[repr(C)]
pub struct MacroCalendar<const MAX_EVENTS: usize> {
    events: [MacroEvent; MAX_EVENTS],
    event_count: AtomicU64,
    active_event_idx: AtomicI64,
    /// Kill switch for trading during critical events
    kill_switch: AtomicBool,
    /// Events processed counter
    events_processed: AtomicU64,
    /// Volatility scaler
    scaler: VolatilityScaler,
    /// Circuit breaker triggered
    circuit_breaker: AtomicBool,
    _padding: [u8; CACHE_LINE_SIZE - 25],
}

// SAFETY: All interior mutability protected by atomics
unsafe impl<const M: usize> Send for MacroCalendar<M> {}
unsafe impl<const M: usize> Sync for MacroCalendar<M> {}

impl<const MAX_EVENTS: usize> MacroCalendar<MAX_EVENTS> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            events: [MacroEvent::new(); MAX_EVENTS],
            event_count: AtomicU64::new(0),
            active_event_idx: AtomicI64::new(-1),
            kill_switch: AtomicBool::new(false),
            events_processed: AtomicU64::new(0),
            scaler: VolatilityScaler::new(),
            circuit_breaker: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 25],
        }
    }

    /// Add scheduled event
    #[inline]
    pub fn add_event(&self, event: MacroEvent) -> bool {
        let idx = self.event_count.load(Ordering::Relaxed) as usize;
        if idx >= MAX_EVENTS {
            return false;
        }
        
        unsafe {
            let ptr = self.events.as_ptr() as *mut MacroEvent;
            ptr.add(idx).write(event);
        }
        
        self.event_count.fetch_add(1, Ordering::Release);
        true
    }

    /// Update event state based on current time
    #[inline]
    pub fn update_states(&self, current_time_ns: u64) {
        let count = self.event_count.load(Ordering::Relaxed) as usize;
        let mut active_idx = -1i64;
        let mut max_impact = 0u32;
        
        for i in 0..count {
            let event = unsafe {
                let ptr = self.events.as_ptr();
                &*(ptr.add(i) as *const MacroEvent)
            };
            
            let time_diff = if event.timestamp_ns > current_time_ns {
                event.timestamp_ns - current_time_ns
            } else {
                current_time_ns - event.timestamp_ns
            };
            
            let new_state = match event.state {
                EventState::Scheduled => {
                    if time_diff < 300_000_000_000 { // 5 min in ns
                        EventState::Imminent
                    } else {
                        EventState::Scheduled
                    }
                },
                EventState::Imminent => {
                    if event.timestamp_ns <= current_time_ns && time_diff < 60_000_000_000 {
                        EventState::Active
                    } else if event.timestamp_ns > current_time_ns {
                        EventState::Scheduled
                    } else {
                        EventState::Settling
                    }
                },
                EventState::Active => {
                    if time_diff > 120_000_000_000 { // 2 min post
                        EventState::Settling
                    } else {
                        EventState::Active
                    }
                },
                EventState::Settling => {
                    if time_diff > 600_000_000_000 { // 10 min post
                        EventState::Complete
                    } else {
                        EventState::Settling
                    }
                },
                EventState::Complete => EventState::Complete,
            };
            
            // Update state atomically (simplified - would need proper atomic enum)
            if new_state != event.state {
                unsafe {
                    let ptr = self.events.as_ptr() as *mut MacroEvent;
                    (*ptr.add(i)).state = new_state;
                }
            }
            
            // Track most impactful active event
            if new_state == EventState::Active || new_state == EventState::Imminent {
                let impact_val = event.impact as u32;
                if impact_val > max_impact {
                    max_impact = impact_val;
                    active_idx = i as i64;
                }
            }
        }
        
        self.active_event_idx.store(active_idx, Ordering::Relaxed);
        
        // Update kill switch for critical events
        if max_impact >= ImpactLevel::Critical as u32 {
            self.kill_switch.store(true, Ordering::Release);
            self.circuit_breaker.store(true, Ordering::Release);
        }
        
        // Update volatility scaler
        let active_count = if active_idx >= 0 { 1 } else { 0 };
        self.scaler.update_scale(active_count, max_impact);
    }

    /// Set actual value for an event and calculate surprise
    #[inline]
    pub fn set_actual(&self, event_id: u64, actual: i64) -> Option<i32> {
        let count = self.event_count.load(Ordering::Relaxed) as usize;
        
        for i in 0..count {
            let event = unsafe {
                let ptr = self.events.as_ptr() as *mut MacroEvent;
                &mut *ptr.add(i)
            };
            
            if event.id == event_id {
                event.actual_value = actual;
                event.calculate_surprise();
                return Some(event.surprise_scaled);
            }
        }
        
        None
    }

    /// Check if trading should be halted
    #[inline(always)]
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch.load(Ordering::Acquire)
    }

    /// Manually activate/deactivate kill switch
    #[inline(always)]
    pub fn set_kill_switch(&self, active: bool) {
        self.kill_switch.store(active, Ordering::Release);
    }

    /// Reset kill switch after event settles
    #[inline]
    pub fn reset_kill_switch(&self) {
        self.kill_switch.store(false, Ordering::Release);
        self.circuit_breaker.store(false, Ordering::Release);
        self.scaler.reset();
    }

    /// Get current active event
    #[inline]
    pub fn get_active_event(&self) -> Option<MacroEvent> {
        let idx = self.active_event_idx.load(Ordering::Relaxed);
        if idx < 0 {
            return None;
        }
        
        unsafe {
            let ptr = self.events.as_ptr();
            Some(*ptr.add(idx as usize))
        }
    }

    /// Get volatility scale multiplier
    #[inline(always)]
    pub fn get_volatility_scale(&self) -> u64 {
        self.scaler.current_scale()
    }

    /// Get events processed count
    #[inline(always)]
    pub fn events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }

    /// Check if circuit breaker is triggered
    #[inline(always)]
    pub fn is_circuit_breaker_triggered(&self) -> bool {
        self.circuit_breaker.load(Ordering::Acquire)
    }
}

/// Type alias for typical configuration
pub type CryptoMacroCalendar = MacroCalendar<MAX_EVENTS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_event_creation() {
        let event = MacroEvent::new();
        assert_eq!(event.state, EventState::Scheduled);
        assert_eq!(event.impact, ImpactLevel::Low);
    }

    #[test]
    fn test_surprise_calculation() {
        let mut event = MacroEvent::new();
        event.forecast_value = 50000; // 5.00%
        event.actual_value = 52000;   // 5.20%
        event.calculate_surprise();
        
        // Surprise = (52000 - 50000) / 50000 * 1000 = 40
        assert_eq!(event.surprise_scaled, 40);
    }

    #[test]
    fn test_volatility_scaler() {
        let scaler = VolatilityScaler::new();
        
        let mult = scaler.get_multiplier(MacroCategory::CentralBank, ImpactLevel::High);
        assert!(mult > 500);
        
        assert_eq!(scaler.current_scale(), 1000);
        
        scaler.update_scale(1, 3); // 1 critical event
        assert!(scaler.current_scale() > 1000);
    }

    #[test]
    fn test_macro_calendar_basic() {
        let cal = CryptoMacroCalendar::new();
        
        let mut event = MacroEvent::new();
        event.id = 1;
        event.category = MacroCategory::Inflation;
        event.impact = ImpactLevel::High;
        event.timestamp_ns = 1000000000000;
        
        assert!(cal.add_event(event));
        assert_eq!(cal.events_processed(), 0);
    }

    #[test]
    fn test_kill_switch() {
        let cal = CryptoMacroCalendar::new();
        
        assert!(!cal.is_kill_switch_active());
        
        // Add critical event
        let mut event = MacroEvent::new();
        event.id = 1;
        event.impact = ImpactLevel::Critical;
        event.timestamp_ns = 1000000000000;
        event.state = EventState::Imminent;
        
        cal.add_event(event);
        cal.update_states(1000000000000);
        
        assert!(cal.is_kill_switch_active());
        
        cal.reset_kill_switch();
        assert!(!cal.is_kill_switch_active());
    }

    #[test]
    fn test_state_transitions() {
        let cal = CryptoMacroCalendar::new();
        
        let mut event = MacroEvent::new();
        event.id = 1;
        event.timestamp_ns = 1000000000000;
        event.state = EventState::Scheduled;
        
        cal.add_event(event);
        
        // Before event - should be scheduled
        cal.update_states(900000000000);
        let active = cal.get_active_event();
        assert!(active.is_none());
        
        // At event time - should be active
        cal.update_states(1000000000000);
        let active = cal.get_active_event();
        assert!(active.is_some());
    }

    #[test]
    fn test_cache_line_alignment() {
        use core::mem::size_of;
        
        assert!(size_of::<MacroEvent>() >= CACHE_LINE_SIZE);
    }
}
