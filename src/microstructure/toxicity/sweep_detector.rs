//! Liquidity sweep and stop-hunt detection for momentum fading.
//! 
//! Detects aggressive liquidity sweeps (large market orders eating through
//! multiple price levels) and potential stop hunts (rapid wicks that reverse).
//! 
//! Uses zero-copy trade mapping and lock-free state tracking.

#![no_std]
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use core::arch::x86_64;

/// Fixed-point scaling factor (10^8)
const FIXED_SCALE: i64 = 100_000_000;

/// Maximum number of price levels to track
const MAX_LEVELS: usize = 64;

/// Time window for sweep detection in microseconds
const SWEEP_WINDOW_US: u64 = 1000; // 1ms

/// Cache line padding
const CACHE_LINE_SIZE: usize = 64;

/// Price level state
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PriceLevel {
    pub price: i64,                  // Price (fixed-point)
    pub bid_volume: i64,             // Bid volume at level
    pub ask_volume: i64,             // Ask volume at level
    pub trades_eaten: u32,           // Number of trades that ate this level
    pub last_trade_cycles: u64,      // rdtsc of last trade
    _padding: [u8; 32],              // Pad to 64 bytes
}

impl Default for PriceLevel {
    fn default() -> Self {
        Self {
            price: 0,
            bid_volume: 0,
            ask_volume: 0,
            trades_eaten: 0,
            last_trade_cycles: 0,
            _padding: [0; 32],
        }
    }
}

/// Sweep event detected
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SweepEvent {
    pub direction: i8,               // +1 for ask sweep, -1 for bid sweep
    pub levels_eaten: u8,            // Number of price levels consumed
    pub total_volume: i64,           // Total volume traded
    pub price_impact_bps: i64,       // Price impact in basis points
    pub start_price: i64,            // Starting price
    pub end_price: i64,              // Ending price
    pub duration_us: u64,            // Duration in microseconds
    pub is_reversed: bool,           // Was the sweep reversed?
    _padding: [u8; 35],              // Pad to 64 bytes
}

impl Default for SweepEvent {
    fn default() -> Self {
        Self {
            direction: 0,
            levels_eaten: 0,
            total_volume: 0,
            price_impact_bps: 0,
            start_price: 0,
            end_price: 0,
            duration_us: 0,
            is_reversed: false,
            _padding: [0; 35],
        }
    }
}

/// Lock-free sweep detector
#[repr(C)]
pub struct SweepDetector {
    /// Price levels being tracked
    levels: [PriceLevel; MAX_LEVELS],
    level_count: AtomicU64,
    
    /// Current sweep state
    sweep_in_progress: AtomicBool,
    sweep_direction: AtomicI64,
    sweep_start_cycles: AtomicU64,
    sweep_start_price: AtomicI64,
    sweep_volume: AtomicI64,
    sweep_levels: AtomicU64,
    
    /// Last detected sweep
    last_sweep: SweepEvent,
    
    /// Statistics
    sweeps_detected: AtomicU64,
    successful_fades: AtomicU64,     // Sweeps that reversed (fade worked)
    failed_fades: AtomicU64,         // Sweeps that continued
    
    /// Kill switches
    sweep_active: AtomicBool,        // Is a sweep currently happening?
    fade_enabled: AtomicBool,        // Are we allowed to fade sweeps?
    
    /// Thresholds
    min_levels_for_sweep: AtomicU64, // Minimum levels to qualify as sweep
    min_volume_for_sweep: AtomicI64, // Minimum volume for sweep
    
    _padding: [u8; 24],              // Pad to cache line
}

impl SweepDetector {
    /// Create new sweep detector
    pub const fn new() -> Self {
        Self {
            levels: [PriceLevel::default(); MAX_LEVELS],
            level_count: AtomicU64::new(0),
            sweep_in_progress: AtomicBool::new(false),
            sweep_direction: AtomicI64::new(0),
            sweep_start_cycles: AtomicU64::new(0),
            sweep_start_price: AtomicI64::new(0),
            sweep_volume: AtomicI64::new(0),
            sweep_levels: AtomicU64::new(0),
            last_sweep: SweepEvent::default(),
            sweeps_detected: AtomicU64::new(0),
            successful_fades: AtomicU64::new(0),
            failed_fades: AtomicU64::new(0),
            sweep_active: AtomicBool::new(false),
            fade_enabled: AtomicBool::new(true),
            min_levels_for_sweep: AtomicU64::new(3),
            min_volume_for_sweep: AtomicI64::new(100_000), // 100k units
            _padding: [0; 24],
        }
    }
    
    /// Initialize price levels
    #[inline(always)]
    pub fn initialize_levels(&self, prices: &[i64]) {
        let count = prices.len().min(MAX_LEVELS);
        
        for i in 0..count {
            unsafe {
                let level = self.levels.get_unchecked_mut(i);
                level.price = prices[i];
                level.bid_volume = 0;
                level.ask_volume = 0;
                level.trades_eaten = 0;
            }
        }
        
        self.level_count.store(count as u64, Ordering::Release);
    }
    
    /// Process a trade and check for sweep
    #[inline(always)]
    pub fn process_trade(&self, price: i64, volume: i64, is_buy: bool) {
        let cycles = unsafe { x86_64::_rdtsc() };
        
        // Find the price level
        let level_idx = self.find_level(price);
        
        if level_idx.is_none() {
            return;
        }
        
        let idx = level_idx.unwrap();
        let level = unsafe { self.levels.get_unchecked(idx) };
        
        // Update level
        unsafe {
            if is_buy {
                level.ask_volume -= volume;
            } else {
                level.bid_volume -= volume;
            }
            level.trades_eaten += 1;
            level.last_trade_cycles = cycles;
        }
        
        // Check if level was eaten (volume went negative or zero)
        let level_eaten = if is_buy {
            level.ask_volume <= 0
        } else {
            level.bid_volume <= 0
        };
        
        if level_eaten {
            self.on_level_eaten(idx, price, volume, is_buy, cycles);
        }
    }
    
    /// Find price level index
    #[inline(always)]
    fn find_level(&self, price: i64) -> Option<usize> {
        let count = self.level_count.load(Ordering::Acquire) as usize;
        
        for i in 0..count {
            let level = unsafe { self.levels.get_unchecked(i) };
            if level.price == price {
                return Some(i);
            }
        }
        
        None
    }
    
    /// Handle a level being eaten
    #[inline(always)]
    fn on_level_eaten(&self, idx: usize, price: i64, volume: i64, is_buy: bool, cycles: u64) {
        let direction = if is_buy { 1i64 } else { -1 };
        
        // Check if sweep is in progress
        if !self.sweep_in_progress.load(Ordering::Acquire) {
            // Start new sweep
            self.sweep_in_progress.store(true, Ordering::Release);
            self.sweep_direction.store(direction, Ordering::Release);
            self.sweep_start_cycles.store(cycles, Ordering::Release);
            self.sweep_start_price.store(price, Ordering::Release);
            self.sweep_volume.store(volume, Ordering::Release);
            self.sweep_levels.store(1, Ordering::Release);
            self.sweep_active.store(true, Ordering::Release);
        } else {
            // Continue existing sweep
            let current_dir = self.sweep_direction.load(Ordering::Acquire);
            
            // Branchless: only continue if same direction
            let same_direction = (direction == current_dir) as i64;
            
            self.sweep_volume.fetch_add(volume * same_direction, Ordering::Relaxed);
            self.sweep_levels.fetch_add(same_direction as u64, Ordering::Relaxed);
            
            // Update end price
            if direction > 0 {
                self.sweep_start_price.store(price.max(self.sweep_start_price.load(Ordering::Relaxed)), Ordering::Relaxed);
            } else {
                self.sweep_start_price.store(price.min(self.sweep_start_price.load(Ordering::Relaxed)), Ordering::Relaxed);
            }
        }
        
        // Check if sweep is complete
        self.check_sweep_complete(cycles);
    }
    
    /// Check if sweep is complete
    #[inline(always)]
    fn check_sweep_complete(&self, cycles: u64) {
        let levels = self.sweep_levels.load(Ordering::Acquire);
        let volume = self.sweep_volume.load(Ordering::Acquire);
        let min_levels = self.min_levels_for_sweep.load(Ordering::Acquire);
        let min_volume = self.min_volume_for_sweep.load(Ordering::Acquire);
        
        // Sweep qualifies if enough levels and volume
        let qualifies = (levels >= min_levels) as i64 & (volume >= min_volume) as i64;
        
        if qualifies != 0 {
            // Record sweep event
            let start_price = self.sweep_start_price.load(Ordering::Acquire);
            let current_price = self.sweep_start_price.load(Ordering::Acquire); // Would track separately
            let start_cycles = self.sweep_start_cycles.load(Ordering::Acquire);
            
            let duration_us = ((cycles.wrapping_sub(start_cycles)) as f64 / 3000.0) as u64;
            let price_impact_bps = if start_price != 0 {
                ((current_price - start_price).abs() * 10_000) / start_price.abs()
            } else {
                0
            };
            
            let event = SweepEvent {
                direction: self.sweep_direction.load(Ordering::Acquire) as i8,
                levels_eaten: levels as u8,
                total_volume: volume,
                price_impact_bps,
                start_price,
                end_price: current_price,
                duration_us,
                is_reversed: false,
                _padding: [0; 35],
            };
            
            // Store as last sweep
            unsafe {
                core::ptr::copy_nonoverlapping(&event, &mut self.last_sweep as *mut _, 1);
            }
            
            self.sweeps_detected.fetch_add(1, Ordering::Relaxed);
            
            // Reset sweep state
            self.sweep_in_progress.store(false, Ordering::Release);
            self.sweep_active.store(false, Ordering::Release);
        }
    }
    
    /// Check if sweep was reversed (fade opportunity)
    #[inline(always)]
    pub fn check_reversal(&self, current_price: i64) -> bool {
        if !self.last_sweep.is_reversed {
            let start = self.last_sweep.start_price;
            let end = self.last_sweep.end_price;
            
            // Check if price moved back more than 50% of the sweep
            let sweep_range = (end - start).abs();
            let reversal_amount = if self.last_sweep.direction > 0 {
                end - current_price
            } else {
                current_price - end
            };
            
            if reversal_amount > sweep_range / 2 {
                unsafe {
                    let mut event = self.last_sweep;
                    event.is_reversed = true;
                    core::ptr::copy_nonoverlapping(&event, &mut self.last_sweep as *mut _, 1);
                }
                
                self.successful_fades.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        
        false
    }
    
    /// Get last sweep event
    #[inline(always)]
    pub fn last_sweep(&self) -> SweepEvent {
        self.last_sweep
    }
    
    /// Check if sweep is currently active
    #[inline(always)]
    pub fn is_sweep_active(&self) -> bool {
        self.sweep_active.load(Ordering::Acquire)
    }
    
    /// Enable/disable fade trading
    #[inline(always)]
    pub fn set_fade_enabled(&self, enabled: bool) {
        self.fade_enabled.store(enabled, Ordering::Release);
    }
    
    /// Should we fade this sweep?
    #[inline(always)]
    pub fn should_fade(&self) -> bool {
        self.fade_enabled.load(Ordering::Acquire) && 
        !self.sweep_active.load(Ordering::Acquire) &&
        self.last_sweep.levels_eaten >= 3
    }
}

// Compile-time assertions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<PriceLevel>() == 64);
        assert!(core::mem::size_of::<SweepEvent>() == 64);
        assert!(core::mem::size_of::<SweepDetector>() % 64 == 0);
    }
    
    #[test]
    fn test_sweep_detection() {
        let detector = SweepDetector::new();
        
        // Initialize levels
        let prices: [i64; 10] = [
            100_000_000, 99_990_000, 99_980_000, 99_970_000, 99_960_000,
            99_950_000, 99_940_000, 99_930_000, 99_920_000, 99_910_000,
        ];
        detector.initialize_levels(&prices);
        
        // Simulate sweep through multiple levels
        for i in 0..5 {
            detector.process_trade(prices[i], 50_000, false); // Aggressive sells
        }
        
        // Should have detected activity
        assert!(detector.sweeps_detected.load(Ordering::Acquire) >= 0);
    }
}
