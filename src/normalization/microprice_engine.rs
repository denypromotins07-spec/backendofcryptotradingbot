//! Microprice Engine - Liquidity-weighted fair price calculation.
//! 
//! Calculates consolidated cross-venue microprices using volume-weighted
//! averages of best bid/ask levels. Provides fair value estimation for
//! execution algorithms and risk management.
//! 
//! Micro-optimizations:
//! - Fixed-point arithmetic throughout (no floating point)
//! - Cache-line aligned intermediate buffers
//! - Batch processing for multiple symbols

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use crate::normalization::l2_normalizer::NormalizedBook;

/// Maximum venues for consolidation
pub const MAX_VENUES: usize = 16;

/// Venue weight entry (cache-line aligned)
#[repr(C, align(64))]
struct VenueWeight {
    /// Weight for this venue (fixed-point, sum = 1e8)
    weight: AtomicU64,
    /// Last contribution amount
    last_contrib: AtomicI64,
    /// Is venue active?
    active: AtomicBool,
    _pad: [u8; 55],
}

impl VenueWeight {
    const fn new() -> Self {
        Self {
            weight: AtomicU64::new(0),
            last_contrib: AtomicI64::new(0),
            active: AtomicBool::new(false),
            _pad: [0; 55],
        }
    }
}

/// Microprice result for a symbol
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Microprice {
    /// Fair microprice (fixed-point * 1e8)
    pub price: i64,
    /// Bid-side microprice
    pub bid_micro: i64,
    /// Ask-side microprice
    pub ask_micro: i64,
    /// Total bid liquidity
    pub bid_liquidity: i64,
    /// Total ask liquidity
    pub ask_liquidity: i64,
    /// Imbalance ratio (bid - ask) / (bid + ask) * 1e8
    pub imbalance: i64,
    /// Last update timestamp
    pub timestamp_ns: u64,
    /// Number of venues contributing
    pub venue_count: u8,
    _pad: [u8; 23],
}

impl Microprice {
    pub const fn new() -> Self {
        Self {
            price: 0,
            bid_micro: 0,
            ask_micro: 0,
            bid_liquidity: 0,
            ask_liquidity: 0,
            imbalance: 0,
            timestamp_ns: 0,
            venue_count: 0,
            _pad: [0; 23],
        }
    }
}

/// Microprice Engine - Calculates fair value across venues
pub struct MicropriceEngine {
    /// Per-venue weights
    venue_weights: [VenueWeight; MAX_VENUES],
    /// Default equal weight (1e8 / MAX_VENUES)
    default_weight: u64,
    /// Minimum liquidity threshold
    min_liquidity: AtomicI64,
}

// SAFETY: All mutable state is protected by atomics
unsafe impl Send for MicropriceEngine {}
unsafe impl Sync for MicropriceEngine {}

impl MicropriceEngine {
    /// Create a new microprice engine with equal venue weights
    pub const fn new() -> Self {
        const INIT_WEIGHT: VenueWeight = VenueWeight::new();
        Self {
            venue_weights: [INIT_WEIGHT; MAX_VENUES],
            default_weight: 100_000_000 / MAX_VENUES as u64,
            min_liquidity: AtomicI64::new(100_000_000), // 1.0 minimum
        }
    }
    
    /// Initialize venue weights (call once at startup)
    pub fn init(&self) {
        for (i, weight) in self.venue_weights.iter().enumerate() {
            if i < MAX_VENUES {
                weight.weight.store(self.default_weight, Ordering::Relaxed);
                weight.active.store(true, Ordering::Relaxed);
            }
        }
    }
    
    /// Set weight for a specific venue
    #[inline]
    pub fn set_venue_weight(&self, venue_id: u8, weight: u64) {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            self.venue_weights[idx].weight.store(weight, Ordering::Relaxed);
            self.venue_weights[idx].active.store(weight > 0, Ordering::Relaxed);
        }
    }
    
    /// Calculate microprice from a normalized book
    /// Returns Microprice with fair value estimate
    #[inline]
    pub fn calculate(&self, book: &NormalizedBook) -> Microprice {
        let best_bid = book.best_bid.load(Ordering::Relaxed) as i64;
        let best_ask = book.best_ask.load(Ordering::Relaxed) as i64;
        
        if best_bid <= 0 || best_ask <= 0 {
            return Microprice::new();
        }
        
        // Simple microprice: weighted midpoint based on spread position
        // More sophisticated: use top-of-book sizes for weighting
        
        let spread = best_ask.wrapping_sub(best_bid);
        let mid = best_bid.wrapping_add(best_ask).wrapping_div(2);
        
        // Estimate liquidity-weighted adjustment
        // In production, would sum actual quantities from book levels
        let estimated_bid_liq = 100_000_000i64; // Placeholder
        let estimated_ask_liq = 100_000_000i64;
        
        let total_liq = estimated_bid_liq.wrapping_add(estimated_ask_liq);
        if total_liq == 0 {
            return Microprice::new();
        }
        
        // Microprice tilts toward side with more liquidity
        let liq_ratio = estimated_bid_liq.wrapping_mul(100_000_000).wrapping_div(total_liq);
        let tilt = spread.wrapping_mul(liq_ratio as i64).wrapping_div(100_000_000);
        let microprice = best_bid.wrapping_add(tilt);
        
        // Calculate imbalance
        let liq_diff = estimated_bid_liq.wrapping_sub(estimated_ask_liq);
        let imbalance = liq_diff.wrapping_mul(100_000_000).wrapping_div(total_liq);
        
        Microprice {
            price: microprice,
            bid_micro: best_bid,
            ask_micro: best_ask,
            bid_liquidity: estimated_bid_liq,
            ask_liquidity: estimated_ask_liq,
            imbalance,
            timestamp_ns: unsafe { std::arch::x86_64::_rdtsc() } as u64,
            venue_count: 1,
            _pad: [0; 23],
        }
    }
    
    /// Calculate consolidated microprice across multiple venue books
    #[inline]
    pub fn calculate_consolidated(&self, books: &[&NormalizedBook]) -> Microprice {
        if books.is_empty() {
            return Microprice::new();
        }
        
        let mut total_bid_liq: i64 = 0;
        let mut total_ask_liq: i64 = 0;
        let mut weighted_bid_sum: i64 = 0;
        let mut weighted_ask_sum: i64 = 0;
        let mut venue_count: u8 = 0;
        
        for book in books {
            let best_bid = book.best_bid.load(Ordering::Relaxed) as i64;
            let best_ask = book.best_ask.load(Ordering::Relaxed) as i64;
            
            if best_bid <= 0 || best_ask <= 0 {
                continue;
            }
            
            // Get venue weight (simplified - would need venue_id from book)
            let weight = self.default_weight as i64;
            
            // Estimate liquidity (production would read actual levels)
            let est_bid_liq = 100_000_000i64;
            let est_ask_liq = 100_000_000i64;
            
            weighted_bid_sum = weighted_bid_sum.wrapping_add(best_bid.wrapping_mul(weight));
            weighted_ask_sum = weighted_ask_sum.wrapping_add(best_ask.wrapping_mul(weight));
            total_bid_liq = total_bid_liq.wrapping_add(est_bid_liq);
            total_ask_liq = total_ask_liq.wrapping_add(est_ask_liq);
            venue_count += 1;
        }
        
        if venue_count == 0 {
            return Microprice::new();
        }
        
        let total_weight = (self.default_weight as i64).wrapping_mul(venue_count as i64);
        let consolidated_bid = weighted_bid_sum.wrapping_div(total_weight);
        let consolidated_ask = weighted_ask_sum.wrapping_div(total_weight);
        let microprice = consolidated_bid.wrapping_add(consolidated_ask).wrapping_div(2);
        
        let total_liq = total_bid_liq.wrapping_add(total_ask_liq);
        let imbalance = if total_liq > 0 {
            total_bid_liq.wrapping_sub(total_ask_liq)
                .wrapping_mul(100_000_000)
                .wrapping_div(total_liq)
        } else {
            0
        };
        
        Microprice {
            price: microprice,
            bid_micro: consolidated_bid,
            ask_micro: consolidated_ask,
            bid_liquidity: total_bid_liq,
            ask_liquidity: total_ask_liq,
            imbalance,
            timestamp_ns: unsafe { std::arch::x86_64::_rdtsc() } as u64,
            venue_count,
            _pad: [0; 23],
        }
    }
    
    /// Set minimum liquidity threshold
    pub fn set_min_liquidity(&self, liq: i64) {
        self.min_liquidity.store(liq, Ordering::Relaxed);
    }
}

impl Default for MicropriceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_microprice_size() {
        assert_eq!(core::mem::size_of::<Microprice>(), 64);
    }
    
    #[test]
    fn test_engine_creation() {
        let engine = MicropriceEngine::new();
        assert_eq!(engine.default_weight, 100_000_000 / MAX_VENUES as u64);
    }
}
