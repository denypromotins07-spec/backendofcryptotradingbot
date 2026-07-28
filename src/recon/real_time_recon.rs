//! Real-Time Reconciliation Engine
//! 
//! Real-time trade, position, and balance reconciliation against exchange feeds.
//! Includes circuit breaker for halting trading on balance discrepancies.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of instruments to reconcile
const MAX_INSTRUMENTS: usize = 256;

/// Reconciliation status
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconStatus {
    Matched = 0,
    Discrepancy = 1,
    Pending = 2,
    CircuitBreaker = 3,
}

/// Balance record - cache-line aligned
#[repr(C)]
struct BalanceRecord {
    /// Internal (our) balance in base units * 10^8
    internal_balance: AtomicI64,
    /// Exchange-reported balance in base units * 10^8
    exchange_balance: AtomicI64,
    /// Last reconciliation timestamp (cycles)
    last_recon_time: AtomicU64,
    /// Discrepancy count
    discrepancy_count: AtomicU32,
    _padding: [u8; CACHE_LINE_SIZE - 28],
}

impl BalanceRecord {
    const fn new() -> Self {
        Self {
            internal_balance: AtomicI64::new(0),
            exchange_balance: AtomicI64::new(0),
            last_recon_time: AtomicU64::new(0),
            discrepancy_count: AtomicU32::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 28],
        }
    }
}

/// Trade record for reconciliation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TradeRecord {
    pub trade_id: u64,
    pub order_id: u64,
    pub instrument_id: u16,
    pub side: u8,
    pub quantity: u64,
    pub price: u64,
    pub fee: u64,
    pub timestamp: u64,
    pub confirmed: bool,
    _padding: [u8; 24],
}

impl TradeRecord {
    const fn empty() -> Self {
        Self {
            trade_id: 0,
            order_id: 0,
            instrument_id: 0,
            side: 0,
            quantity: 0,
            price: 0,
            fee: 0,
            timestamp: 0,
            confirmed: false,
            _padding: [0u8; 24],
        }
    }
}

/// Reconciliation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReconResult {
    pub status: ReconStatus,
    pub instrument_id: u16,
    pub internal_balance: i64,
    pub exchange_balance: i64,
    pub discrepancy: i64,
    pub discrepancy_bps: i16,
    pub timestamp: u64,
    _padding: [u8; 14],
}

impl ReconResult {
    const fn matched() -> Self {
        Self {
            status: ReconStatus::Matched,
            instrument_id: 0,
            internal_balance: 0,
            exchange_balance: 0,
            discrepancy: 0,
            discrepancy_bps: 0,
            timestamp: 0,
            _padding: [0u8; 14],
        }
    }
}

/// Real-time reconciliation engine
pub struct RealTimeRecon {
    /// Per-instrument balance records
    balances: [BalanceRecord; MAX_INSTRUMENTS],
    /// Pending trades awaiting confirmation
    pending_trades: [TradeRecord; 1024],
    /// Number of pending trades
    num_pending: AtomicU64,
    /// Confirmed trades circular buffer index
    confirmed_idx: AtomicU64,
    /// Circuit breaker threshold (basis points)
    circuit_breaker_threshold_bp: AtomicU64,
    /// Circuit breaker triggered flag
    circuit_breaker_active: AtomicBool,
    /// Global reconciliation enabled
    recon_enabled: AtomicBool,
    /// Total discrepancy count
    total_discrepancies: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for RealTimeRecon {}
unsafe impl Sync for RealTimeRecon {}

impl RealTimeRecon {
    /// Create new reconciliation engine
    pub const fn new() -> Self {
        const EMPTY_BALANCE: BalanceRecord = BalanceRecord::new();
        const EMPTY_TRADE: TradeRecord = TradeRecord::empty();
        
        Self {
            balances: [EMPTY_BALANCE; MAX_INSTRUMENTS],
            pending_trades: [EMPTY_TRADE; 1024],
            num_pending: AtomicU64::new(0),
            confirmed_idx: AtomicU64::new(0),
            circuit_breaker_threshold_bp: AtomicU64::new(100), // 1% threshold
            circuit_breaker_active: AtomicBool::new(false),
            recon_enabled: AtomicBool::new(true),
            total_discrepancies: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Update internal balance
    #[inline(always)]
    pub fn update_internal_balance(&self, instrument_id: u16, delta: i64) {
        if instrument_id >= MAX_INSTRUMENTS as u16 {
            return;
        }
        let idx = instrument_id as usize;
        self.balances[idx].internal_balance.fetch_add(delta, Ordering::AcqRel);
    }

    /// Update exchange-reported balance
    #[inline(always)]
    pub fn update_exchange_balance(&self, instrument_id: u16, balance: i64) -> ReconResult {
        if instrument_id >= MAX_INSTRUMENTS as u16 || !self.recon_enabled.load(Ordering::Relaxed) {
            return ReconResult::matched();
        }

        let idx = instrument_id as usize;
        let record = &self.balances[idx];
        
        record.exchange_balance.store(balance, Ordering::Release);
        record.last_recon_time.store(unsafe { _rdtsc() }, Ordering::Release);

        // Check for discrepancy
        let internal = record.internal_balance.load(Ordering::Acquire);
        let discrepancy = internal - balance;
        
        // Calculate discrepancy in basis points
        let abs_internal = internal.unsigned_abs();
        let abs_discrepancy = discrepancy.unsigned_abs();
        let discrepancy_bps = if abs_internal > 0 {
            ((abs_discrepancy * 10000) / abs_internal) as i16
        } else {
            0
        };

        // Check circuit breaker
        let threshold = self.circuit_breaker_threshold_bp.load(Ordering::Relaxed);
        if discrepancy_bps as u64 >= threshold {
            self.trigger_circuit_breaker(instrument_id);
            
            return ReconResult {
                status: ReconStatus::CircuitBreaker,
                instrument_id,
                internal_balance: internal,
                exchange_balance: balance,
                discrepancy,
                discrepancy_bps,
                timestamp: unsafe { _rdtsc() },
                _padding: [0u8; 14],
            };
        }

        if discrepancy != 0 {
            record.discrepancy_count.fetch_add(1, Ordering::Relaxed);
            self.total_discrepancies.fetch_add(1, Ordering::Relaxed);
            
            ReconResult {
                status: ReconStatus::Discrepancy,
                instrument_id,
                internal_balance: internal,
                exchange_balance: balance,
                discrepancy,
                discrepancy_bps,
                timestamp: unsafe { _rdtsc() },
                _padding: [0u8; 14],
            }
        } else {
            ReconResult {
                status: ReconStatus::Matched,
                instrument_id,
                internal_balance: internal,
                exchange_balance: balance,
                discrepancy: 0,
                discrepancy_bps: 0,
                timestamp: unsafe { _rdtsc() },
                _padding: [0u8; 14],
            }
        }
    }

    /// Add pending trade
    #[inline(always)]
    pub fn add_pending_trade(&self, trade: TradeRecord) -> bool {
        let pending = self.num_pending.load(Ordering::Relaxed);
        if pending >= 1024 {
            return false;
        }

        let idx = pending as usize;
        self.pending_trades[idx] = trade;
        self.num_pending.fetch_add(1, Ordering::Release);
        true
    }

    /// Confirm a trade matches exchange report
    #[inline(always)]
    pub fn confirm_trade(&self, trade_id: u64) -> bool {
        let pending = self.num_pending.load(Ordering::Relaxed);
        
        for i in 0..pending.min(1024) {
            if self.pending_trades[i as usize].trade_id == trade_id {
                // Mark as confirmed (would normally remove from pending)
                self.pending_trades[i as usize].confirmed = true;
                self.confirmed_idx.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Trigger circuit breaker
    #[inline(always)]
    fn trigger_circuit_breaker(&self, instrument_id: u16) {
        self.circuit_breaker_active.store(true, Ordering::SeqCst);
        // Log event (in production would notify risk system)
        let _ = instrument_id;
    }

    /// Check if circuit breaker is active
    #[inline(always)]
    pub fn is_circuit_breaker_active(&self) -> bool {
        self.circuit_breaker_active.load(Ordering::Acquire)
    }

    /// Reset circuit breaker (manual intervention required)
    #[inline(always)]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker_active.store(false, Ordering::SeqCst);
    }

    /// Get balance discrepancy for instrument
    #[inline(always)]
    pub fn get_discrepancy(&self, instrument_id: u16) -> i64 {
        if instrument_id >= MAX_INSTRUMENTS as u16 {
            return 0;
        }
        let idx = instrument_id as usize;
        let internal = self.balances[idx].internal_balance.load(Ordering::Relaxed);
        let exchange = self.balances[idx].exchange_balance.load(Ordering::Relaxed);
        internal - exchange
    }

    /// Get internal balance
    #[inline(always)]
    pub fn get_internal_balance(&self, instrument_id: u16) -> i64 {
        if instrument_id >= MAX_INSTRUMENTS as u16 {
            return 0;
        }
        self.balances[instrument_id as usize].internal_balance.load(Ordering::Relaxed)
    }

    /// Get exchange balance
    #[inline(always)]
    pub fn get_exchange_balance(&self, instrument_id: u16) -> i64 {
        if instrument_id >= MAX_INSTRUMENTS as u16 {
            return 0;
        }
        self.balances[instrument_id as usize].exchange_balance.load(Ordering::Relaxed)
    }

    /// Set circuit breaker threshold
    #[inline(always)]
    pub fn set_circuit_breaker_threshold_bp(&self, threshold_bp: u64) {
        self.circuit_breaker_threshold_bp.store(threshold_bp, Ordering::Release);
    }

    /// Get total discrepancy count
    #[inline(always)]
    pub fn get_total_discrepancies(&self) -> u64 {
        self.total_discrepancies.load(Ordering::Relaxed)
    }

    /// Get number of pending trades
    #[inline(always)]
    pub fn get_pending_count(&self) -> u64 {
        self.num_pending.load(Ordering::Relaxed)
    }

    /// SIMD-accelerated multi-instrument reconciliation
    #[inline(always)]
    pub fn reconcile_multiple_simd(&self, instrument_ids: &[u16; 4], exchange_balances: &[i64; 4]) -> [ReconResult; 4] {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.reconcile_multiple_avx2(instrument_ids, exchange_balances)
            } else {
                let mut results = [ReconResult::matched(); 4];
                for i in 0..4 {
                    results[i] = self.update_exchange_balance(instrument_ids[i], exchange_balances[i]);
                }
                results
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(always)]
    unsafe fn reconcile_multiple_avx2(&self, instrument_ids: &[u16; 4], exchange_balances: &[i64; 4]) -> [ReconResult; 4] {
        use core::arch::x86_64::*;

        // Load internal balances
        let mut internal = [0i64; 4];
        for i in 0..4 {
            if instrument_ids[i] < MAX_INSTRUMENTS as u16 {
                internal[i] = self.balances[instrument_ids[i] as usize].internal_balance.load(Ordering::Relaxed);
            }
        }

        // Use AVX2 for parallel comparison
        let internal_vec = _mm256_loadu_si256(internal.as_ptr() as *const __m256i);
        let exchange_vec = _mm256_loadu_si256(exchange_balances.as_ptr() as *const __m256i);

        // Calculate differences
        let diff_vec = _mm256_sub_epi64(internal_vec, exchange_vec);

        // Extract results
        let mut results = [ReconResult::matched(); 4];
        for i in 0..4 {
            let diff = _mm256_extract_epi64::<0>(diff_vec) >> (i * 64);
            let diff = diff as i64;
            
            if diff == 0 {
                results[i].status = ReconStatus::Matched;
            } else {
                results[i].status = ReconStatus::Discrepancy;
                results[i].discrepancy = diff;
            }
            results[i].instrument_id = instrument_ids[i];
            results[i].internal_balance = internal[i];
            results[i].exchange_balance = exchange_balances[i];
            results[i].timestamp = unsafe { _rdtsc() };
        }

        results
    }

    /// Enable reconciliation
    #[inline(always)]
    pub fn enable(&self) {
        self.recon_enabled.store(true, Ordering::Release);
    }

    /// Disable reconciliation
    #[inline(always)]
    pub fn disable(&self) {
        self.recon_enabled.store(false, Ordering::Release);
    }

    /// Reset all balances
    #[inline(always)]
    pub fn reset_all(&self) {
        for i in 0..MAX_INSTRUMENTS {
            self.balances[i] = BalanceRecord::new();
        }
        self.num_pending.store(0, Ordering::Release);
        self.circuit_breaker_active.store(false, Ordering::SeqCst);
        self.total_discrepancies.store(0, Ordering::Release);
    }
}

impl Default for RealTimeRecon {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recon_result_size() {
        assert_eq!(core::mem::size_of::<ReconResult>(), 48);
    }

    #[test]
    fn test_balance_update() {
        let recon = RealTimeRecon::new();
        
        recon.update_internal_balance(0, 1_000_000);
        let result = recon.update_exchange_balance(0, 1_000_000);
        
        assert_eq!(result.status, ReconStatus::Matched);
        assert_eq!(result.discrepancy, 0);
    }

    #[test]
    fn test_discrepancy_detection() {
        let recon = RealTimeRecon::new();
        
        recon.update_internal_balance(0, 1_000_000);
        let result = recon.update_exchange_balance(0, 990_000); // 1% discrepancy
        
        assert_eq!(result.status, ReconStatus::Discrepancy);
        assert_eq!(result.discrepancy, 10_000);
    }

    #[test]
    fn test_circuit_breaker() {
        let recon = RealTimeRecon::new();
        recon.set_circuit_breaker_threshold_bp(50); // 0.5% threshold
        
        recon.update_internal_balance(0, 1_000_000);
        let result = recon.update_exchange_balance(0, 990_000); // 1% discrepancy
        
        assert_eq!(result.status, ReconStatus::CircuitBreaker);
        assert!(recon.is_circuit_breaker_active());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let recon = RealTimeRecon::new();
        recon.set_circuit_breaker_threshold_bp(50);
        
        recon.update_internal_balance(0, 1_000_000);
        recon.update_exchange_balance(0, 990_000);
        
        assert!(recon.is_circuit_breaker_active());
        
        recon.reset_circuit_breaker();
        assert!(!recon.is_circuit_breaker_active());
    }

    #[test]
    fn test_pending_trades() {
        let recon = RealTimeRecon::new();
        
        let trade = TradeRecord {
            trade_id: 12345,
            order_id: 67890,
            instrument_id: 0,
            side: 0,
            quantity: 100,
            price: 50000,
            fee: 10,
            timestamp: 1000,
            confirmed: false,
            _padding: [0u8; 24],
        };
        
        assert!(recon.add_pending_trade(trade));
        assert_eq!(recon.get_pending_count(), 1);
        
        assert!(recon.confirm_trade(12345));
    }

    #[test]
    fn test_discrepancy_tracking() {
        let recon = RealTimeRecon::new();
        
        recon.update_internal_balance(0, 1_000_000);
        recon.update_exchange_balance(0, 999_000);
        recon.update_exchange_balance(0, 998_000);
        
        assert!(recon.get_total_discrepancies() >= 1);
    }

    #[test]
    fn test_multi_instrument_recon() {
        let recon = RealTimeRecon::new();
        
        recon.update_internal_balance(0, 1_000_000);
        recon.update_internal_balance(1, 2_000_000);
        
        let ids = [0u16, 1, 2, 3];
        let balances = [1_000_000i64, 2_000_000, 0, 0];
        
        let results = recon.reconcile_multiple_simd(&ids, &balances);
        
        assert_eq!(results[0].status, ReconStatus::Matched);
        assert_eq!(results[1].status, ReconStatus::Matched);
    }
}
