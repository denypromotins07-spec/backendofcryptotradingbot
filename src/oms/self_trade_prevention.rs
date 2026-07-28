//! Self-Trade Prevention Engine
//! 
//! Self-trade prevention across subaccounts using atomic bitset matching.
//! Uses core::arch for manually unrolled loops and O(1) lookups.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of subaccounts
const MAX_SUBACCOUNTS: usize = 256;

/// Maximum number of instruments per subaccount
const MAX_INSTRUMENTS_PER_ACCOUNT: usize = 64;

/// Self-trade prevention mode
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StpMode {
    /// Allow self-trades
    None = 0,
    /// Cancel aggressive order
    CancelAggressive = 1,
    /// Cancel resting order
    CancelResting = 2,
    /// Cancel both orders
    CancelBoth = 3,
    /// Decrement and cancel (reduce quantity)
    DecrementAndCancel = 4,
}

/// Order direction for STP matching
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    Buy = 0,
    Sell = 1,
}

/// Subaccount order book entry - tracks resting orders
#[repr(C)]
struct AccountOrderBook {
    /// Bitmask of instruments with resting buy orders
    buy_instruments: AtomicU64,
    /// Bitmask of instruments with resting sell orders
    sell_instruments: AtomicU64,
    /// Per-instrument buy order count
    buy_counts: [AtomicU64; MAX_INSTRUMENTS_PER_ACCOUNT],
    /// Per-instrument sell order count
    sell_counts: [AtomicU64; MAX_INSTRUMENTS_PER_ACCOUNT],
    _padding: [u8; CACHE_LINE_SIZE],
}

impl AccountOrderBook {
    const fn new() -> Self {
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        Self {
            buy_instruments: AtomicU64::new(0),
            sell_instruments: AtomicU64::new(0),
            buy_counts: [ZERO_U64; MAX_INSTRUMENTS_PER_ACCOUNT],
            sell_counts: [ZERO_U64; MAX_INSTRUMENTS_PER_ACCOUNT],
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline(always)]
    fn add_buy(&self, instrument_id: u8) {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return;
        }
        let idx = instrument_id as usize;
        self.buy_instruments.fetch_or(1u64 << idx, Ordering::Relaxed);
        self.buy_counts[idx].fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn add_sell(&self, instrument_id: u8) {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return;
        }
        let idx = instrument_id as usize;
        self.sell_instruments.fetch_or(1u64 << idx, Ordering::Relaxed);
        self.sell_counts[idx].fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn remove_buy(&self, instrument_id: u8) {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return;
        }
        let idx = instrument_id as usize;
        let count = self.buy_counts[idx].fetch_sub(1, Ordering::Relaxed);
        if count <= 1 {
            self.buy_instruments.fetch_and(!(1u64 << idx), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn remove_sell(&self, instrument_id: u8) {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return;
        }
        let idx = instrument_id as usize;
        let count = self.sell_counts[idx].fetch_sub(1, Ordering::Relaxed);
        if count <= 1 {
            self.sell_instruments.fetch_and(!(1u64 << idx), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn has_resting_buy(&self, instrument_id: u8) -> bool {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return false;
        }
        let idx = instrument_id as usize;
        (self.buy_instruments.load(Ordering::Relaxed) & (1u64 << idx)) != 0
    }

    #[inline(always)]
    fn has_resting_sell(&self, instrument_id: u8) -> bool {
        if instrument_id >= MAX_INSTRUMENTS_PER_ACCOUNT as u8 {
            return false;
        }
        let idx = instrument_id as usize;
        (self.sell_instruments.load(Ordering::Relaxed) & (1u64 << idx)) != 0
    }
}

/// Self-trade prevention result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StpResult {
    pub should_block: bool,
    pub mode: StpMode,
    pub resting_order_account: u16,
    pub resting_order_id: u64,
    pub aggressor_order_id: u64,
    pub matched_quantity: u64,
    pub _padding: [u8; 24],
}

impl StpResult {
    const fn none() -> Self {
        Self {
            should_block: false,
            mode: StpMode::None,
            resting_order_account: 0,
            resting_order_id: 0,
            aggressor_order_id: 0,
            matched_quantity: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Self-trade prevention engine
pub struct StpEngine {
    /// Per-subaccount order books
    order_books: [AccountOrderBook; MAX_SUBACCOUNTS],
    /// STP mode per subaccount pair (encoded as index)
    stp_modes: [AtomicU64; MAX_SUBACCOUNTS],
    /// Global STP enabled flag
    stp_enabled: AtomicBool,
    /// Default STP mode
    default_mode: StpMode,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for StpEngine {}
unsafe impl Sync for StpEngine {}

impl StpEngine {
    /// Create new STP engine
    pub const fn new() -> Self {
        const EMPTY_BOOK: AccountOrderBook = AccountOrderBook::new();
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        
        Self {
            order_books: [EMPTY_BOOK; MAX_SUBACCOUNTS],
            stp_modes: [ZERO_U64; MAX_SUBACCOUNTS],
            stp_enabled: AtomicBool::new(true),
            default_mode: StpMode::CancelAggressive,
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Set STP mode for a subaccount
    #[inline(always)]
    pub fn set_stp_mode(&self, account_id: u16, mode: StpMode) {
        if account_id < MAX_SUBACCOUNTS as u16 {
            self.stp_modes[account_id as usize].store(mode as u64, Ordering::Release);
        }
    }

    /// Get STP mode for a subaccount
    #[inline(always)]
    pub fn get_stp_mode(&self, account_id: u16) -> StpMode {
        if account_id < MAX_SUBACCOUNTS as u16 {
            match self.stp_modes[account_id as usize].load(Ordering::Relaxed) {
                0 => StpMode::None,
                1 => StpMode::CancelAggressive,
                2 => StpMode::CancelResting,
                3 => StpMode::CancelBoth,
                4 => StpMode::DecrementAndCancel,
                _ => self.default_mode,
            }
        } else {
            self.default_mode
        }
    }

    /// Register resting buy order
    #[inline(always)]
    pub fn register_resting_buy(&self, account_id: u16, instrument_id: u8) {
        if account_id < MAX_SUBACCOUNTS as u16 {
            self.order_books[account_id as usize].add_buy(instrument_id);
        }
    }

    /// Register resting sell order
    #[inline(always)]
    pub fn register_resting_sell(&self, account_id: u16, instrument_id: u8) {
        if account_id < MAX_SUBACCOUNTS as u16 {
            self.order_books[account_id as usize].add_sell(instrument_id);
        }
    }

    /// Remove resting buy order
    #[inline(always)]
    pub fn remove_resting_buy(&self, account_id: u16, instrument_id: u8) {
        if account_id < MAX_SUBACCOUNTS as u16 {
            self.order_books[account_id as usize].remove_buy(instrument_id);
        }
    }

    /// Remove resting sell order
    #[inline(always)]
    pub fn remove_resting_sell(&self, account_id: u16, instrument_id: u8) {
        if account_id < MAX_SUBACCOUNTS as u16 {
            self.order_books[account_id as usize].remove_sell(instrument_id);
        }
    }

    /// Check for self-trade before submitting aggressive order
    #[inline(always)]
    pub fn check_aggressive_order(
        &self,
        aggressor_account: u16,
        instrument_id: u8,
        direction: OrderDirection,
        quantity: u64,
        order_id: u64,
    ) -> StpResult {
        if !self.stp_enabled.load(Ordering::Relaxed) {
            return StpResult::none();
        }

        let aggressor_book = &self.order_books[aggressor_account as usize];
        let aggressor_mode = self.get_stp_mode(aggressor_account);

        if aggressor_mode == StpMode::None {
            return StpResult::none();
        }

        // Check for opposite side resting orders in same account
        let has_conflict = match direction {
            OrderDirection::Buy => aggressor_book.has_resting_sell(instrument_id),
            OrderDirection::Sell => aggressor_book.has_resting_buy(instrument_id),
        };

        if !has_conflict {
            return StpResult::none();
        }

        // Self-trade detected
        StpResult {
            should_block: true,
            mode: aggressor_mode,
            resting_order_account: aggressor_account,
            resting_order_id: 0, // Would be filled from order book
            aggressor_order_id: order_id,
            matched_quantity: quantity,
            _padding: [0u8; 24],
        }
    }

    /// SIMD-accelerated cross-account STP check
    /// Checks multiple accounts simultaneously for self-trade conflicts
    #[inline(always)]
    pub fn check_cross_account_stp_simd(
        &self,
        aggressor_account: u16,
        instrument_id: u8,
        direction: OrderDirection,
        account_ids: &[u16; 4],
    ) -> [bool; 4] {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.check_cross_account_stp_avx2(aggressor_account, instrument_id, direction, account_ids)
            } else {
                let mut results = [false; 4];
                for i in 0..4 {
                    results[i] = self.check_single_account_stp(
                        account_ids[i],
                        instrument_id,
                        direction,
                    );
                }
                results
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(always)]
    unsafe fn check_cross_account_stp_avx2(
        &self,
        _aggressor_account: u16,
        instrument_id: u8,
        direction: OrderDirection,
        account_ids: &[u16; 4],
    ) -> [bool; 4] {
        use core::arch::x86_64::*;

        // Load instrument bitmasks for 4 accounts
        let mut buy_masks = [0u64; 4];
        let mut sell_masks = [0u64; 4];

        for i in 0..4 {
            let acc_id = account_ids[i] as usize;
            if acc_id < MAX_SUBACCOUNTS {
                buy_masks[i] = self.order_books[acc_id].buy_instruments.load(Ordering::Relaxed);
                sell_masks[i] = self.order_books[acc_id].sell_instruments.load(Ordering::Relaxed);
            }
        }

        // Create bitmask for the instrument
        let inst_mask = 1u64 << (instrument_id as u64 % 64);

        // Use AVX2 to check all accounts in parallel
        let buy_vec = _mm256_loadu_si256(buy_masks.as_ptr() as *const __m256i);
        let sell_vec = _mm256_loadu_si256(sell_masks.as_ptr() as *const __m256i);
        let mask_vec = _mm256_set1_epi64x(inst_mask as i64);

        // AND with instrument mask
        let buy_match = _mm256_and_si256(buy_vec, mask_vec);
        let sell_match = _mm256_and_si256(sell_vec, mask_vec);

        // Check if any bits are set (non-zero)
        let buy_nonzero = _mm256_testz_si256(buy_match, buy_match) == 0;
        let sell_nonzero = _mm256_testz_si256(sell_match, sell_match) == 0;

        // Extract results based on direction
        let mut results = [false; 4];
        
        // Manually unroll for O(1) extraction
        let buy_bits = _mm256_movemask_epi8(buy_match);
        let sell_bits = _mm256_movemask_epi8(sell_match);

        match direction {
            OrderDirection::Buy => {
                // Check for resting sells (conflict for aggressive buy)
                results[0] = (sell_bits & 0x1) != 0;
                results[1] = (sell_bits & 0x4) != 0;
                results[2] = (sell_bits & 0x10) != 0;
                results[3] = (sell_bits & 0x40) != 0;
            }
            OrderDirection::Sell => {
                // Check for resting buys (conflict for aggressive sell)
                results[0] = (buy_bits & 0x1) != 0;
                results[1] = (buy_bits & 0x4) != 0;
                results[2] = (buy_bits & 0x10) != 0;
                results[3] = (buy_bits & 0x40) != 0;
            }
        }

        results
    }

    /// Single account STP check
    #[inline(always)]
    fn check_single_account_stp(
        &self,
        account_id: u16,
        instrument_id: u8,
        direction: OrderDirection,
    ) -> bool {
        if account_id >= MAX_SUBACCOUNTS as u16 {
            return false;
        }

        let book = &self.order_books[account_id as usize];
        match direction {
            OrderDirection::Buy => book.has_resting_sell(instrument_id),
            OrderDirection::Sell => book.has_resting_buy(instrument_id),
        }
    }

    /// Apply STP action based on result
    #[inline(always)]
    pub fn apply_stp_action(&self, result: &StpResult) -> StpAction {
        match result.mode {
            StpMode::CancelAggressive => StpAction::CancelAggressive,
            StpMode::CancelResting => StpAction::CancelResting(result.resting_order_account),
            StpMode::CancelBoth => StpAction::CancelBoth,
            StpMode::DecrementAndCancel => {
                StpAction::Decrement(result.matched_quantity)
            }
            StpMode::None => StpAction::Allow,
        }
    }

    /// Enable STP globally
    #[inline(always)]
    pub fn enable_stp(&self) {
        self.stp_enabled.store(true, Ordering::SeqCst);
    }

    /// Disable STP globally
    #[inline(always)]
    pub fn disable_stp(&self) {
        self.stp_enabled.store(false, Ordering::SeqCst);
    }

    /// Check if STP is enabled
    #[inline(always)]
    pub fn is_stp_enabled(&self) -> bool {
        self.stp_enabled.load(Ordering::Relaxed)
    }

    /// Set default STP mode
    #[inline(always)]
    pub fn set_default_mode(&mut self, mode: StpMode) {
        self.default_mode = mode;
    }

    /// Reset all order books
    #[inline(always)]
    pub fn reset_all(&self) {
        for i in 0..MAX_SUBACCOUNTS {
            self.order_books[i] = AccountOrderBook::new();
        }
    }
}

impl Default for StpEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Action to take after STP check
#[derive(Debug, Clone, Copy)]
pub enum StpAction {
    Allow,
    CancelAggressive,
    CancelResting(u16), // account_id
    CancelBoth,
    Decrement(u64), // quantity to remove
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stp_result_size() {
        assert_eq!(core::mem::size_of::<StpResult>(), 48);
    }

    #[test]
    fn test_register_and_check() {
        let engine = StpEngine::new();
        
        // Register resting buy
        engine.register_resting_buy(0, 0);
        
        // Check aggressive sell from same account
        let result = engine.check_aggressive_order(
            0,
            0,
            OrderDirection::Sell,
            100,
            12345,
        );
        
        assert!(result.should_block);
        assert_eq!(result.mode, StpMode::CancelAggressive);
    }

    #[test]
    fn test_no_self_trade_different_accounts() {
        let engine = StpEngine::new();
        
        // Register order in account 0
        engine.register_resting_buy(0, 0);
        
        // Check aggressive sell from account 1
        let result = engine.check_aggressive_order(
            1,
            0,
            OrderDirection::Sell,
            100,
            12345,
        );
        
        assert!(!result.should_block);
    }

    #[test]
    fn test_no_self_trade_different_instruments() {
        let engine = StpEngine::new();
        
        // Register resting buy on instrument 0
        engine.register_resting_buy(0, 0);
        
        // Check aggressive sell on instrument 1
        let result = engine.check_aggressive_order(
            0,
            1,
            OrderDirection::Sell,
            100,
            12345,
        );
        
        assert!(!result.should_block);
    }

    #[test]
    fn test_stp_modes() {
        let mut engine = StpEngine::new();
        
        engine.set_stp_mode(0, StpMode::CancelBoth);
        assert_eq!(engine.get_stp_mode(0), StpMode::CancelBoth);
        
        engine.set_stp_mode(1, StpMode::DecrementAndCancel);
        assert_eq!(engine.get_stp_mode(1), StpMode::DecrementAndCancel);
    }

    #[test]
    fn test_stp_disable() {
        let engine = StpEngine::new();
        
        engine.register_resting_buy(0, 0);
        
        assert!(engine.is_stp_enabled());
        
        engine.disable_stp();
        
        let result = engine.check_aggressive_order(
            0,
            0,
            OrderDirection::Sell,
            100,
            12345,
        );
        
        assert!(!result.should_block);
    }

    #[test]
    fn test_apply_stp_action() {
        let engine = StpEngine::new();
        
        let result = StpResult {
            should_block: true,
            mode: StpMode::CancelBoth,
            resting_order_account: 0,
            resting_order_id: 100,
            aggressor_order_id: 200,
            matched_quantity: 50,
            _padding: [0u8; 24],
        };
        
        let action = engine.apply_stp_action(&result);
        assert!(matches!(action, StpAction::CancelBoth));
    }

    #[test]
    fn test_remove_order() {
        let engine = StpEngine::new();
        
        engine.register_resting_buy(0, 0);
        
        // Should have conflict
        let result1 = engine.check_aggressive_order(0, 0, OrderDirection::Sell, 100, 1);
        assert!(result1.should_block);
        
        // Remove the resting order
        engine.remove_resting_buy(0, 0);
        
        // Should not have conflict anymore
        let result2 = engine.check_aggressive_order(0, 0, OrderDirection::Sell, 100, 2);
        assert!(!result2.should_block);
    }

    #[test]
    fn test_cross_account_simd() {
        let engine = StpEngine::new();
        
        // Set up some resting orders
        engine.register_resting_sell(0, 0);
        engine.register_resting_sell(1, 0);
        
        let account_ids = [0u16, 1, 2, 3];
        let results = engine.check_cross_account_stp_simd(
            99,
            0,
            OrderDirection::Buy,
            &account_ids,
        );
        
        // Accounts 0 and 1 have resting sells, so they should conflict
        assert!(results[0]);
        assert!(results[1]);
        assert!(!results[2]);
        assert!(!results[3]);
    }
}
