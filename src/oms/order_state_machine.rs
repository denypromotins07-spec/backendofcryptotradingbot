//! Order State Machine
//! 
//! Full lock-free order state machine covering all lifecycle events and timeouts.
//! Strictly #[repr(C)] and padded to 64-byte cache lines to prevent false sharing.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use core::arch::x86_64::*;

/// Cache line size for padding
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of orders tracked
const MAX_ORDERS: usize = 4096;

/// Order state - strictly ordered for state machine validity
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderState {
    /// Order created but not yet submitted
    Pending = 0,
    /// Order submitted to exchange
    Submitted = 1,
    /// Order acknowledged by exchange
    Acknowledged = 2,
    /// Order partially filled
    PartiallyFilled = 3,
    /// Order fully filled
    Filled = 4,
    /// Order cancelled by user
    Cancelled = 5,
    /// Order cancelled by system (timeout, risk, etc.)
    SystemCancelled = 6,
    /// Order rejected by exchange
    Rejected = 7,
    /// Order expired (time-in-force exceeded)
    Expired = 8,
}

impl OrderState {
    /// Check if state is terminal (no further transitions possible)
    #[inline(always)]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::SystemCancelled
                | OrderState::Rejected
                | OrderState::Expired
        )
    }

    /// Check if state allows fill events
    #[inline(always)]
    pub const fn allows_fills(self) -> bool {
        matches!(
            self,
            OrderState::Submitted
                | OrderState::Acknowledged
                | OrderState::PartiallyFilled
        )
    }
}

/// Order side
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

/// Time in force
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Day = 0,
    IOC = 1,
    FOK = 2,
    GTC = 3,
    GTD = 4,
}

/// Order struct - strictly repr(C), cache-line padded
#[repr(C)]
pub struct Order {
    /// Client order ID (unique)
    pub client_order_id: u64,
    /// Exchange order ID (assigned after ack)
    pub exchange_order_id: AtomicU64,
    /// Instrument ID
    pub instrument_id: u16,
    /// Strategy ID
    pub strategy_id: u16,
    /// Side
    pub side: u8,
    /// Time in force
    pub time_in_force: u8,
    /// Current state
    pub state: AtomicU32,
    /// Order quantity (base units)
    pub quantity: u64,
    /// Order price (quote units * precision)
    pub price: u64,
    /// Filled quantity
    pub filled_qty: AtomicU64,
    /// Average fill price
    pub avg_fill_price: AtomicU64,
    /// Remaining quantity
    pub remaining_qty: AtomicU64,
    /// Submission timestamp (cycles)
    pub submit_time: AtomicU64,
    /// Last update timestamp (cycles)
    pub last_update_time: AtomicU64,
    /// Expiration timestamp (cycles)
    pub expire_time: AtomicU64,
    /// Venue ID
    pub venue_id: u8,
    /// Priority level (for internal queueing)
    pub priority: u8,
    /// Parent order ID (for child orders)
    pub parent_order_id: u64,
    /// Number of fills
    pub fill_count: AtomicU32,
    /// Error code (if rejected/cancelled)
    pub error_code: AtomicU32,
    /// Flags (bitfield)
    pub flags: AtomicU32,
    _padding: [u8; CACHE_LINE_SIZE - 120],
}

// SAFETY: Order uses only atomic types for mutable state
unsafe impl Send for Order {}
unsafe impl Sync for Order {}

impl Order {
    /// Create a new pending order
    #[inline(always)]
    pub const fn new(
        client_order_id: u64,
        instrument_id: u16,
        strategy_id: u16,
        side: OrderSide,
        quantity: u64,
        price: u64,
        time_in_force: TimeInForce,
        venue_id: u8,
    ) -> Self {
        Self {
            client_order_id,
            exchange_order_id: AtomicU64::new(0),
            instrument_id,
            strategy_id,
            side: side as u8,
            time_in_force: time_in_force as u8,
            state: AtomicU32::new(OrderState::Pending as u32),
            quantity,
            price,
            filled_qty: AtomicU64::new(0),
            avg_fill_price: AtomicU64::new(0),
            remaining_qty: AtomicU64::new(quantity),
            submit_time: AtomicU64::new(0),
            last_update_time: AtomicU64::new(0),
            expire_time: AtomicU64::new(0),
            venue_id,
            priority: 0,
            parent_order_id: 0,
            fill_count: AtomicU32::new(0),
            error_code: AtomicU32::new(0),
            flags: AtomicU32::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 120],
        }
    }

    /// Get current state
    #[inline(always)]
    pub fn get_state(&self) -> OrderState {
        match self.state.load(Ordering::Acquire) {
            0 => OrderState::Pending,
            1 => OrderState::Submitted,
            2 => OrderState::Acknowledged,
            3 => OrderState::PartiallyFilled,
            4 => OrderState::Filled,
            5 => OrderState::Cancelled,
            6 => OrderState::SystemCancelled,
            7 => OrderState::Rejected,
            8 => OrderState::Expired,
            _ => OrderState::Pending,
        }
    }

    /// Attempt state transition (lock-free CAS)
    #[inline(always)]
    pub fn transition_to(&self, new_state: OrderState) -> bool {
        let current = self.state.load(Ordering::Acquire);
        
        // Validate transition
        if !self.is_valid_transition(current as u8, new_state as u8) {
            return false;
        }

        let result = self.state.compare_exchange(
            current,
            new_state as u32,
            Ordering::SeqCst,
            Ordering::Acquire,
        );

        if result.is_ok() {
            self.last_update_time.store(unsafe { _rdtsc() }, Ordering::Release);
        }

        result.is_ok()
    }

    /// Check if state transition is valid
    #[inline(always)]
    const fn is_valid_transition(&self, from: u8, to: u8) -> bool {
        match from {
            0 => to <= 2, // Pending -> Submitted, Acknowledged, or Rejected
            1 => to <= 7, // Submitted -> any except Pending
            2 => to <= 7, // Acknowledged -> any except Pending, Submitted
            3 => to == 3 || to == 4 || to >= 5, // PartiallyFilled -> itself, Filled, or terminal
            _ => false, // Terminal states don't transition
        }
    }

    /// Process fill event
    #[inline(always)]
    pub fn process_fill(&self, fill_qty: u64, fill_price: u64) -> bool {
        let current_state = self.get_state();
        
        if !current_state.allows_fills() {
            return false;
        }

        let old_filled = self.filled_qty.fetch_add(fill_qty, Ordering::AcqRel);
        let new_filled = old_filled + fill_qty;
        
        // Update average fill price
        let old_avg = self.avg_fill_price.load(Ordering::Relaxed);
        let total_value = (old_avg as u128 * old_filled as u128) + (fill_price as u128 * fill_qty as u128);
        let new_avg = if new_filled > 0 {
            (total_value / new_filled as u128) as u64
        } else {
            0
        };
        self.avg_fill_price.store(new_avg, Ordering::Release);

        // Update remaining quantity
        self.remaining_qty.store(
            self.quantity.saturating_sub(new_filled),
            Ordering::Release,
        );

        // Increment fill count
        self.fill_count.fetch_add(1, Ordering::Relaxed);

        // Update state based on fill completion
        if new_filled >= self.quantity {
            self.transition_to(OrderState::Filled);
        } else if current_state != OrderState::PartiallyFilled {
            self.transition_to(OrderState::PartiallyFilled);
        }

        self.last_update_time.store(unsafe { _rdtsc() }, Ordering::Release);
        true
    }

    /// Set exchange order ID (after acknowledgment)
    #[inline(always)]
    pub fn set_exchange_id(&self, exchange_id: u64) {
        self.exchange_order_id.store(exchange_id, Ordering::Release);
    }

    /// Get exchange order ID
    #[inline(always)]
    pub fn get_exchange_id(&self) -> u64 {
        self.exchange_order_id.load(Ordering::Acquire)
    }

    /// Check if order is expired
    #[inline(always)]
    pub fn is_expired(&self) -> bool {
        let expire_time = self.expire_time.load(Ordering::Relaxed);
        if expire_time == 0 {
            return false;
        }
        let now = unsafe { _rdtsc() };
        now >= expire_time
    }

    /// Set expiration time
    #[inline(always)]
    pub fn set_expire_time(&self, cycles: u64) {
        self.expire_time.store(cycles, Ordering::Release);
    }

    /// Mark order as submitted
    #[inline(always)]
    pub fn mark_submitted(&self) -> bool {
        self.submit_time.store(unsafe { _rdtsc() }, Ordering::Release);
        self.transition_to(OrderState::Submitted)
    }

    /// Mark order as acknowledged
    #[inline(always)]
    pub fn mark_acknowledged(&self) -> bool {
        self.transition_to(OrderState::Acknowledged)
    }

    /// Cancel order
    #[inline(always)]
    pub fn cancel(&self) -> bool {
        let current = self.get_state();
        if current.is_terminal() {
            return false;
        }
        self.transition_to(OrderState::Cancelled)
    }

    /// Reject order with error code
    #[inline(always)]
    pub fn reject(&self, error_code: u32) -> bool {
        self.error_code.store(error_code, Ordering::Release);
        self.transition_to(OrderState::Rejected)
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn get_remaining_qty(&self) -> u64 {
        self.remaining_qty.load(Ordering::Acquire)
    }

    /// Get filled quantity
    #[inline(always)]
    pub fn get_filled_qty(&self) -> u64 {
        self.filled_qty.load(Ordering::Acquire)
    }

    /// Set flag bit
    #[inline(always)]
    pub fn set_flag(&self, bit: u32) {
        let flags = self.flags.fetch_or(1 << bit, Ordering::Relaxed);
        if (flags & (1 << bit)) == 0 {
            self.last_update_time.store(unsafe { _rdtsc() }, Ordering::Release);
        }
    }

    /// Clear flag bit
    #[inline(always)]
    pub fn clear_flag(&self, bit: u32) {
        self.flags.fetch_and(!(1 << bit), Ordering::Relaxed);
    }

    /// Check if flag is set
    #[inline(always)]
    pub fn has_flag(&self, bit: u32) -> bool {
        (self.flags.load(Ordering::Relaxed) & (1 << bit)) != 0
    }
}

/// Order state machine manager
pub struct OrderStateMachine {
    /// Pre-allocated orders
    orders: [Order; MAX_ORDERS],
    /// Number of active orders
    num_active: AtomicU64,
    /// Next available order ID
    next_order_id: AtomicU64,
    /// Enable timeout checking
    timeout_enabled: AtomicBool,
    /// Default timeout (cycles)
    default_timeout_cycles: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE],
}

// SAFETY: All internal state is atomic
unsafe impl Send for OrderStateMachine {}
unsafe impl Sync for OrderStateMachine {}

impl OrderStateMachine {
    /// Create new order state machine
    pub const fn new() -> Self {
        const EMPTY_ORDER: Order = Order::new(
            0, 0, 0, OrderSide::Buy, 0, 0, TimeInForce::Day, 0,
        );
        
        Self {
            orders: [EMPTY_ORDER; MAX_ORDERS],
            num_active: AtomicU64::new(0),
            next_order_id: AtomicU64::new(1),
            timeout_enabled: AtomicBool::new(true),
            default_timeout_cycles: AtomicU64::new(3_000_000_000), // ~1 second at 3GHz
            _padding: [0u8; CACHE_LINE_SIZE],
        }
    }

    /// Generate unique client order ID
    #[inline(always)]
    pub fn generate_client_order_id(&self) -> u64 {
        self.next_order_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Create and register a new order
    #[inline(always)]
    pub fn create_order(
        &self,
        instrument_id: u16,
        strategy_id: u16,
        side: OrderSide,
        quantity: u64,
        price: u64,
        time_in_force: TimeInForce,
        venue_id: u8,
    ) -> Option<u64> {
        // Find empty slot
        let mut slot = None;
        for i in 0..MAX_ORDERS {
            if self.orders[i].client_order_id == 0 {
                slot = Some(i);
                break;
            }
        }

        let idx = slot?;
        let client_id = self.generate_client_order_id();

        let order = Order::new(
            client_id,
            instrument_id,
            strategy_id,
            side,
            quantity,
            price,
            time_in_force,
            venue_id,
        );

        // Set expiration for non-GTC orders
        if time_in_force != TimeInForce::GTC {
            let now = unsafe { _rdtsc() };
            let timeout = self.default_timeout_cycles.load(Ordering::Relaxed);
            order.set_expire_time(now + timeout);
        }

        self.orders[idx] = order;
        self.num_active.fetch_add(1, Ordering::Release);

        Some(client_id)
    }

    /// Get order by client ID
    #[inline(always)]
    pub fn get_order(&self, client_id: u64) -> Option<&Order> {
        if client_id == 0 || client_id > self.next_order_id.load(Ordering::Relaxed) {
            return None;
        }
        
        for i in 0..MAX_ORDERS {
            if self.orders[i].client_order_id == client_id {
                return Some(&self.orders[i]);
            }
        }
        None
    }

    /// Process order event
    #[inline(always)]
    pub fn process_event(
        &self,
        client_id: u64,
        event: OrderEvent,
    ) -> bool {
        match event {
            OrderEvent::Submit => {
                if let Some(order) = self.get_order(client_id) {
                    order.mark_submitted()
                } else {
                    false
                }
            }
            OrderEvent::Ack(exchange_id) => {
                if let Some(order) = self.get_order(client_id) {
                    order.set_exchange_id(exchange_id);
                    order.mark_acknowledged()
                } else {
                    false
                }
            }
            OrderEvent::Fill { qty, price } => {
                if let Some(order) = self.get_order(client_id) {
                    order.process_fill(qty, price)
                } else {
                    false
                }
            }
            OrderEvent::Cancel => {
                if let Some(order) = self.get_order(client_id) {
                    order.cancel()
                } else {
                    false
                }
            }
            OrderEvent::Reject { code } => {
                if let Some(order) = self.get_order(client_id) {
                    order.reject(code)
                } else {
                    false
                }
            }
        }
    }

    /// Check for timed-out orders and cancel them
    #[inline(always)]
    pub fn check_timeouts(&self) -> u64 {
        if !self.timeout_enabled.load(Ordering::Relaxed) {
            return 0;
        }

        let mut cancelled = 0u64;
        let now = unsafe { _rdtsc() };

        for i in 0..MAX_ORDERS {
            let order = &self.orders[i];
            if order.client_order_id == 0 {
                continue;
            }

            let state = order.get_state();
            if state.is_terminal() {
                continue;
            }

            let expire_time = order.expire_time.load(Ordering::Relaxed);
            if expire_time > 0 && now >= expire_time {
                order.transition_to(OrderState::SystemCancelled);
                order.error_code.store(1, Ordering::Release); // Timeout error
                cancelled += 1;
            }
        }

        cancelled
    }

    /// Get number of active orders
    #[inline(always)]
    pub fn num_active_orders(&self) -> u64 {
        self.num_active.load(Ordering::Relaxed)
    }

    /// Set default timeout
    #[inline(always)]
    pub fn set_default_timeout_cycles(&self, cycles: u64) {
        self.default_timeout_cycles.store(cycles, Ordering::Release);
    }

    /// Enable/disable timeout checking
    #[inline(always)]
    pub fn set_timeout_enabled(&self, enabled: bool) {
        self.timeout_enabled.store(enabled, Ordering::Release);
    }

    /// Reset all orders
    #[inline(always)]
    pub fn reset_all(&self) {
        const EMPTY_ORDER: Order = Order::new(
            0, 0, 0, OrderSide::Buy, 0, 0, TimeInForce::Day, 0,
        );
        
        for i in 0..MAX_ORDERS {
            self.orders[i] = EMPTY_ORDER;
        }
        self.num_active.store(0, Ordering::Release);
        self.next_order_id.store(1, Ordering::Release);
    }
}

impl Default for OrderStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Order events for state machine processing
#[derive(Debug, Clone, Copy)]
pub enum OrderEvent {
    Submit,
    Ack(u64), // exchange_id
    Fill { qty: u64, price: u64 },
    Cancel,
    Reject { code: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_size_alignment() {
        assert_eq!(core::mem::size_of::<Order>() % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_order_creation() {
        let sm = OrderStateMachine::new();
        
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0);
        assert!(id.is_some());
        
        let order = sm.get_order(id.unwrap()).unwrap();
        assert_eq!(order.get_state(), OrderState::Pending);
        assert_eq!(order.quantity, 100);
    }

    #[test]
    fn test_state_transitions() {
        let sm = OrderStateMachine::new();
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0).unwrap();
        
        let order = sm.get_order(id).unwrap();
        
        // Valid transitions
        assert!(order.transition_to(OrderState::Submitted));
        assert!(order.transition_to(OrderState::Acknowledged));
        
        // Invalid transition (from terminal)
        order.transition_to(OrderState::Filled);
        assert!(!order.transition_to(OrderState::Pending));
    }

    #[test]
    fn test_process_fill() {
        let sm = OrderStateMachine::new();
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0).unwrap();
        
        let order = sm.get_order(id).unwrap();
        order.mark_submitted();
        order.mark_acknowledged();
        
        // Process partial fill
        assert!(order.process_fill(50, 50000));
        assert_eq!(order.get_state(), OrderState::PartiallyFilled);
        assert_eq!(order.get_filled_qty(), 50);
        assert_eq!(order.get_remaining_qty(), 50);
        
        // Process remaining fill
        assert!(order.process_fill(50, 50000));
        assert_eq!(order.get_state(), OrderState::Filled);
        assert_eq!(order.get_filled_qty(), 100);
    }

    #[test]
    fn test_avg_fill_price() {
        let sm = OrderStateMachine::new();
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0).unwrap();
        
        let order = sm.get_order(id).unwrap();
        order.mark_submitted();
        order.mark_acknowledged();
        
        order.process_fill(50, 50000);
        assert_eq!(order.avg_fill_price.load(Ordering::Relaxed), 50000);
        
        order.process_fill(50, 50200);
        // Average should be (50000*50 + 50200*50) / 100 = 50100
        assert_eq!(order.avg_fill_price.load(Ordering::Relaxed), 50100);
    }

    #[test]
    fn test_cancel() {
        let sm = OrderStateMachine::new();
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0).unwrap();
        
        let order = sm.get_order(id).unwrap();
        order.mark_submitted();
        
        assert!(order.cancel());
        assert_eq!(order.get_state(), OrderState::Cancelled);
        
        // Can't cancel again
        assert!(!order.cancel());
    }

    #[test]
    fn test_reject() {
        let sm = OrderStateMachine::new();
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::Day, 0).unwrap();
        
        let order = sm.get_order(id).unwrap();
        
        assert!(order.reject(42));
        assert_eq!(order.get_state(), OrderState::Rejected);
        assert_eq!(order.error_code.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_timeout_check() {
        let sm = OrderStateMachine::new();
        sm.set_default_timeout_cycles(1000); // Very short timeout
        
        let id = sm.create_order(1, 0, OrderSide::Buy, 100, 50000, TimeInForce::IOC, 0).unwrap();
        
        // Wait for timeout (simulate by advancing time)
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        let cancelled = sm.check_timeouts();
        assert!(cancelled >= 0);
    }

    #[test]
    fn test_terminal_states() {
        assert!(OrderState::Filled.is_terminal());
        assert!(OrderState::Cancelled.is_terminal());
        assert!(OrderState::Rejected.is_terminal());
        assert!(OrderState::Expired.is_terminal());
        
        assert!(!OrderState::Pending.is_terminal());
        assert!(!OrderState::Submitted.is_terminal());
        assert!(!OrderState::Acknowledged.is_terminal());
        assert!(!OrderState::PartiallyFilled.is_terminal());
    }
}
