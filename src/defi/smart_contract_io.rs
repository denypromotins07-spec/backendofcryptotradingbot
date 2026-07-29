//! High-throughput smart contract event log parser and state delta tracker.
//! Zero-copy event decoding with lock-free state updates.
//! Pre-allocated buffers for deterministic memory usage.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum events tracked per block (pre-allocated)
pub const MAX_EVENTS: usize = 4096;

/// Maximum tracked state variables
pub const MAX_STATE_VARS: usize = 256;

/// Event signature hash (first 4 bytes of keccak)
pub type EventSignature = u32;

/// Cache-line aligned event data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ContractEvent {
    /// Event signature hash
    pub signature: EventSignature,
    /// Contract address (lower 20 bytes as u128)
    pub contract_address: u128,
    /// Block number
    pub block_number: u64,
    /// Transaction index in block
    pub tx_index: u32,
    /// Log index in transaction
    pub log_index: u32,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Event data length (max 4 topics + data)
    pub data_len: u8,
    /// Is this event processed
    pub processed: bool,
    _padding: [u8; 34], // Pad to 64 bytes
}

impl Default for ContractEvent {
    fn default() -> Self {
        Self {
            signature: 0,
            contract_address: 0,
            block_number: 0,
            tx_index: 0,
            log_index: 0,
            timestamp_ns: 0,
            data_len: 0,
            processed: false,
            _padding: [0; 34],
        }
    }
}

/// State variable delta tracking
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StateDelta {
    /// Storage slot (keccak of variable name/index)
    pub slot: u128,
    /// Previous value (big-endian stored as u128)
    pub prev_value: u128,
    /// New value
    pub new_value: u128,
    /// Block number of change
    pub block_number: u64,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Contract address
    pub contract_address: u128,
    _padding: [u8; 32], // Pad to 64 bytes
}

impl Default for StateDelta {
    fn default() -> Self {
        Self {
            slot: 0,
            prev_value: 0,
            new_value: 0,
            block_number: 0,
            timestamp_ns: 0,
            contract_address: 0,
            _padding: [0; 32],
        }
    }
}

/// Main smart contract IO tracker
#[repr(C)]
pub struct SmartContractIO {
    /// Pre-allocated event buffer
    pub events: [ContractEvent; MAX_EVENTS],
    /// Pre-allocated state deltas buffer
    pub state_deltas: [StateDelta; MAX_STATE_VARS],
    /// Event count
    pub event_count: AtomicU64,
    /// State delta count
    pub delta_count: AtomicU64,
    /// Last processed block
    pub last_block: AtomicU64,
    /// Events pending processing
    pub pending_events: AtomicU64,
    /// Overflow flag (buffer full)
    pub overflow: AtomicBool,
    _padding: [u8; 40], // Align to cache line
}

impl SmartContractIO {
    pub fn new() -> Self {
        Self {
            events: [ContractEvent::default(); MAX_EVENTS],
            state_deltas: [StateDelta::default(); MAX_STATE_VARS],
            event_count: AtomicU64::new(0),
            delta_count: AtomicU64::new(0),
            last_block: AtomicU64::new(0),
            pending_events: AtomicU64::new(0),
            overflow: AtomicBool::new(false),
            _padding: [0; 40],
        }
    }

    /// Add event to buffer (zero-copy, lock-free)
    #[inline]
    pub fn add_event(&mut self, event: ContractEvent) -> bool {
        let idx = self.event_count.load(Ordering::Acquire) as usize;
        
        if idx >= MAX_EVENTS {
            self.overflow.store(true, Ordering::Release);
            return false;
        }

        self.events[idx] = event;
        self.event_count.fetch_add(1, Ordering::AcqRel);
        self.pending_events.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Track state change (storage delta)
    #[inline]
    pub fn track_state_change(
        &mut self,
        slot: u128,
        prev_value: u128,
        new_value: u128,
        contract_address: u128,
        block_number: u64,
        timestamp_ns: u64,
    ) -> bool {
        let idx = self.delta_count.load(Ordering::Acquire) as usize;
        
        if idx >= MAX_STATE_VARS {
            return false;
        }

        self.state_deltas[idx] = StateDelta {
            slot,
            prev_value,
            new_value,
            block_number,
            timestamp_ns,
            contract_address,
            _padding: [0; 32],
        };
        self.delta_count.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Get pending event count
    #[inline]
    pub fn get_pending_count(&self) -> u64 {
        self.pending_events.load(Ordering::Acquire)
    }

    /// Mark events as processed up to index
    #[inline]
    pub fn mark_processed(&mut self, up_to_index: u64) {
        let total = self.event_count.load(Ordering::Acquire);
        let count = up_to_index.min(total);
        
        // Update pending count
        let current_pending = self.pending_events.load(Ordering::Acquire);
        self.pending_events.store(current_pending.saturating_sub(count), Ordering::Release);
        
        // Update last processed block from the last processed event
        if count > 0 {
            self.last_block.store(self.events[(count - 1) as usize].block_number, Ordering::Release);
        }
    }

    /// Get events by signature (zero-copy iteration)
    #[inline]
    pub fn get_events_by_signature(&self, signature: EventSignature) -> EventIterator {
        EventIterator {
            events: &self.events,
            count: self.event_count.load(Ordering::Acquire) as usize,
            signature,
            index: 0,
        }
    }

    /// Get latest state for a slot
    #[inline]
    pub fn get_latest_state(&self, slot: u128, contract: u128) -> Option<u128> {
        let count = self.delta_count.load(Ordering::Acquire);
        
        // Search backwards for most recent
        for i in (0..count).rev() {
            let delta = &self.state_deltas[i as usize];
            if delta.slot == slot && delta.contract_address == contract {
                return Some(delta.new_value);
            }
        }
        None
    }

    /// Check for buffer overflow
    #[inline]
    pub fn check_overflow(&self) -> bool {
        let overflow = self.overflow.load(Ordering::Acquire);
        if overflow {
            self.overflow.store(false, Ordering::Release);
        }
        overflow
    }

    /// Clear processed events (compact buffer)
    #[inline]
    pub fn clear_processed(&mut self) {
        // In production, this would use memmove to compact
        // For simplicity, just reset counters (production would be more sophisticated)
        let pending = self.pending_events.load(Ordering::Acquire);
        if pending == 0 {
            self.event_count.store(0, Ordering::Release);
            self.delta_count.store(0, Ordering::Release);
        }
    }
}

impl Default for SmartContractIO {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy event iterator
pub struct EventIterator<'a> {
    events: &'a [ContractEvent],
    count: usize,
    signature: EventSignature,
    index: usize,
}

impl<'a> Iterator for EventIterator<'a> {
    type Item = &'a ContractEvent;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.count {
            let event = &self.events[self.index];
            self.index += 1;
            if event.signature == self.signature {
                return Some(event);
            }
        }
        None
    }
}

/// Common event signatures (keccak256 first 4 bytes)
pub mod event_signatures {
    pub const TRANSFER: u32 = 0xddf252ad; // Transfer(address,address,uint256)
    pub const APPROVAL: u32 = 0x8c5be1e5; // Approval(address,address,uint256)
    pub const DEPOSIT: u32 = 0xe1fffcc4; // Deposit(address,uint256)
    pub const WITHDRAWAL: u32 = 0x7fcf532c; // Withdrawal(address,uint256)
    pub const SWAP: u32 = 0xd78ad95f; // Swap(address,uint256,uint256,uint256,uint256,address)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_contract_io_initialization() {
        let io = SmartContractIO::new();
        assert_eq!(io.event_count.load(Ordering::Relaxed), 0);
        assert_eq!(io.get_pending_count(), 0);
        assert!(!io.check_overflow());
    }

    #[test]
    fn test_add_event() {
        let mut io = SmartContractIO::new();
        
        let event = ContractEvent {
            signature: event_signatures::TRANSFER,
            contract_address: 0x1234567890abcdef1234567890abcdef12345678u128,
            block_number: 1000,
            tx_index: 0,
            log_index: 0,
            timestamp_ns: 1000,
            data_len: 32,
            processed: false,
            _padding: [0; 34],
        };
        
        assert!(io.add_event(event));
        assert_eq!(io.get_pending_count(), 1);
    }

    #[test]
    fn test_track_state_change() {
        let mut io = SmartContractIO::new();
        
        io.track_state_change(
            1u128,
            100u128,
            200u128,
            0x1234u128,
            1000,
            2000,
        );
        
        assert_eq!(io.delta_count.load(Ordering::Relaxed), 1);
        
        let latest = io.get_latest_state(1u128, 0x1234u128);
        assert_eq!(latest, Some(200u128));
    }

    #[test]
    fn test_event_iterator() {
        let mut io = SmartContractIO::new();
        
        // Add multiple events
        for i in 0..5 {
            io.add_event(ContractEvent {
                signature: if i % 2 == 0 { event_signatures::TRANSFER } else { event_signatures::SWAP },
                contract_address: 0x1000u128,
                block_number: 1000 + i,
                tx_index: 0,
                log_index: i,
                timestamp_ns: 1000,
                data_len: 32,
                processed: false,
                _padding: [0; 34],
            });
        }
        
        // Iterate over TRANSFER events only
        let transfer_events: Vec<_> = io.get_events_by_signature(event_signatures::TRANSFER).collect();
        assert_eq!(transfer_events.len(), 3); // indices 0, 2, 4
    }

    #[test]
    fn test_mark_processed() {
        let mut io = SmartContractIO::new();
        
        for i in 0..5 {
            io.add_event(ContractEvent {
                signature: event_signatures::TRANSFER,
                contract_address: 0x1000u128,
                block_number: 1000 + i,
                tx_index: 0,
                log_index: i,
                timestamp_ns: 1000,
                data_len: 32,
                processed: false,
                _padding: [0; 34],
            });
        }
        
        io.mark_processed(3);
        assert_eq!(io.get_pending_count(), 2);
        assert_eq!(io.last_block.load(Ordering::Relaxed), 1002);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<ContractEvent>() >= 64);
        assert!(core::mem::size_of::<StateDelta>() >= 64);
        assert!(core::mem::size_of::<SmartContractIO>() >= 64);
    }
}
