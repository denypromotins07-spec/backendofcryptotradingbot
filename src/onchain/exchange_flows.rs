//! Lock-free aggregator for CEX/DEX net inflows and outflows using streaming RPC.
//! 
//! Implements zero-copy flow aggregation with circular buffers for O(1) complexity
//! and lock-free atomic state management for ultra-low-latency updates.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// Cache line size for false sharing prevention
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of exchanges tracked
const MAX_EXCHANGES: usize = 32;

/// Circular buffer size for rolling calculations
const ROLLING_WINDOW_SIZE: usize = 512;

/// Fixed-point scale (6 decimal precision)
const FIXED_SCALE: u64 = 1_000_000;

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
    
    #[inline]
    fn fetch_sub(&self, delta: u64) -> u64 {
        self.value.fetch_sub(delta, Ordering::Relaxed)
    }
}

/// Padded atomic i64 for signed flow values
#[repr(C)]
struct PaddedAtomicI64 {
    value: AtomicI64,
    _padding: [u8; CACHE_LINE_SIZE - 8],
}

impl PaddedAtomicI64 {
    const fn new(val: i64) -> Self {
        Self {
            value: AtomicI64::new(val),
            _padding: [0u8; CACHE_LINE_SIZE - 8],
        }
    }
    
    #[inline]
    fn load(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }
    
    #[inline]
    fn fetch_add(&self, delta: i64) -> i64 {
        self.value.fetch_add(delta, Ordering::Relaxed)
    }
}

/// Exchange flow data - cache-line aligned
#[repr(C)]
struct ExchangeFlowData {
    /// Exchange identifier hash
    exchange_id: u64,
    /// Total inflows (fixed-point)
    total_inflows_fixed: PaddedAtomicU64,
    /// Total outflows (fixed-point)
    total_outflows_fixed: PaddedAtomicU64,
    /// Net flow (inflows - outflows, fixed-point, signed)
    net_flow_fixed: PaddedAtomicI64,
    /// Transaction count
    tx_count: PaddedAtomicU64,
    /// Last update timestamp (CPU cycles)
    last_update_cycles: PaddedAtomicU64,
    /// Is CEX (vs DEX)
    is_cex: bool,
    /// Is active
    is_active: bool,
    /// Padding to reach 64 bytes for remaining fields
    _padding: [u8; 62],
}

const _: () = assert!(core::mem::size_of::<ExchangeFlowData>() == 192);

/// Circular buffer for rolling window flow aggregation
#[repr(C)]
struct FlowRollingBuffer {
    /// Pre-allocated buffer storage
    buffer: [i64; ROLLING_WINDOW_SIZE],
    /// Head index (atomic for lock-free access)
    head: AtomicU64,
    /// Current sum (fixed-point, signed)
    sum_fixed: AtomicI64,
    /// Window size
    window_size: usize,
    /// Padding
    _padding: [u8; CACHE_LINE_SIZE - 20],
}

impl FlowRollingBuffer {
    const fn new(window_size: usize) -> Self {
        Self {
            buffer: [0i64; ROLLING_WINDOW_SIZE],
            head: AtomicU64::new(0),
            sum_fixed: AtomicI64::new(0),
            window_size,
            _padding: [0u8; CACHE_LINE_SIZE - 20],
        }
    }
    
    /// Push a new flow value with O(1) complexity
    #[inline]
    pub fn push(&self, flow_fixed: i64) -> i64 {
        let head = self.head.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = head % self.window_size;
        
        // Get old value being replaced (branchless)
        let old_value = unsafe { *self.buffer.get_unchecked(idx) };
        
        // Update sum (branchless delta calculation)
        let delta = flow_fixed - old_value;
        let new_sum = self.sum_fixed.fetch_add(delta, Ordering::Relaxed) + delta;
        
        // Store new value
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = flow_fixed;
        }
        
        new_sum
    }
    
    /// Get current rolling sum
    #[inline]
    pub fn get_sum(&self) -> i64 {
        self.sum_fixed.load(Ordering::Relaxed)
    }
    
    /// Get rolling average
    #[inline]
    pub fn get_average(&self) -> i64 {
        let sum = self.get_sum();
        let count = core::cmp::min(self.head.load(Ordering::Relaxed) as usize, self.window_size);
        if count == 0 { return 0; }
        sum / count as i64
    }
    
    /// Reset buffer
    #[inline]
    pub fn reset(&self) {
        self.head.store(0, Ordering::Relaxed);
        self.sum_fixed.store(0, Ordering::Relaxed);
    }
}

/// Flow direction enum (encoded as integers for branchless ops)
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum FlowDirection {
    Inflow = 0,
    Outflow = 1,
}

/// Streaming RPC flow record (zero-copy compatible)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RpcFlowRecord {
    /// Source address hash
    source_hash: u64,
    /// Destination address hash
    dest_hash: u64,
    /// Amount in fixed-point
    amount_fixed: u64,
    /// Timestamp (CPU cycles)
    timestamp_cycles: u64,
    /// Exchange ID
    exchange_id: u16,
    /// Chain ID
    chain_id: u8,
    /// Flow direction (0=inflow, 1=outflow)
    direction: u8,
    /// Is DEX flag
    is_dex: bool,
    /// Token type (0=native, 1=ERC20, 2=SPL)
    token_type: u8,
    /// Padding
    _padding: [u8; 27],
}

const _: () = assert!(core::mem::size_of::<RpcFlowRecord>() == 64);

/// Main exchange flows aggregator
#[repr(C)]
pub struct ExchangeFlowsAggregator {
    /// Exchange flow data array (pre-allocated)
    exchanges: [ExchangeFlowData; MAX_EXCHANGES],
    /// Global net flow (all exchanges combined)
    global_net_flow_fixed: AtomicI64,
    /// Global inflow total
    global_inflow_fixed: PaddedAtomicU64,
    /// Global outflow total
    global_outflow_fixed: PaddedAtomicU64,
    /// Rolling buffer for net flow trends
    net_flow_rolling: FlowRollingBuffer,
    /// Rolling buffer for inflow trends
    inflow_rolling: FlowRollingBuffer,
    /// Rolling buffer for outflow trends
    outflow_rolling: FlowRollingBuffer,
    /// Active exchange count
    active_count: AtomicU64,
    /// Stream sequence number
    sequence_number: AtomicU64,
    /// Gap detection enabled
    gap_detection_enabled: AtomicBool,
    /// Last valid sequence number
    last_sequence: AtomicU64,
    /// Gap detected flag
    gap_detected: AtomicBool,
}

impl ExchangeFlowsAggregator {
    /// Create a new exchange flows aggregator
    pub const fn new() -> Self {
        // Initialize exchanges array
        let exchanges = [ExchangeFlowData {
            exchange_id: 0,
            total_inflows_fixed: PaddedAtomicU64::new(0),
            total_outflows_fixed: PaddedAtomicU64::new(0),
            net_flow_fixed: PaddedAtomicI64::new(0),
            tx_count: PaddedAtomicU64::new(0),
            last_update_cycles: PaddedAtomicU64::new(0),
            is_cex: false,
            is_active: false,
            _padding: [0u8; 62],
        }; MAX_EXCHANGES];
        
        Self {
            exchanges,
            global_net_flow_fixed: AtomicI64::new(0),
            global_inflow_fixed: PaddedAtomicU64::new(0),
            global_outflow_fixed: PaddedAtomicU64::new(0),
            net_flow_rolling: FlowRollingBuffer::new(ROLLING_WINDOW_SIZE),
            inflow_rolling: FlowRollingBuffer::new(ROLLING_WINDOW_SIZE),
            outflow_rolling: FlowRollingBuffer::new(ROLLING_WINDOW_SIZE),
            active_count: AtomicU64::new(0),
            sequence_number: AtomicU64::new(0),
            gap_detection_enabled: AtomicBool::new(true),
            last_sequence: AtomicU64::new(0),
            gap_detected: AtomicBool::new(false),
        }
    }
    
    /// Process a flow record from streaming RPC (lock-free, zero-copy)
    #[inline]
    pub fn process_flow_record(&self, record: &RpcFlowRecord) -> bool {
        // Sequence number tracking for gap detection
        let seq = self.sequence_number.fetch_add(1, Ordering::Relaxed);
        
        // Gap detection (branchless)
        if self.gap_detection_enabled.load(Ordering::Relaxed) {
            let expected_seq = self.last_sequence.load(Ordering::Relaxed) + 1;
            let gap_exists = ((seq != expected_seq) as u8) != 0;
            self.gap_detected.store(gap_exists != 0, Ordering::Relaxed);
        }
        self.last_sequence.store(seq, Ordering::Relaxed);
        
        // Determine flow sign based on direction (branchless)
        let flow_signed = if record.direction == FlowDirection::Inflow as u8 {
            record.amount_fixed as i64
        } else {
            -(record.amount_fixed as i64)
        };
        
        // Update exchange-specific flows
        self.update_exchange_flow(record.exchange_id, record.amount_fixed, record.direction, record.is_dex);
        
        // Update global aggregates
        self.update_global_flows(flow_signed, record.amount_fixed, record.direction);
        
        // Update rolling windows
        self.net_flow_rolling.push(flow_signed);
        if record.direction == FlowDirection::Inflow as u8 {
            self.inflow_rolling.push(record.amount_fixed as i64);
        } else {
            self.outflow_rolling.push(record.amount_fixed as i64);
        }
        
        true
    }
    
    /// Update exchange-specific flow data (lock-free)
    #[inline]
    fn update_exchange_flow(&self, exchange_id: u16, amount_fixed: u64, direction: u8, is_dex: bool) {
        let idx = (exchange_id as usize) % MAX_EXCHANGES;
        
        unsafe {
            let exchange = &mut *self.exchanges.get_unchecked_mut(idx);
            
            // Mark as active if first time
            let was_inactive = !exchange.is_active;
            exchange.is_active |= was_inactive;
            exchange.is_cex = !is_dex;
            exchange.exchange_id = exchange_id as u64;
            
            // Branchless inflow/outflow update
            let is_inflow = (direction == FlowDirection::Inflow as u8) as u64;
            let is_outflow = 1 - is_inflow;
            
            // Update totals (branchless multiplication)
            let inflow_delta = amount_fixed * is_inflow;
            let outflow_delta = amount_fixed * is_outflow;
            
            exchange.total_inflows_fixed.fetch_add(inflow_delta);
            exchange.total_outflows_fixed.fetch_add(outflow_delta);
            
            // Update net flow
            let net_delta = if is_inflow != 0 {
                amount_fixed as i64
            } else {
                -(amount_fixed as i64)
            };
            exchange.net_flow_fixed.fetch_add(net_delta);
            
            // Increment transaction count
            exchange.tx_count.fetch_add(1);
        }
    }
    
    /// Update global flow aggregates
    #[inline]
    fn update_global_flows(&self, flow_signed: i64, amount_fixed: u64, direction: u8) {
        // Update global net flow
        self.global_net_flow_fixed.fetch_add(flow_signed, Ordering::Relaxed);
        
        // Branchless inflow/outflow update
        let is_inflow = (direction == FlowDirection::Inflow as u8) as u64;
        let is_outflow = 1 - is_inflow;
        
        self.global_inflow_fixed.fetch_add(amount_fixed * is_inflow);
        self.global_outflow_fixed.fetch_add(amount_fixed * is_outflow);
    }
    
    /// Get net flow for a specific exchange
    #[inline]
    pub fn get_exchange_net_flow(&self, exchange_id: u16) -> i64 {
        let idx = (exchange_id as usize) % MAX_EXCHANGES;
        unsafe {
            self.exchanges.get_unchecked(idx).net_flow_fixed.load()
        }
    }
    
    /// Get global net flow
    #[inline]
    pub fn get_global_net_flow(&self) -> i64 {
        self.global_net_flow_fixed.load(Ordering::Relaxed)
    }
    
    /// Get rolling average net flow
    #[inline]
    pub fn get_rolling_avg_net_flow(&self) -> i64 {
        self.net_flow_rolling.get_average()
    }
    
    /// Get inflow/outflow ratio (fixed-point, scaled by 1e6)
    #[inline]
    pub fn get_flow_ratio(&self) -> u64 {
        let inflows = self.global_inflow_fixed.load();
        let outflows = self.global_outflow_fixed.load();
        
        if outflows == 0 {
            return u64::MAX; // Infinite ratio (only inflows)
        }
        
        (inflows * FIXED_SCALE) / outflows
    }
    
    /// Check if gap was detected in stream
    #[inline]
    pub fn is_gap_detected(&self) -> bool {
        self.gap_detected.load(Ordering::Relaxed)
    }
    
    /// Reset gap detection flag
    #[inline]
    pub fn reset_gap_flag(&self) {
        self.gap_detected.store(false, Ordering::Relaxed);
    }
    
    /// Get total transaction count across all exchanges
    #[inline]
    pub fn get_total_tx_count(&self) -> u64 {
        let mut total = 0u64;
        for i in 0..MAX_EXCHANGES {
            unsafe {
                total += self.exchanges.get_unchecked(i).tx_count.load();
            }
        }
        total
    }
}

impl Default for ExchangeFlowsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_inflow_processing() {
        let aggregator = ExchangeFlowsAggregator::new();
        
        let inflow_record = RpcFlowRecord {
            source_hash: 0x1111111111111111,
            dest_hash: 0x2222222222222222,
            amount_fixed: 1_000_000_000_000, // 1M
            timestamp_cycles: 0,
            exchange_id: 1,
            chain_id: 1,
            direction: FlowDirection::Inflow as u8,
            is_dex: false,
            token_type: 1,
            _padding: [0u8; 27],
        };
        
        aggregator.process_flow_record(&inflow_record);
        
        assert_eq!(aggregator.get_global_net_flow(), 1_000_000_000_000);
        assert_eq!(aggregator.get_exchange_net_flow(1), 1_000_000_000_000);
    }
    
    #[test]
    fn test_outflow_processing() {
        let aggregator = ExchangeFlowsAggregator::new();
        
        let outflow_record = RpcFlowRecord {
            source_hash: 0x3333333333333333,
            dest_hash: 0x4444444444444444,
            amount_fixed: 500_000_000_000, // 500k
            timestamp_cycles: 0,
            exchange_id: 2,
            chain_id: 1,
            direction: FlowDirection::Outflow as u8,
            is_dex: true,
            token_type: 1,
            _padding: [0u8; 27],
        };
        
        aggregator.process_flow_record(&outflow_record);
        
        assert_eq!(aggregator.get_global_net_flow(), -500_000_000_000);
    }
    
    #[test]
    fn test_net_flow_calculation() {
        let aggregator = ExchangeFlowsAggregator::new();
        
        // Add inflow
        let inflow = RpcFlowRecord {
            source_hash: 0xAAAA,
            dest_hash: 0xBBBB,
            amount_fixed: 2_000_000_000_000,
            timestamp_cycles: 0,
            exchange_id: 1,
            chain_id: 1,
            direction: FlowDirection::Inflow as u8,
            is_dex: false,
            token_type: 0,
            _padding: [0u8; 27],
        };
        
        // Add outflow
        let outflow = RpcFlowRecord {
            source_hash: 0xCCCC,
            dest_hash: 0xDDDD,
            amount_fixed: 800_000_000_000,
            timestamp_cycles: 0,
            exchange_id: 1,
            chain_id: 1,
            direction: FlowDirection::Outflow as u8,
            is_dex: false,
            token_type: 0,
            _padding: [0u8; 27],
        };
        
        aggregator.process_flow_record(&inflow);
        aggregator.process_flow_record(&outflow);
        
        assert_eq!(aggregator.get_global_net_flow(), 1_200_000_000_000);
        assert_eq!(aggregator.get_flow_ratio(), (2_000_000_000_000 * FIXED_SCALE) / 800_000_000_000);
    }
    
    #[test]
    fn test_rolling_window() {
        let aggregator = ExchangeFlowsAggregator::new();
        
        for i in 0..10 {
            let record = RpcFlowRecord {
                source_hash: i as u64,
                dest_hash: (i + 1) as u64,
                amount_fixed: 100_000_000_000,
                timestamp_cycles: 0,
                exchange_id: 1,
                chain_id: 1,
                direction: FlowDirection::Inflow as u8,
                is_dex: true,
                token_type: 1,
                _padding: [0u8; 27],
            };
            aggregator.process_flow_record(&record);
        }
        
        let avg = aggregator.get_rolling_avg_net_flow();
        assert_eq!(avg, 100_000_000_000);
    }
}
