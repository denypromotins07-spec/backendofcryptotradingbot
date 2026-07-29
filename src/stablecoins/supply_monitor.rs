//! Cross-chain USDT/USDC supply and flow tracking across Ethereum, Solana, and Tron.
//! 
//! Uses fixed-point arithmetic, lock-free atomic operations, and pre-allocated
//! buffers for zero-copy cross-chain supply monitoring.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of chains tracked
const MAX_CHAINS: usize = 16;

/// Maximum number of token contracts per chain
const MAX_CONTRACTS_PER_CHAIN: usize = 32;

/// Fixed-point scale (6 decimal precision)
const FIXED_SCALE: u64 = 1_000_000;

/// Chain IDs
const CHAIN_ETHEREUM: u8 = 1;
const CHAIN_SOLANA: u8 = 2;
const CHAIN_TRON: u8 = 3;

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

/// Token contract info - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct TokenContract {
    /// Contract address hash
    address_hash: u64,
    /// Token type (0=USDT, 1=USDC, 2=other)
    token_type: u8,
    /// Chain ID
    chain_id: u8,
    /// Decimals
    decimals: u8,
    /// Is active
    is_active: bool,
    /// Current supply (fixed-point)
    supply_fixed: u64,
    /// 24h change (fixed-point, signed stored as u64 with sign bit)
    change_24h_fixed: i64,
    /// Last update timestamp
    last_update_cycles: u64,
    /// Padding to reach 64 bytes
    _padding: [u8; 34],
}

const _: () = assert!(core::mem::size_of::<TokenContract>() == 64);

/// Chain-level aggregate data - cache-line aligned
#[repr(C)]
struct ChainAggregate {
    /// Chain ID
    chain_id: u8,
    /// Padding
    _pad1: [u8; 7],
    /// Total USDT supply on chain (fixed-point)
    usdt_supply_fixed: PaddedAtomicU64,
    /// Total USDC supply on chain (fixed-point)
    usdc_supply_fixed: PaddedAtomicU64,
    /// 24h inflow (fixed-point)
    inflow_24h_fixed: PaddedAtomicU64,
    /// 24h outflow (fixed-point)
    outflow_24h_fixed: PaddedAtomicU64,
    /// Contract count
    contract_count: AtomicU64,
}

impl ChainAggregate {
    const fn new(chain_id: u8) -> Self {
        Self {
            chain_id,
            _pad1: [0u8; 7],
            usdt_supply_fixed: PaddedAtomicU64::new(0),
            usdc_supply_fixed: PaddedAtomicU64::new(0),
            inflow_24h_fixed: PaddedAtomicU64::new(0),
            outflow_24h_fixed: PaddedAtomicU64::new(0),
            contract_count: AtomicU64::new(0),
        }
    }
}

/// Circular buffer for rolling flow calculations
#[repr(C)]
struct FlowBuffer {
    /// Pre-allocated buffer
    buffer: [i64; 288], // 24 hours at 5-minute intervals
    /// Head index
    head: AtomicU64,
    /// Sum
    sum_fixed: AtomicI64,
    /// Window size
    window_size: usize,
}

impl FlowBuffer {
    const fn new(window_size: usize) -> Self {
        Self {
            buffer: [0i64; 288],
            head: AtomicU64::new(0),
            sum_fixed: AtomicI64::new(0),
            window_size,
        }
    }
    
    #[inline]
    pub fn push(&self, flow_fixed: i64) -> i64 {
        let head = self.head.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = head % self.window_size;
        
        let old = unsafe { *self.buffer.get_unchecked(idx) };
        let delta = flow_fixed - old;
        
        let new_sum = self.sum_fixed.fetch_add(delta, Ordering::Relaxed) + delta;
        
        unsafe {
            *self.buffer.get_unchecked_mut(idx) = flow_fixed;
        }
        
        new_sum
    }
    
    #[inline]
    pub fn get_net_flow(&self) -> i64 {
        self.sum_fixed.load(Ordering::Relaxed)
    }
}

/// Main stablecoin supply monitor
#[repr(C)]
pub struct StablecoinSupplyMonitor {
    /// Token contracts (pre-allocated)
    contracts: [TokenContract; MAX_CHAINS * MAX_CONTRACTS_PER_CHAIN],
    /// Chain aggregates
    chain_aggregates: [ChainAggregate; MAX_CHAINS],
    /// Global USDT supply (fixed-point)
    global_usdt_supply_fixed: PaddedAtomicU64,
    /// Global USDC supply (fixed-point)
    global_usdc_supply_fixed: PaddedAtomicU64,
    /// Rolling flow buffer
    flow_buffer: FlowBuffer,
    /// Contract count
    contract_count: AtomicU64,
    /// Active chain count
    active_chains: AtomicU64,
    /// Last sync timestamp
    last_sync_cycles: AtomicU64,
    /// Is ready
    is_ready: AtomicBool,
}

impl StablecoinSupplyMonitor {
    /// Create a new supply monitor
    pub const fn new() -> Self {
        Self {
            contracts: [TokenContract {
                address_hash: 0,
                token_type: 0,
                chain_id: 0,
                decimals: 6,
                is_active: false,
                supply_fixed: 0,
                change_24h_fixed: 0,
                last_update_cycles: 0,
                _padding: [0u8; 34],
            }; MAX_CHAINS * MAX_CONTRACTS_PER_CHAIN],
            chain_aggregates: [ChainAggregate::new(0); MAX_CHAINS],
            global_usdt_supply_fixed: PaddedAtomicU64::new(0),
            global_usdc_supply_fixed: PaddedAtomicU64::new(0),
            flow_buffer: FlowBuffer::new(288),
            contract_count: AtomicU64::new(0),
            active_chains: AtomicU64::new(0),
            last_sync_cycles: AtomicU64::new(0),
            is_ready: AtomicBool::new(false),
        }
    }
    
    /// Register a token contract
    #[inline]
    pub fn register_contract(&self, contract: TokenContract) -> bool {
        let idx = self.contract_count.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= MAX_CHAINS * MAX_CONTRACTS_PER_CHAIN {
            self.contract_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            *self.contracts.get_unchecked_mut(idx) = contract;
        }
        
        // Update chain aggregate
        self.update_chain_aggregate(contract.chain_id, contract.token_type, contract.supply_fixed);
        
        true
    }
    
    /// Update chain-level aggregate
    #[inline]
    fn update_chain_aggregate(&self, chain_id: u8, token_type: u8, supply_fixed: u64) {
        let chain_idx = (chain_id as usize).min(MAX_CHAINS - 1);
        
        unsafe {
            let agg = &mut *self.chain_aggregates.get_unchecked_mut(chain_idx);
            agg.chain_id = chain_id;
            
            match token_type {
                0 => agg.usdt_supply_fixed.fetch_add(supply_fixed),
                1 => agg.usdc_supply_fixed.fetch_add(supply_fixed),
                _ => {}
            }
        }
        
        // Update global totals
        match token_type {
            0 => self.global_usdt_supply_fixed.fetch_add(supply_fixed),
            1 => self.global_usdc_supply_fixed.fetch_add(supply_fixed),
            _ => {}
        }
    }
    
    /// Update contract supply
    #[inline]
    pub fn update_supply(&self, address_hash: u64, new_supply_fixed: u64) {
        for i in 0..self.contract_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let contract = &mut *self.contracts.get_unchecked_mut(i);
                if contract.address_hash == address_hash {
                    let old_supply = contract.supply_fixed;
                    let delta = if new_supply_fixed >= old_supply {
                        new_supply_fixed - old_supply
                    } else {
                        0
                    };
                    
                    contract.supply_fixed = new_supply_fixed;
                    
                    // Record flow
                    let flow_signed = if new_supply_fixed >= old_supply {
                        delta as i64
                    } else {
                        -(old_supply - new_supply_fixed) as i64
                    };
                    self.flow_buffer.push(flow_signed);
                    
                    break;
                }
            }
        }
    }
    
    /// Get total supply for a token type
    #[inline]
    pub fn get_total_supply(&self, token_type: u8) -> u64 {
        match token_type {
            0 => self.global_usdt_supply_fixed.load(),
            1 => self.global_usdc_supply_fixed.load(),
            _ => 0,
        }
    }
    
    /// Get supply on specific chain
    #[inline]
    pub fn get_chain_supply(&self, chain_id: u8, token_type: u8) -> u64 {
        let chain_idx = (chain_id as usize).min(MAX_CHAINS - 1);
        unsafe {
            let agg = self.chain_aggregates.get_unchecked(chain_idx);
            match token_type {
                0 => agg.usdt_supply_fixed.load(),
                1 => agg.usdc_supply_fixed.load(),
                _ => 0,
            }
        }
    }
    
    /// Get net flow over rolling window
    #[inline]
    pub fn get_net_flow(&self) -> i64 {
        self.flow_buffer.get_net_flow()
    }
    
    /// Get USDT/USDC ratio (fixed-point, scaled by 1e6)
    #[inline]
    pub fn get_stablecoin_ratio(&self) -> u64 {
        let usdt = self.global_usdt_supply_fixed.load();
        let usdc = self.global_usdc_supply_fixed.load();
        
        if usdc == 0 { return u64::MAX; }
        (usdt * FIXED_SCALE) / usdc
    }
    
    /// Mark as synced
    #[inline]
    pub fn mark_synced(&self) {
        use core::arch::x86_64::_rdtsc;
        unsafe {
            self.last_sync_cycles.store(_rdtsc(), Ordering::Relaxed);
        }
        self.is_ready.store(true, Ordering::Relaxed);
    }
    
    /// Check if ready
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(Ordering::Relaxed)
    }
}

impl Default for StablecoinSupplyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_register_contract() {
        let monitor = StablecoinSupplyMonitor::new();
        
        let contract = TokenContract {
            address_hash: 0x1234567890ABCDEF,
            token_type: 0, // USDT
            chain_id: CHAIN_ETHEREUM,
            decimals: 6,
            is_active: true,
            supply_fixed: 50_000_000_000_000, // 50B
            change_24h_fixed: 0,
            last_update_cycles: 0,
            _padding: [0u8; 34],
        };
        
        assert!(monitor.register_contract(contract));
        
        let total = monitor.get_total_supply(0);
        assert_eq!(total, 50_000_000_000_000);
    }
    
    #[test]
    fn test_cross_chain_supply() {
        let monitor = StablecoinSupplyMonitor::new();
        
        // Ethereum USDT
        let eth_usdt = TokenContract {
            address_hash: 0x1111,
            token_type: 0,
            chain_id: CHAIN_ETHEREUM,
            decimals: 6,
            is_active: true,
            supply_fixed: 40_000_000_000_000,
            change_24h_fixed: 0,
            last_update_cycles: 0,
            _padding: [0u8; 34],
        };
        
        // Tron USDT
        let tron_usdt = TokenContract {
            address_hash: 0x2222,
            token_type: 0,
            chain_id: CHAIN_TRON,
            decimals: 6,
            is_active: true,
            supply_fixed: 50_000_000_000_000,
            change_24h_fixed: 0,
            last_update_cycles: 0,
            _padding: [0u8; 34],
        };
        
        monitor.register_contract(eth_usdt);
        monitor.register_contract(tron_usdt);
        
        assert_eq!(monitor.get_chain_supply(CHAIN_ETHEREUM, 0), 40_000_000_000_000);
        assert_eq!(monitor.get_chain_supply(CHAIN_TRON, 0), 50_000_000_000_000);
        assert_eq!(monitor.get_total_supply(0), 90_000_000_000_000);
    }
}
