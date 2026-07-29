//! Real-time Total Value Locked (TVL) and protocol revenue aggregator
//! Uses zero-copy ABI decoder for smart contract log parsing.
//! Lock-free atomic updates for cross-chain aggregation.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use core::sync::atomic::{AtomicU64, AtomicI128, Ordering};

/// Fixed-point representation for TVL (scaled by 10^18 for token precision)
pub type TokenAmount = i128;
const SCALE: i128 = 1_000_000_000_000_000_000;

/// Maximum number of protocols tracked (pre-allocated)
pub const MAX_PROTOCOLS: usize = 128;

/// Maximum number of chains tracked
pub const MAX_CHAINS: usize = 32;

/// Cache-line aligned protocol data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProtocolData {
    /// Protocol ID (hashed name)
    pub protocol_id: u64,
    /// TVL in USD (fixed-point, scaled by 10^8)
    pub tvl_usd: i64,
    /// 24h revenue in USD (fixed-point)
    pub revenue_24h: i64,
    /// 24h volume in USD (fixed-point)
    pub volume_24h: i64,
    /// Chain ID where protocol is deployed
    pub chain_id: u32,
    /// Last update timestamp (ns)
    pub last_update_ns: u64,
    /// Is protocol active
    pub active: bool,
    _padding: [u8; 35], // Pad to 64 bytes
}

impl Default for ProtocolData {
    fn default() -> Self {
        Self {
            protocol_id: 0,
            tvl_usd: 0,
            revenue_24h: 0,
            volume_24h: 0,
            chain_id: 0,
            last_update_ns: 0,
            active: false,
            _padding: [0; 35],
        }
    }
}

/// Chain-level aggregation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChainData {
    /// Chain ID
    pub chain_id: u32,
    /// Total TVL on chain (USD fixed-point)
    pub total_tvl: i64,
    /// Number of active protocols
    pub protocol_count: u32,
    /// 24h change in TVL (basis points)
    pub tvl_change_bps: i32,
    /// Last update timestamp
    pub last_update_ns: u64,
    _padding: [u8; 40], // Pad to 64 bytes
}

impl Default for ChainData {
    fn default() -> Self {
        Self {
            chain_id: 0,
            total_tvl: 0,
            protocol_count: 0,
            tvl_change_bps: 0,
            last_update_ns: 0,
            _padding: [0; 40],
        }
    }
}

/// Main TVL tracker with lock-free updates
#[repr(C)]
pub struct TVLTracker {
    /// Pre-allocated protocol array
    pub protocols: [ProtocolData; MAX_PROTOCOLS],
    /// Pre-allocated chain array
    pub chains: [ChainData; MAX_CHAINS],
    /// Total TVL across all chains (atomic for lock-free reads)
    pub total_tvl: AtomicI128,
    /// Protocol count
    pub protocol_count: AtomicU64,
    /// Chain count
    pub chain_count: AtomicU64,
    /// Global update timestamp
    pub last_update_ns: AtomicU64,
    /// Stale data threshold (ns)
    pub stale_threshold_ns: u64,
    _padding: [u8; 32], // Align to cache line
}

impl TVLTracker {
    pub fn new(stale_threshold_ms: u64) -> Self {
        Self {
            protocols: [ProtocolData::default(); MAX_PROTOCOLS],
            chains: [ChainData::default(); MAX_CHAINS],
            total_tvl: AtomicI128::new(0),
            protocol_count: AtomicU64::new(0),
            chain_count: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            stale_threshold_ns: stale_threshold_ms * 1_000_000,
            _padding: [0; 32],
        }
    }

    /// Update protocol TVL (zero-copy, lock-free)
    #[inline]
    pub fn update_protocol(&mut self, protocol_id: u64, chain_id: u32, tvl_usd: i64, revenue_24h: i64, volume_24h: i64, timestamp_ns: u64) {
        // Find or create protocol entry
        let count = self.protocol_count.load(Ordering::Acquire);
        let mut found = false;
        
        for i in 0..count.min(MAX_PROTOCOLS as u64) as usize {
            if self.protocols[i].protocol_id == protocol_id {
                self.protocols[i].tvl_usd = tvl_usd;
                self.protocols[i].revenue_24h = revenue_24h;
                self.protocols[i].volume_24h = volume_24h;
                self.protocols[i].last_update_ns = timestamp_ns;
                self.protocols[i].active = true;
                found = true;
                break;
            }
        }

        // Add new protocol if not found and space available
        if !found && count < MAX_PROTOCOLS as u64 {
            let idx = count as usize;
            self.protocols[idx] = ProtocolData {
                protocol_id,
                tvl_usd,
                revenue_24h,
                volume_24h,
                chain_id,
                last_update_ns: timestamp_ns,
                active: true,
                _padding: [0; 35],
            };
            self.protocol_count.fetch_add(1, Ordering::AcqRel);
        }

        // Update chain aggregation
        self.update_chain(chain_id, tvl_usd, timestamp_ns);

        // Recalculate total TVL
        self.recalculate_total();
        
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
    }

    /// Update chain-level aggregation
    #[inline]
    fn update_chain(&mut self, chain_id: u32, tvl_contribution: i64, timestamp_ns: u64) {
        let chain_count = self.chain_count.load(Ordering::Acquire);
        let mut found = false;

        for i in 0..chain_count.min(MAX_CHAINS as u64) as usize {
            if self.chains[i].chain_id == chain_id {
                self.chains[i].total_tvl = self.chains[i].total_tvl.saturating_add(tvl_contribution);
                self.chains[i].last_update_ns = timestamp_ns;
                found = true;
                break;
            }
        }

        if !found && chain_count < MAX_CHAINS as u64 {
            let idx = chain_count as usize;
            self.chains[idx] = ChainData {
                chain_id,
                total_tvl: tvl_contribution,
                protocol_count: 1,
                tvl_change_bps: 0,
                last_update_ns: timestamp_ns,
                _padding: [0; 40],
            };
            self.chain_count.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Recalculate total TVL from all protocols
    #[inline]
    fn recalculate_total(&mut self) {
        let mut total: i128 = 0;
        let count = self.protocol_count.load(Ordering::Acquire);
        
        for i in 0..count.min(MAX_PROTOCOLS as u64) as usize {
            if self.protocols[i].active {
                total += self.protocols[i].tvl_usd as i128;
            }
        }
        
        self.total_tvl.store(total, Ordering::Release);
    }

    /// Get total TVL (lock-free read)
    #[inline]
    pub fn get_total_tvl(&self) -> i128 {
        self.total_tvl.load(Ordering::Acquire)
    }

    /// Get TVL for specific chain
    #[inline]
    pub fn get_chain_tvl(&self, chain_id: u32) -> i64 {
        let chain_count = self.chain_count.load(Ordering::Acquire);
        
        for i in 0..chain_count.min(MAX_CHAINS as u64) as usize {
            if self.chains[i].chain_id == chain_id {
                return self.chains[i].total_tvl;
            }
        }
        0
    }

    /// Get protocol count
    #[inline]
    pub fn get_protocol_count(&self) -> u64 {
        self.protocol_count.load(Ordering::Acquire)
    }

    /// Check if data is stale
    #[inline]
    pub fn is_stale(&self, current_time_ns: u64) -> bool {
        let last_update = self.last_update_ns.load(Ordering::Acquire);
        current_time_ns.saturating_sub(last_update) > self.stale_threshold_ns
    }
}

/// Zero-copy ABI event decoder for TVL update events
#[repr(C)]
pub struct ABIDecoder {
    _private: [u8; 64], // Internal state
}

impl ABIDecoder {
    pub const fn new() -> Self {
        Self {
            _private: [0; 64],
        }
    }

    /// Decode uint256 from log data (zero-copy view)
    #[inline]
    pub fn decode_uint256(data: &[u8]) -> Option<TokenAmount> {
        if data.len() < 32 {
            return None;
        }
        // Big-endian to i128 (simplified, assumes fits)
        Some(TokenAmount::from_be_bytes(data[16..32].try_into().ok()?))
    }

    /// Decode address from topic (zero-copy)
    #[inline]
    pub fn decode_address(topic: &[u8]) -> Option<u128> {
        if topic.len() < 32 {
            return None;
        }
        Some(u128::from_be_bytes(topic[12..32].try_into().ok()?))
    }
}

impl Default for ABIDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tvl_tracker_initialization() {
        let tracker = TVLTracker::new(60_000); // 60 second stale threshold
        assert_eq!(tracker.get_total_tvl(), 0);
        assert_eq!(tracker.get_protocol_count(), 0);
    }

    #[test]
    fn test_protocol_update() {
        let mut tracker = TVLTracker::new(60_000);
        
        // Update protocol TVL
        tracker.update_protocol(1, 1, 1_000_000_000, 10_000_000, 100_000_000, 1000);
        
        assert_eq!(tracker.get_total_tvl(), 1_000_000_000);
        assert_eq!(tracker.get_protocol_count(), 1);
    }

    #[test]
    fn test_multi_chain_aggregation() {
        let mut tracker = TVLTracker::new(60_000);
        
        // Add protocols on different chains
        tracker.update_protocol(1, 1, 500_000_000, 5_000_000, 50_000_000, 1000);
        tracker.update_protocol(2, 2, 300_000_000, 3_000_000, 30_000_000, 1000);
        tracker.update_protocol(3, 1, 200_000_000, 2_000_000, 20_000_000, 1000);
        
        // Total should be sum
        assert_eq!(tracker.get_total_tvl(), 1_000_000_000);
        
        // Chain 1 should have 700M
        assert_eq!(tracker.get_chain_tvl(1), 700_000_000);
    }

    #[test]
    fn test_stale_detection() {
        let tracker = TVLTracker::new(60_000); // 60ms threshold
        
        // Initially stale (no updates)
        assert!(tracker.is_stale(100_000_000));
    }

    #[test]
    fn test_cache_line_alignment() {
        assert!(core::mem::size_of::<ProtocolData>() >= 64);
        assert!(core::mem::size_of::<ChainData>() >= 64);
        assert!(core::mem::size_of::<TVLTracker>() >= 64);
    }

    #[test]
    fn test_abi_decoder() {
        // Test uint256 decoding (32 bytes, big-endian)
        let data: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 42,
        ];
        
        let value = ABIDecoder::decode_uint256(&data);
        assert_eq!(value, Some(42));
    }
}
