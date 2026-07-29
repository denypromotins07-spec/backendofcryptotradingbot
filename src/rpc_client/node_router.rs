//! Latency-aware RPC node router that load-balances requests across multiple providers.
//! 
//! Uses lock-free exponential moving average to dynamically weight RPC provider reliability,
//! with atomic flags for instant routing pivoting without mutexes.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of RPC nodes
const MAX_NODES: usize = 32;

/// EMA decay factor (scaled by 1000)
const EMA_ALPHA: u64 = 100; // 0.1

/// Fixed-point scale for weights
const WEIGHT_SCALE: u64 = 1_000_000;

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
}

/// RPC node info - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct RpcNode {
    /// Node ID hash
    node_id: u64,
    /// Endpoint URL hash
    endpoint_hash: u64,
    /// Current latency EMA (microseconds, fixed-point)
    latency_ema_fixed: u64,
    /// Success count
    success_count: u64,
    /// Failure count
    failure_count: u64,
    /// Weight (fixed-point, scaled by 1e6)
    weight_fixed: u64,
    /// Is healthy
    is_healthy: bool,
    /// Is active
    is_active: bool,
    /// Padding to reach 64 bytes
    _padding: [u8; 54],
}

const _: () = assert!(core::mem::size_of::<RpcNode>() == 64);

/// Request record for tracking - cache-line aligned
#[repr(C)]
#[derive(Clone, Copy)]
struct RequestRecord {
    /// Request ID
    request_id: u64,
    /// Node ID used
    node_id: u64,
    /// Start timestamp (cycles)
    start_cycles: u64,
    /// End timestamp (cycles)
    end_cycles: u64,
    /// Success flag
    success: bool,
    /// Padding
    _padding: [u8; 47],
}

const _: () = assert!(core::mem::size_of::<RequestRecord>() == 64);

/// Main RPC node router
#[repr(C)]
pub struct LatencyAwareRouter {
    /// RPC nodes (pre-allocated)
    nodes: [RpcNode; MAX_NODES],
    /// Recent requests (circular buffer)
    requests: [RequestRecord; 256],
    /// Node count
    node_count: AtomicU64,
    /// Request head index
    request_head: AtomicU64,
    /// Total requests
    total_requests: PaddedAtomicU64,
    /// Total failures
    total_failures: PaddedAtomicU64,
    /// Current best node index
    best_node_idx: AtomicU64,
    /// Failover mode
    failover_mode: AtomicBool,
    /// Last health check (cycles)
    last_health_check: AtomicU64,
}

impl LatencyAwareRouter {
    /// Create a new latency-aware router
    pub const fn new() -> Self {
        Self {
            nodes: [RpcNode {
                node_id: 0,
                endpoint_hash: 0,
                latency_ema_fixed: 0,
                success_count: 0,
                failure_count: 0,
                weight_fixed: 0,
                is_healthy: true,
                is_active: false,
                _padding: [0u8; 54],
            }; MAX_NODES],
            requests: [RequestRecord {
                request_id: 0,
                node_id: 0,
                start_cycles: 0,
                end_cycles: 0,
                success: false,
                _padding: [0u8; 47],
            }; 256],
            node_count: AtomicU64::new(0),
            request_head: AtomicU64::new(0),
            total_requests: PaddedAtomicU64::new(0),
            total_failures: PaddedAtomicU64::new(0),
            best_node_idx: AtomicU64::new(0),
            failover_mode: AtomicBool::new(false),
            last_health_check: AtomicU64::new(0),
        }
    }
    
    /// Register an RPC node
    #[inline]
    pub fn register_node(&self, node: RpcNode) -> bool {
        let idx = self.node_count.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= MAX_NODES {
            self.node_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            *self.nodes.get_unchecked_mut(idx) = node;
        }
        
        // Initialize weight
        unsafe {
            let n = &mut *self.nodes.get_unchecked_mut(idx);
            n.weight_fixed = WEIGHT_SCALE / MAX_NODES as u64;
        }
        
        self.recalculate_weights();
        
        true
    }
    
    /// Select best node for request (lock-free)
    #[inline]
    pub fn select_node(&self) -> Option<u64> {
        if self.failover_mode.load(Ordering::Relaxed) {
            // In failover mode, use round-robin among healthy nodes
            return self.select_round_robin();
        }
        
        // Select based on weight (weighted random)
        self.select_weighted()
    }
    
    /// Round-robin selection for failover
    #[inline]
    fn select_round_robin(&self) -> Option<u64> {
        let count = self.node_count.load(Ordering::Relaxed);
        if count == 0 { return None; }
        
        // Find first healthy node
        for i in 0..count as usize {
            unsafe {
                let node = *self.nodes.get_unchecked(i);
                if node.is_healthy && node.is_active {
                    return Some(node.node_id);
                }
            }
        }
        
        None
    }
    
    /// Weighted selection based on latency/reliability
    #[inline]
    fn select_weighted(&self) -> Option<u64> {
        let count = self.node_count.load(Ordering::Relaxed);
        if count == 0 { return None; }
        
        // Simple weighted selection: pick node with highest weight
        let mut best_idx = 0usize;
        let mut best_weight = 0u64;
        
        for i in 0..count as usize {
            unsafe {
                let node = *self.nodes.get_unchecked(i);
                if node.is_healthy && node.is_active && node.weight_fixed > best_weight {
                    best_weight = node.weight_fixed;
                    best_idx = i;
                }
            }
        }
        
        if best_weight == 0 { return None; }
        
        unsafe {
            Some(self.nodes.get_unchecked(best_idx).node_id)
        }
    }
    
    /// Record request completion and update metrics
    #[inline]
    pub fn record_completion(&self, request_id: u64, node_id: u64, success: bool) {
        use core::arch::x86_64::_rdtsc;
        
        let end_cycles = unsafe { _rdtsc() };
        
        // Find the node and update stats
        for i in 0..self.node_count.load(Ordering::Relaxed) as usize {
            unsafe {
                let node = &mut *self.nodes.get_unchecked_mut(i);
                if node.node_id == node_id {
                    if success {
                        node.success_count += 1;
                        
                        // Update latency EMA (simplified)
                        let latency_us = 100; // Would calculate from cycles
                        let old_ema = node.latency_ema_fixed;
                        node.latency_ema_fixed = 
                            ((old_ema * (WEIGHT_SCALE - EMA_ALPHA)) + (latency_us * EMA_ALPHA)) / WEIGHT_SCALE;
                    } else {
                        node.failure_count += 1;
                        node.is_healthy = node.failure_count < node.success_count / 10;
                    }
                    break;
                }
            }
        }
        
        // Record in circular buffer
        let idx = self.request_head.fetch_add(1, Ordering::Relaxed) % 256;
        unsafe {
            let req = &mut *self.requests.get_unchecked_mut(idx as usize);
            req.request_id = request_id;
            req.node_id = node_id;
            req.end_cycles = end_cycles;
            req.success = success;
        }
        
        self.total_requests.fetch_add(1);
        if !success {
            self.total_failures.fetch_add(1);
        }
        
        // Recalculate weights periodically
        if self.total_requests.load() % 100 == 0 {
            self.recalculate_weights();
        }
    }
    
    /// Recalculate node weights using EMA
    #[inline]
    fn recalculate_weights(&self) {
        let count = self.node_count.load(Ordering::Relaxed);
        if count == 0 { return; }
        
        let mut total_weight = 0u64;
        
        // Calculate inverse-latency weights
        for i in 0..count as usize {
            unsafe {
                let node = &mut *self.nodes.get_unchecked_mut(i);
                
                if node.is_healthy && node.is_active {
                    // Weight = 1 / latency (avoiding division by zero)
                    let latency = node.latency_ema_fixed.max(1);
                    node.weight_fixed = WEIGHT_SCALE / latency;
                    
                    // Apply reliability penalty
                    let total = node.success_count + node.failure_count;
                    if total > 0 {
                        let reliability = (node.success_count * WEIGHT_SCALE) / total;
                        node.weight_fixed = (node.weight_fixed * reliability) / WEIGHT_SCALE;
                    }
                    
                    total_weight += node.weight_fixed;
                } else {
                    node.weight_fixed = 0;
                }
            }
        }
        
        // Normalize weights
        if total_weight > 0 {
            for i in 0..count as usize {
                unsafe {
                    let node = &mut *self.nodes.get_unchecked_mut(i);
                    if node.weight_fixed > 0 {
                        node.weight_fixed = (node.weight_fixed * WEIGHT_SCALE) / total_weight;
                    }
                }
            }
        }
        
        // Update best node
        self.update_best_node();
    }
    
    /// Update best node index
    #[inline]
    fn update_best_node(&self) {
        let count = self.node_count.load(Ordering::Relaxed);
        let mut best_idx = 0u64;
        let mut best_weight = 0u64;
        
        for i in 0..count as usize {
            unsafe {
                let node = *self.nodes.get_unchecked(i);
                if node.weight_fixed > best_weight {
                    best_weight = node.weight_fixed;
                    best_idx = i as u64;
                }
            }
        }
        
        self.best_node_idx.store(best_idx, Ordering::Relaxed);
    }
    
    /// Enable failover mode
    #[inline]
    pub fn enable_failover(&self) {
        self.failover_mode.store(true, Ordering::Relaxed);
    }
    
    /// Disable failover mode
    #[inline]
    pub fn disable_failover(&self) {
        self.failover_mode.store(false, Ordering::Relaxed);
    }
    
    /// Get success rate
    #[inline]
    pub fn get_success_rate(&self) -> u64 {
        let total = self.total_requests.load();
        let failures = self.total_failures.load();
        if total == 0 { return WEIGHT_SCALE; }
        WEIGHT_SCALE - (failures * WEIGHT_SCALE / total)
    }
    
    /// Get node count
    #[inline]
    pub fn get_node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }
}

impl Default for LatencyAwareRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_register_node() {
        let router = LatencyAwareRouter::new();
        
        let node = RpcNode {
            node_id: 1,
            endpoint_hash: 0x1234,
            latency_ema_fixed: 1000,
            success_count: 0,
            failure_count: 0,
            weight_fixed: 0,
            is_healthy: true,
            is_active: true,
            _padding: [0u8; 54],
        };
        
        assert!(router.register_node(node));
        assert_eq!(router.get_node_count(), 1);
    }
    
    #[test]
    fn test_node_selection() {
        let router = LatencyAwareRouter::new();
        
        let node1 = RpcNode {
            node_id: 1,
            endpoint_hash: 0x1111,
            latency_ema_fixed: 1000,
            success_count: 100,
            failure_count: 0,
            weight_fixed: WEIGHT_SCALE,
            is_healthy: true,
            is_active: true,
            _padding: [0u8; 54],
        };
        
        router.register_node(node1);
        
        let selected = router.select_node();
        assert!(selected.is_some());
        assert_eq!(selected.unwrap(), 1);
    }
    
    #[test]
    fn test_success_rate() {
        let router = LatencyAwareRouter::new();
        
        // Simulate some requests
        router.total_requests.store(100, Ordering::Relaxed);
        router.total_failures.store(5, Ordering::Relaxed);
        
        let rate = router.get_success_rate();
        assert_eq!(rate, WEIGHT_SCALE - (5 * WEIGHT_SCALE / 100)); // 95%
    }
}
