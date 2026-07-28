// src/lib.rs
//! Ultra-Low-Latency Crypto Trading Bot Library
//!
//! This is a high-performance trading infrastructure designed for:
//! - Microsecond execution latency
//! - 6.5GB strict memory limit enforcement
//! - Lock-free concurrent data structures
//! - CPU cache-line optimized memory layouts
//! - Kernel-bypass readiness
//!
//! # Architecture
//!
//! The library is organized into four main modules:
//!
//! ## Core (`src/core/`)
//! - Custom bump allocator with circuit breaker
//! - LMAX Disruptor-style event bus
//! - Single-threaded deterministic event loop
//!
//! ## Transport (`src/transport/`)
//! - Wait-free SPSC ring buffers
//! - Shared-memory IPC channels
//! - Memory-mapped file handlers
//!
//! ## Market Data (`src/market_data/`)
//! - SBE codec with AVX2 acceleration
//! - Flat-array lock-free order book
//! - Normalized tick feed with gap detection
//!
//! ## Hardware (`src/hardware/`)
//! - Custom thread pool with CPU affinity
//! - NUMA-aware memory allocator
//! - OS-level core pinning

#![no_std]
#![cfg_attr(test, allow(dead_code))]
#![warn(clippy::all)]
#![forbid(clippy::box_collection, clippy::vec_box)]

extern crate alloc;
extern crate std;

/// Compile-time assertion macro
#[macro_export]
macro_rules! const_assert {
    ($x:expr) => {
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

// Core modules
pub mod allocator;
pub mod event_loop;

// Transport modules  
pub mod ring_buffer;
pub mod ipc_channel;
pub mod shared_memory;

// Market data modules
pub mod sbe_codec;
pub mod order_book;
pub mod tick_feed;

// Hardware modules
pub mod thread_pool;
pub mod numa_allocator;
pub mod core_pinner;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get library build info
pub fn build_info() -> &'static str {
    concat!(
        "Ultra-Low-Latency Crypto Trading Bot v",
        env!("CARGO_PKG_VERSION"),
        "\n",
        "Built with Rust"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_build_info() {
        let info = build_info();
        assert!(info.contains("Ultra-Low-Latency"));
    }
}
