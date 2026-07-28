# Stage 2 Complete: Normalized Data Pipeline Architecture

## Overview

This stage implements the ultra-low-latency market data pipeline for the HFT crypto trading bot, focusing on exchange gateways, market data normalization, kernel-bypass networking, and feed recovery mechanisms.

## Memory Constraints

All components strictly enforce the **6.5GB RAM limit** through:
- Zero-copy parsing with pre-allocated buffers
- Custom stack-based allocators (no heap in hot path)
- Strict memory bounds via fixed-size arrays
- Cache-line aligned structures (`#[repr(C, align(64))]`)

## Directory Structure

```
src/
├── gateways/              # Chapter 1: Multi-Exchange Gateway Abstraction
│   ├── gateway_manager.rs     # Unified trait-based routing layer
│   ├── binance_ws_adapter.rs  # Zero-allocation WebSocket parser
│   └── fix_protocol_engine.rs # FIX 4.4 institutional gateway
│
├── normalization/         # Chapter 2: Cross-Venue Normalization
│   ├── symbol_mapper.rs       # Lock-free O(1) symbol mapping
│   ├── l2_normalizer.rs       # L2/L3 incremental delta processor
│   └── microprice_engine.rs   # Liquidity-weighted fair value
│
├── network/               # Chapter 3: Low-Latency Infrastructure
│   ├── xdp_wrapper.rs         # AF_XDP/DPDK kernel-bypass abstraction
│   ├── ptp_clock_sync.rs      # IEEE 1588 hardware timestamping
│   └── latency_probe.rs       # Nanosecond tick-to-trade measurement
│
├── recovery/              # Chapter 4: Feed Recovery & Quality
│   ├── sequence_tracker.rs    # Lock-free gap detection
│   ├── book_resync.rs         # Non-blocking snapshot reconciliation
│   └── quality_monitor.rs     # Anomaly detection & circuit breaker
│
└── transport/
    └── ring_buffer.rs     # Lock-free SPSC ring buffer
```

## Key Components

### 1. Gateway Manager (`gateway_manager.rs`)
- Unified `Gateway` trait for multi-venue routing
- `MarketEvent` struct fits exactly in 64-byte cache line
- State machine handling exchange halts, auctions, post-only modes
- Venue reliability scoring for dynamic routing weights

### 2. Binance WebSocket Adapter (`binance_ws_adapter.rs`)
- Hand-rolled SIMD-accelerated JSON parser using AVX2 intrinsics
- Zero heap allocations in hot path
- Pre-allocated 1MB parse buffer
- Sub-millisecond reconnection logic

### 3. FIX Protocol Engine (`fix_protocol_engine.rs`)
- FIX 4.4 compliant message parsing
- Direct SOH delimiter scanning with SIMD
- Fixed-point decimal conversion (no floating point)
- Bypasses redundant checks for trusted counterparties

### 4. Symbol Mapper (`symbol_mapper.rs`)
- Robin Hood hashing for consistent O(1) lookups
- Lock-free atomic operations
- Power-of-2 table size for bitwise modulo
- Maps exchange-specific symbols to internal u32 IDs

### 5. L2 Normalizer (`l2_normalizer.rs`)
- Cache-aligned bid/ask arrays (separate to prevent false sharing)
- Per-venue sequence tracking
- Atomic updates without locking
- Publishes to LMAX Disruptor ring buffer

### 6. Microprice Engine (`microprice_engine.rs`)
- Volume-weighted fair value calculation
- Consolidated cross-venue book construction
- Fixed-point arithmetic throughout
- Imbalance ratio computation

### 7. XDP Wrapper (`xdp_wrapper.rs`)
- AF_XDP/DPDK kernel-bypass socket abstraction
- Falls back to standard sockets if unavailable (WSL support)
- Zero-copy packet reception via mmap'd rings
- Batch processing for amortized syscall cost

### 8. PTP Clock Sync (`ptp_clock_sync.rs`)
- IEEE 1588 hardware timestamping
- User-space clock drift correction
- Linear regression for drift estimation
- TSC-based local timestamps

### 9. Latency Probe (`latency_probe.rs`)
- Nanosecond-precision tick-to-trade RTT measurement
- Lock-free probe injection
- Exponential moving average for latency tracking
- Full pipeline timing: socket → strategy → order

### 10. Sequence Tracker (`sequence_tracker.rs`)
- Per-venue atomic counters
- Automatic gap detection
- REST fallback triggers for resync
- Circular buffer for recent history

### 11. Book Resync (`book_resync.rs`)
- Double-buffered book state (active/shadow)
- Non-blocking snapshot application
- Atomic swap on successful reconciliation
- Zero-copy snapshot integration

### 12. Quality Monitor (`quality_monitor.rs`)
- Real-time anomaly detection (stale books, spikes)
- Circuit breaker with trip/half-open/closed states
- Statistical thresholds with rolling windows
- Halts trading on excessive anomalies

## Performance Guarantees

| Metric | Target | Implementation |
|--------|--------|----------------|
| Socket to ring buffer | < 2μs | Kernel-bypass + zero-copy |
| Symbol lookup | O(1) | Robin Hood hash table |
| Book update | Lock-free | Atomic operations |
| Gap detection | Immediate | Per-venue sequence tracking |
| Circuit breaker trip | < 100ns | Atomic flag check |

## Compile-Time Checks

- All market data structs verified to fit in 64-byte cache lines
- `#[repr(C, packed)]` for network packet structs
- Manual unaligned read handling to prevent CPU penalties
- `cargo fmt` and `cargo clippy` with strict HFT lints

## Testing

Each module includes unit tests verifying:
- Struct sizes and alignments
- Creation and initialization
- Core functionality (gap detection, parsing, etc.)

## Next Steps (Stage 3)

- Order execution engine
- Risk management subsystem
- Strategy framework integration
- Performance benchmarking suite

---

*Generated for HFT Crypto Trading Bot - Stage 2*
*Memory Budget: 6.5GB | Target Latency: < 2μs tick-to-trade*
