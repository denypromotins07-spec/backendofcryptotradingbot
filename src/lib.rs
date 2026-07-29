//! Ultra-Low-Latency Crypto Trading Bot - Stage 13
//!
//! This library implements Advanced Order Types, Exchange Matching Emulation,
//! Latency Arbitrage, and High-Frequency Data Compression for HFT crypto trading.
//!
//! ## Architecture
//!
//! ### Chapter 1: Advanced Order Types & Execution Nuances (Stage 13)
//! - Post-only, IOC, FOK, GTT order state machines with strict validation
//! - Reduce-only logic and position netting validators
//! - Iceberg order slicer with randomized child sizes
//!
//! ### Chapter 2: Exchange Specific Quirks & Matching Engine Emulation (Stage 13)
//! - Binance matching engine emulation (price-time priority, STP rules)
//! - Bybit quirks (unified margin, fee rebates, order limits)
//! - OKX quirks (portfolio margin, combo margins, API rate limits)
//!
//! ### Chapter 3: Latency Arbitrage & Cross-Venue Sniping (Stage 13)
//! - Cross-venue latency arbitrage sniper using consolidated micro-prices
//! - Stale quote detector and toxic flow identifier
//! - Network jitter exploiter for microsecond edge extraction
//!
//! ### Chapter 4: High-Frequency Data Compression & State Serialization (Stage 13)
//! - Zero-copy L2 delta compression with run-length and dictionary encoding
//! - Custom lock-free LZ4 variant for financial time-series
//! - Ultra-fast state snapshot and recovery for zero-downtime hot-restarts
//!
//! ## Memory Safety Guarantees
//!
//! - All structures are `#[repr(C)]` and padded to 64-byte cache lines
//! - Zero heap allocations in hot paths (pre-allocated buffers)
//! - Fixed-point arithmetic throughout to avoid FPU non-determinism
//! - Lock-free atomics for thread-safe state transitions
//! - SIMD intrinsics (AVX2) for vectorized comparisons

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]
#![cfg_attr(target_arch = "x86_64", feature(stdsimd))]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

pub mod orders;
pub mod exchange_quirks;
pub mod latency_arb;
pub mod compression;

// Re-export previous stages
pub mod smc;
pub mod indicators;
pub mod defi;
pub mod portfolio;
pub mod ml_inference;
pub mod options;
pub mod features;
pub mod macro_engine;

/// Re-export all public types for convenience
pub mod prelude {
    // SMC exports
    pub use crate::smc::structure_engine::{StructureEngine, SwingPoint, FixedPrice as SmcFixedPrice};
    pub use crate::smc::liquidity_pools::{LiquidityPools, LiquidityPool};
    pub use crate::smc::order_blocks::{OrderBlocksTracker, OrderBlock, FairValueGap, Candle};
    
    // Indicator exports
    pub use crate::indicators::streaming_ta::{StreamingEMA, StreamingSMA, StreamingVWAP, IndicatorBundle};
    pub use crate::indicators::momentum_osc::{MomentumRSI, MomentumMACD, MomentumADX};
    pub use crate::indicators::volatility_bands::{VolatilityBollinger, VolatilityATR, VolatilityKeltner, VolatilityBundle};
    
    // DeFi exports
    pub use crate::defi::tvl_tracker::{TVLTracker, ProtocolData, ChainData, ABIDecoder};
    pub use crate::defi::validator_metrics::{ValidatorMetricsTracker, ValidatorData, StakingYieldCalculator};
    pub use crate::defi::smart_contract_io::{SmartContractIO, ContractEvent, StateDelta};
    
    // Portfolio exports
    pub use crate::portfolio::risk_parity::{RiskParityCalculator, HierarchicalRiskParity, AssetRisk};
    pub use crate::portfolio::markowitz_solver::{MarkowitzOptimizer};
    pub use crate::portfolio::rebalancing_engine::{RebalancingEngine, RebalanceOrder, RebalanceTrigger};
}
