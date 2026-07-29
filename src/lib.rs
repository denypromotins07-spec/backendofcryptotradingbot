//! Ultra-Low-Latency Crypto Trading Bot - Stage 8
//!
//! This library implements Smart Money Concepts (SMC), Streaming Technical Analysis,
//! DeFi Analytics, and Real-Time Portfolio Optimization for HFT crypto trading.
//!
//! ## Architecture
//!
//! ### Chapter 1: Smart Money Concepts (SMC) & Liquidity Engineering
//! - Break of Structure (BOS) and Change of Character (CHoCH) detection
//! - Equal highs/lows identification and liquidity sweep detection
//! - Order block, Breaker block, and Fair Value Gap (FVG) mapping
//!
//! ### Chapter 2: Streaming Technical Analysis & Indicator Engine
//! - Lock-free streaming EMA, SMA, and VWAP calculators
//! - RSI, MACD, and ADX with SIMD-optimized rolling windows
//! - Bollinger Bands, ATR, and Keltner Channels with branchless math
//!
//! ### Chapter 3: DeFi Analytics, TVL, and Validator Metrics
//! - Real-time Total Value Locked (TVL) and protocol revenue aggregator
//! - Staking yield, validator uptime, and slashing risk monitor
//! - High-throughput smart contract event log parser
//!
//! ### Chapter 4: Real-Time Portfolio Optimization & Risk Parity
//! - Risk Parity and Hierarchical Risk Parity (HRP) weight calculator
//! - SIMD-accelerated Mean-Variance optimization with Ledoit-Wolf shrinkage
//! - Threshold-based and time-sliced atomic rebalancing execution router
//!
//! ## Memory Safety Guarantees
//!
//! - All structures are `#[repr(C)]` and padded to 64-byte cache lines
//! - Zero heap allocations in hot paths (pre-allocated buffers)
//! - Fixed-point arithmetic throughout to avoid FPU non-determinism
//! - Lock-free atomics for thread-safe state transitions

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]
#![cfg_attr(target_arch = "x86_64", feature(stdsimd))]

pub mod smc;
pub mod indicators;
pub mod defi;
pub mod portfolio;

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
