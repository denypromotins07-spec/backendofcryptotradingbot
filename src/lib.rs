//! Ultra-Low-Latency Crypto Trading Bot - Stage 9
//!
//! This library implements Smart Money Concepts (SMC), Streaming Technical Analysis,
//! DeFi Analytics, Real-Time Portfolio Optimization, ML Inference, Options Pricing,
//! Feature Engineering, and Macro Asset Correlation for HFT crypto trading.
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
//! ### Chapter 5: Lightweight Machine Learning Inference (Stage 9)
//! - Oblivious Decision Trees (CatBoost-style) inference engine
//! - Minimalist ONNX tensor parser for neural net execution
//! - Streaming SGD for continuous online learning
//!
//! ### Chapter 6: Advanced Options Pricing (Stage 9)
//! - Lock-free implied volatility surface construction
//! - SIMD-accelerated Black-Scholes-Merton pricing and Greeks
//! - Real-time Gamma Exposure (GEX) tracker
//!
//! ### Chapter 7: High-Dimensional Feature Engineering (Stage 9)
//! - Memory-mapped feature store for online/offline serving
//! - Streaming PCA using Oja's rule for dimensionality reduction
//! - Online mutual information and feature importance tracker
//!
//! ### Chapter 8: Macro Asset Correlation & Regime Detection (Stage 9)
//! - Real-time cross-asset correlation tracker (DXY, Gold, Oil, Bonds)
//! - Hidden Markov Model for macroeconomic regime shifts
//! - Fear & Greed composite index from options skew and funding rates
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
