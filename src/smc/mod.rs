//! Smart Money Concepts (SMC) & Liquidity Engineering
//!
//! This module implements pattern recognition for institutional order flow:
//! - Break of Structure (BOS) and Change of Character (CHoCH)
//! - Equal highs/lows and liquidity sweeps
//! - Order blocks, breaker blocks, and Fair Value Gaps (FVG)

pub mod structure_engine;
pub mod liquidity_pools;
pub mod order_blocks;
