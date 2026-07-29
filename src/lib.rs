//! Ultra-Low-Latency Crypto Trading Bot - Stage 6
//!
//! This library implements On-Chain Analytics, Network Health Monitoring,
//! Stablecoin Safeguards, and Zero-Copy RPC Parsing for HFT crypto trading.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]
#![cfg_attr(target_arch = "x86_64", feature(stdsimd))]

pub mod onchain;
pub mod network_health;
pub mod stablecoins;
pub mod rpc_client;

/// Re-export all public types for convenience
pub mod prelude {
    pub use crate::onchain::{WhaleTracker, ExchangeFlowsAggregator, TokenUnlocksTracker};
    pub use crate::network_health::{Eip1559Predictor, SolanaLeaderMonitor, BtcMempoolTracker};
    pub use crate::stablecoins::{StablecoinSupplyMonitor, StablecoinDepegMonitor, BridgeFinalityMonitor};
    pub use crate::rpc_client::{ZeroCopyRpcParser, WsSubscriptionManager, LatencyAwareRouter};
}
