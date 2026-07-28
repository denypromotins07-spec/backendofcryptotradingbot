//! Ultra-Low-Latency Crypto Trading Bot - Stage 5
//!
//! This library implements the Self-Learning "SOUL.md" Core, Observability,
//! Security Governance, and Compliance modules for an HFT crypto trading bot.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod soul;
pub mod observability;
pub mod security;
pub mod compliance;

/// Re-export all public types for convenience
pub mod prelude {
    pub use crate::soul::{SoulMemory, OnlineBandit, MistakeAnalyzer};
    pub use crate::observability::{MetricsBus, TraceLogger, AnomalyDetector};
    pub use crate::security::{SecretVault, HsmKmsLayer, NetworkAcl};
    pub use crate::compliance::{AuditLedger, RateLimiter, JurisdictionFilter};
}
