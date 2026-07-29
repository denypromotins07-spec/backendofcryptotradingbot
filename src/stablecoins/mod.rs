//! Stablecoins Module
//! 
//! Chapter 3: Stablecoin Supply, Depeg Safeguards, and Bridge Finality

pub mod supply_monitor;
pub mod depeg_safeguard;
pub mod bridge_finality;

pub use supply_monitor::StablecoinSupplyMonitor;
pub use depeg_safeguard::StablecoinDepegMonitor;
pub use bridge_finality::BridgeFinalityMonitor;
