//! Macro Engine Module
//! Macro asset correlation and cross-asset regime detection.

pub mod cross_asset_corr;
pub mod macro_regime;
pub mod fear_greed_index;

pub use cross_asset_corr::*;
pub use macro_regime::*;
pub use fear_greed_index::*;
