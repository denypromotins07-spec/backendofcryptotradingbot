//! Streaming Technical Analysis & Indicator Engine
//!
//! This module provides lock-free, O(1) streaming indicators:
//! - EMA, SMA, VWAP with incremental updates
//! - RSI, MACD, ADX momentum oscillators
//! - Bollinger Bands, ATR, Keltner Channels

pub mod streaming_ta;
pub mod momentum_osc;
pub mod volatility_bands;
