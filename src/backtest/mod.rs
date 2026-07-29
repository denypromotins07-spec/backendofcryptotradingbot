//! Chapter 4: High-Performance Event-Driven Backtesting & Walk-Forward Engine
//! 
//! Deterministic nanosecond-precision backtesting with compile-time assertions.
//! Shared memory IPC for walk-forward optimization.

pub mod event_replay;
pub mod walk_forward;
pub mod slippage_modeler;
