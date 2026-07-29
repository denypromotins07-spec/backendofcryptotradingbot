//! Chapter 2: Deep Order Flow Microstructure & Absorption Detection
//! 
//! Zero-allocation order flow analysis with lock-free data structures.
//! All structs are #[repr(C)] and padded to 64-byte cache lines.

pub mod footprint_chart;
pub mod volume_profile;
pub mod delta_absorption;
