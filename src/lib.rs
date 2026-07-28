//! Ultra-low-latency crypto trading bot - Stage 2
//! 
//! This crate implements the market data pipeline with:
//! - Multi-exchange gateway abstraction
//! - Cross-venue normalization
//! - Kernel-bypass networking
//! - Feed recovery mechanisms

#![no_std]
#![feature(const_mut_refs)]
#![feature(maybe_uninit_zeroed)]
#![allow(dead_code)]
#![allow(unused_variables)]

extern crate alloc;

pub mod transport {
    pub mod ring_buffer;
}

pub mod gateways {
    pub mod gateway_manager;
    pub mod binance_ws_adapter;
    pub mod fix_protocol_engine;
}

pub mod normalization {
    pub mod symbol_mapper;
    pub mod l2_normalizer;
    pub mod microprice_engine;
}

pub mod network {
    pub mod xdp_wrapper;
    pub mod ptp_clock_sync;
    pub mod latency_probe;
}

pub mod recovery {
    pub mod sequence_tracker;
    pub mod book_resync;
    pub mod quality_monitor;
}

// Re-export main types
pub use gateways::gateway_manager::{Gateway, GatewayManager, MarketEvent, GatewayState};
pub use normalization::symbol_mapper::SymbolMapper;
pub use normalization::l2_normalizer::{L2Normalizer, NormalizedBook};
pub use normalization::microprice_engine::MicropriceEngine;
pub use network::latency_probe::LatencyProbeManager;
pub use recovery::sequence_tracker::SequenceTracker;
pub use recovery::book_resync::BookResyncManager;
pub use recovery::quality_monitor::QualityMonitor;

// Stage 3: Strategy Engine Modules
pub mod strategy {
    pub mod signals {
        pub mod signal_engine;
        pub mod lead_lag_model;
        pub mod relative_value;
    }
    
    pub mod arbitrage {
        pub mod funding_arb;
        pub mod triangular_arb;
        pub mod cross_venue_arb;
    }
    
    pub mod microstructure {
        pub mod market_maker;
        pub mod order_flow_alpha;
        pub mod liquidation_cascade;
    }
    
    pub mod regime {
        pub mod regime_classifier;
        pub mod options_gamma;
        pub mod ensemble_router;
    }
}
