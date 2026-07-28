//! Ultra-Low-Latency HFT Trading Bot - Stage 4
//! 
//! This stage implements:
//! - Chapter 1: Global Pre-Trade Risk Bus & Dynamic Position Sizing
//! - Chapter 2: Smart Order Routing & Algorithmic Execution  
//! - Chapter 3: Order Management System (OMS) & State Machines
//! - Chapter 4: Real-Time Reconciliation & Transaction Cost Analysis

#![no_std]
#![feature(avx2)]
#![feature(asm_experimental_arch)]
#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]

pub mod risk {
    pub mod pre_trade_bus;
    pub mod position_sizer;
    pub mod var_calculator;
}

pub mod execution {
    pub mod smart_router;
    pub mod twap_vwap;
    pub mod iceberg_handler;
}

pub mod oms {
    pub mod order_state_machine;
    pub mod idempotency_key;
    pub mod self_trade_prevention;
}

pub mod recon {
    pub mod real_time_recon;
    pub mod tca_engine;
    pub mod settlement_tracker;
}
