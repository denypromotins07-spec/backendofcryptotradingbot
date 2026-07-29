//! Real-Time Portfolio Optimization & Risk Parity
//!
//! This module provides portfolio construction and rebalancing:
//! - Risk Parity and Hierarchical Risk Parity (HRP)
//! - Mean-Variance optimization with Ledoit-Wolf shrinkage
//! - Threshold-based and time-sliced rebalancing

pub mod risk_parity;
pub mod markowitz_solver;
pub mod rebalancing_engine;
