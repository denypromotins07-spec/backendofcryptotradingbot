//! Observability Module
//!
//! Chapter 2: Zero-Allocation Metrics, Tracing, and Latency Telemetry

pub mod metrics_bus;
pub mod trace_logger;
pub mod bot_anomaly;

pub use metrics_bus::MetricsBus;
pub use trace_logger::TraceLogger;
pub use bot_anomaly::AnomalyDetector;
