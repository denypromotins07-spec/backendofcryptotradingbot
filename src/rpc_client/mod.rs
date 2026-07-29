//! RPC Client Module
//! 
//! Chapter 4: Zero-Copy JSON-RPC Parsing and Latency-Aware Node Routing

pub mod zero_copy_rpc;
pub mod ws_subscription;
pub mod node_router;

pub use zero_copy_rpc::ZeroCopyRpcParser;
pub use ws_subscription::WsSubscriptionManager;
pub use node_router::LatencyAwareRouter;
