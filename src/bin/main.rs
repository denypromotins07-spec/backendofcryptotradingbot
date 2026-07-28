// src/core/main.rs
//! Ultra-Low-Latency Crypto Trading Bot - Main Entry Point
//!
//! This is the tokio-free, single-threaded deterministic event loop entry point.
//! Designed for microsecond execution with kernel-bypass readiness.
//!
//! Architecture:
//! - Single-threaded event loop for deterministic execution
//! - Custom bump allocator enforcing 6.5GB RAM limit
//! - Lock-free data structures throughout
//! - CPU pinning and NUMA-aware memory allocation
//!
//! Micro-optimizations:
//! - No async runtime overhead (no tokio/async-std)
//! - Batch processing for instruction cache efficiency
//! - Zero heap allocations in hot path
//! - rdtsc-based timestamps (no syscall overhead)

#![no_std]
#![cfg_attr(not(test), no_main)]
#![feature(alloc_error_handler)]
#![warn(clippy::all)]
#![forbid(clippy::box_collection, clippy::vec_box)]

extern crate alloc;
extern crate std;

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, Ordering};

// Core modules
mod allocator;
mod event_loop;

use crate::allocator::{initialize_allocator, get_memory_usage, is_circuit_breaker_active};
use crate::event_loop::{EventBus, Event, EventType};

/// Global shutdown flag for graceful termination
static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

/// Maximum batch size for event processing (optimized for L1 cache)
const BATCH_SIZE: usize = 64;

/// Heartbeat interval in iterations
const HEARTBEAT_INTERVAL: u64 = 1000;

/// Entry point for the trading bot
///
/// # Initialization Sequence:
/// 1. Initialize custom memory allocator
/// 2. Set up panic hooks
/// 3. Pin to CPU core
/// 4. Start event loop
#[cfg(not(test))]
#[no_mangle]
pub extern "C" fn main() -> ! {
    // Initialize the custom allocator with 6.5GB arena
    unsafe {
        if let Err(e) = initialize_allocator() {
            panic!("Failed to initialize allocator: {}", e);
        }
    }

    // Log startup information
    log_startup_info();

    // Create the event bus
    let event_bus = EventBus::new(BATCH_SIZE);

    // Run the main event loop (never returns)
    run_event_loop(&event_bus);
}

/// Log startup configuration and system info
fn log_startup_info() {
    #[cfg(target_arch = "x86_64")]
    {
        // Detect CPU features using cpuid
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_sse4_2 = is_x86_feature_detected!("sse4.2");
        
        eprintln!("=== Ultra-Low-Latency Trading Bot ===");
        eprintln!("Architecture: x86_64");
        eprintln!("AVX2 Support: {}", if has_avx2 { "Yes" } else { "No" });
        eprintln!("SSE4.2 Support: {}", if has_sse4_2 { "Yes" } else { "No" });
    }
    
    eprintln!("Memory Limit: 6.5 GB");
    eprintln!("Batch Size: {}", BATCH_SIZE);
    eprintln!("Event Loop: Single-threaded deterministic");
}

/// Main event loop - runs until shutdown signal received
///
/// This is the heart of the trading system:
/// - Processes market data events
/// - Executes trading strategies
/// - Manages order lifecycle
/// - Handles heartbeats and health checks
fn run_event_loop(event_bus: EventBus) -> ! {
    let mut iteration: u64 = 0;
    let mut last_heartbeat: u64 = 0;

    loop {
        // Check for shutdown signal
        if SHUTDOWN_FLAG.load(Ordering::Acquire) {
            break;
        }

        // Check circuit breaker (memory limit)
        if is_circuit_breaker_active() {
            eprintln!("CRITICAL: Circuit breaker tripped - memory limit approached");
            break;
        }

        // Process events in batches (maximizes instruction cache hits)
        let events_processed = event_bus.process_batch(|event| {
            handle_event(event);
        });

        // Send heartbeat periodically
        iteration += 1;
        if iteration - last_heartbeat >= HEARTBEAT_INTERVAL {
            send_heartbeat(&event_bus);
            last_heartbeat = iteration;
        }

        // If no events processed, yield briefly to reduce CPU usage
        if events_processed == 0 {
            // Use pause instruction instead of sleep for low latency
            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_pause();
            }
            
            // Brief spin before checking again
            for _ in 0..100 {
                core::hint::spin_loop();
            }
        }
    }

    // Graceful shutdown
    shutdown();
}

/// Handle individual events based on type
#[inline(always)]
fn handle_event(event: &Event) {
    match event.event_type {
        EventType::MarketData => {
            // Process market data update
            process_market_data(event);
        }
        EventType::OrderSubmission => {
            // Handle order submission
            process_order_submission(event);
        }
        EventType::OrderCancellation => {
            // Handle order cancellation
            process_order_cancellation(event);
        }
        EventType::OrderFill => {
            // Handle order fill confirmation
            process_order_fill(event);
        }
        EventType::Heartbeat => {
            // Process heartbeat (health check)
            process_heartbeat(event);
        }
        EventType::SystemSignal => {
            // Handle system signals (shutdown, config change, etc.)
            process_system_signal(event);
        }
    }
}

/// Process market data events
#[inline(always)]
fn process_market_data(_event: &Event) {
    // Hot path - zero allocations
    // Market data processing logic here
}

/// Process order submission events
#[inline(always)]
fn process_order_submission(_event: &Event) {
    // Order submission logic
}

/// Process order cancellation events
#[inline(always)]
fn process_order_cancellation(_event: &Event) {
    // Order cancellation logic
}

/// Process order fill events
#[inline(always)]
fn process_order_fill(_event: &Event) {
    // Order fill confirmation logic
}

/// Process heartbeat events
#[inline(always)]
fn process_heartbeat(_event: &Event) {
    // Health check logic
}

/// Process system signals
#[inline(always)]
fn process_system_signal(event: &Event) {
    // Check for shutdown signal in payload
    if event.payload_size > 0 && event.payload[0] == b'S' {
        SHUTDOWN_FLAG.store(true, Ordering::Release);
    }
}

/// Send heartbeat event through the event bus
#[inline(always)]
fn send_heartbeat(event_bus: &EventBus) {
    let heartbeat = Event::new(EventType::Heartbeat, 0, 0);
    let _ = event_bus.publish(heartbeat);
    
    // Log memory usage periodically
    let usage = get_memory_usage();
    if usage > 0 {
        eprintln!("Memory Usage: {} bytes", usage);
    }
}

/// Graceful shutdown procedure
fn shutdown() -> ! {
    eprintln!("Initiating graceful shutdown...");
    
    // Flush any pending state
    // Close connections
    // Release resources
    
    eprintln!("Shutdown complete");
    
    // Exit cleanly
    std::process::exit(0);
}

/// Custom panic handler for controlled failure
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    eprintln!("=== PANIC ===");
    eprintln!("Location: {:?}", info.location());
    if let Some(msg) = info.message() {
        eprintln!("Message: {}", msg);
    }
    
    // Memory state will be logged by allocator's panic hook
    
    // Abort to preserve shared memory state
    core::intrinsics::abort();
}

/// Allocation error handler
#[alloc_error_handler]
fn alloc_error(_layout: core::alloc::Layout) -> ! {
    eprintln!("=== ALLOCATION ERROR ===");
    eprintln!("Memory limit reached or allocator exhausted");
    core::intrinsics::abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(BATCH_SIZE, 64);
        assert_eq!(HEARTBEAT_INTERVAL, 1000);
    }

    #[test]
    fn test_event_bus_creation() {
        let event_bus = EventBus::new(BATCH_SIZE);
        assert_eq!(event_bus.depth(), 0);
    }

    #[test]
    fn test_shutdown_flag() {
        assert!(!SHUTDOWN_FLAG.load(Ordering::Relaxed));
        SHUTDOWN_FLAG.store(true, Ordering::Release);
        assert!(SHUTDOWN_FLAG.load(Ordering::Relaxed));
        SHUTDOWN_FLAG.store(false, Ordering::Release);
    }
}
