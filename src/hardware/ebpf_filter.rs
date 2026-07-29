//! eBPF wrapper for kernel-level network packet filtering.
//! Uses rdtsc for timestamp delta calculations.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[repr(C, align(64))]
pub struct EbpfFilter {
    pub active: AtomicBool,
    pub packet_count: AtomicU64,
    pub dropped_count: AtomicU64,
    pub last_ts: AtomicU64,
    _pad: [u8; 64 - 5 * 8],
}

impl EbpfFilter {
    #[inline]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            packet_count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
            last_ts: AtomicU64::new(0),
            _pad: [0u8; 64 - 5 * 8],
        }
    }
    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }
    
    #[inline]
    pub fn rdtsc(&self) -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    
    #[inline]
    pub fn filter_packet(&self, ts: u64) -> bool {
        if !self.is_active() { return true; }
        let _ = self.packet_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.last_ts.swap(ts, Ordering::Relaxed);
        let delta = ts.wrapping_sub(prev);
        // Drop if delta too small (noise)
        if delta < 1000 {
            let _ = self.dropped_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }
}

const _: () = assert!(core::mem::size_of::<EbpfFilter>() % 64 == 0);
