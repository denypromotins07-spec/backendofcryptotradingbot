//! CPU frequency scaling and power management governor lock.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[repr(C, align(64))]
pub struct CpuGovernor {
    pub active: AtomicBool,
    pub locked: AtomicBool,
    pub target_freq: AtomicU64,
    pub current_freq: AtomicU64,
    pub core_id: u32,
    _pad: [u8; 64 - 3 * 8 - 2 * 4 - 8],
}

impl CpuGovernor {
    #[inline]
    pub const fn new(core_id: u32) -> Self {
        Self {
            active: AtomicBool::new(false),
            locked: AtomicBool::new(false),
            target_freq: AtomicU64::new(0),
            current_freq: AtomicU64::new(0),
            core_id,
            _pad: [0u8; 64 - 3 * 8 - 2 * 4 - 8],
        }
    }
    
    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }
    #[inline]
    pub fn is_locked(&self) -> bool { self.locked.load(Ordering::Relaxed) }
    
    #[inline]
    pub fn lock_to_performance(&self, freq: u64) -> bool {
        if !self.is_active() { return false; }
        self.target_freq.store(freq, Ordering::Relaxed);
        self.current_freq.store(freq, Ordering::Relaxed);
        self.locked.store(true, Ordering::SeqCst);
        true
    }
    
    #[inline]
    pub fn unlock(&self) {
        self.locked.store(false, Ordering::SeqCst);
    }
}

const _: () = assert!(core::mem::size_of::<CpuGovernor>() % 64 == 0);
