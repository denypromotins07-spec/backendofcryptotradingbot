//! HugeTLBfs memory allocator integration to minimize TLB misses.

#![allow(clippy::missing_docs_in_private_items)]

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const CACHE_LINE_SIZE: usize = 64;
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB

#[repr(C, align(64))]
pub struct HugeTlbAllocator {
    pub active: AtomicBool,
    pub allocated_pages: AtomicUsize,
    pub max_pages: usize,
    pub base_ptr: *mut u8,
    pub current_offset: AtomicUsize,
    _pad: [u8; CACHE_LINE_SIZE - 2 * 8 - 2 * 8 - 8],
}

unsafe impl Send for HugeTlbAllocator {}
unsafe impl Sync for HugeTlbAllocator {}

impl HugeTlbAllocator {
    #[inline]
    pub const fn new(max_pages: usize) -> Self {
        Self {
            active: AtomicBool::new(false),
            allocated_pages: AtomicUsize::new(0),
            max_pages,
            base_ptr: core::ptr::null_mut(),
            current_offset: AtomicUsize::new(0),
            _pad: [0u8; CACHE_LINE_SIZE - 2 * 8 - 2 * 8 - 8],
        }
    }
    
    #[inline]
    pub fn activate(&self) { self.active.store(true, Ordering::Relaxed); }
    #[inline]
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }
    
    #[inline]
    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        if !self.is_active() || self.base_ptr.is_null() { return None; }
        let aligned_size = (size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);
        let offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        if offset + aligned_size > self.max_pages * HUGE_PAGE_SIZE { return None; }
        unsafe { Some(self.base_ptr.add(offset)) }
    }
}

const _: () = assert!(core::mem::size_of::<HugeTlbAllocator>() % 64 == 0);
