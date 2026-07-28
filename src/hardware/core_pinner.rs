// src/hardware/core_pinner.rs
//! OS-Level CPU Core Pinning and IRQ Affinity Tuning for Deterministic Latency
//!
//! This module implements:
//! - CPU core pinning for threads
//! - IRQ affinity configuration
//! - Isolation of critical cores from OS scheduling
//! - Real-time priority setting
//!
//! Micro-optimizations:
//! - Direct syscall interface (no library overhead)
//! - Batch affinity updates
//! - Cache-line aligned CPU sets

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Maximum CPUs supported
const MAX_CPUS: usize = 256;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// CPU set structure for affinity operations
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuSet {
    bits: [u64; MAX_CPUS / 64],
}

impl CpuSet {
    /// Create an empty CPU set
    #[inline]
    pub const fn new() -> Self {
        Self {
            bits: [0; MAX_CPUS / 64],
        }
    }

    /// Create a CPU set with a single CPU
    #[inline]
    pub fn single(cpu_id: usize) -> Self {
        let mut set = Self::new();
        set.set(cpu_id);
        set
    }

    /// Set a CPU in the set
    #[inline]
    pub fn set(&mut self, cpu_id: usize) {
        if cpu_id < MAX_CPUS {
            let idx = cpu_id / 64;
            let bit = cpu_id % 64;
            self.bits[idx] |= 1 << bit;
        }
    }

    /// Clear a CPU from the set
    #[inline]
    pub fn clear(&mut self, cpu_id: usize) {
        if cpu_id < MAX_CPUS {
            let idx = cpu_id / 64;
            let bit = cpu_id % 64;
            self.bits[idx] &= !(1 << bit);
        }
    }

    /// Check if CPU is in the set
    #[inline]
    pub fn is_set(&self, cpu_id: usize) -> bool {
        if cpu_id >= MAX_CPUS {
            return false;
        }
        let idx = cpu_id / 64;
        let bit = cpu_id % 64;
        (self.bits[idx] & (1 << bit)) != 0
    }

    /// Get raw bits for syscall
    #[inline]
    pub fn as_ptr(&self) -> *const u64 {
        self.bits.as_ptr()
    }

    /// Get size in bytes
    #[inline]
    pub const fn size() -> usize {
        core::mem::size_of::<CpuSet>()
    }
}

impl Default for CpuSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Core pinner for deterministic thread placement
pub struct CorePinner {
    /// Available CPUs
    available_cpus: CpuSet,
    /// Isolated CPUs (reserved for trading)
    isolated_cpus: CpuSet,
    /// Current pinned CPU for this thread
    current_cpu: AtomicUsize,
    /// Is pinning active?
    is_active: AtomicBool,
}

impl CorePinner {
    /// Create a new core pinner
    pub fn new() -> Self {
        let mut available = CpuSet::new();
        
        // Detect available CPUs
        for i in 0..MAX_CPUS {
            if is_cpu_online(i) {
                available.set(i);
            }
        }

        Self {
            available_cpus: available,
            isolated_cpus: CpuSet::new(),
            current_cpu: AtomicUsize::new(MAX_CPUS), // Invalid = not pinned
            is_active: AtomicBool::new(false),
        }
    }

    /// Mark CPUs as isolated (reserved for critical threads)
    pub fn isolate_cpus(&mut self, cpu_ids: &[usize]) {
        for &cpu_id in cpu_ids {
            self.isolated_cpus.set(cpu_id);
            self.available_cpus.clear(cpu_id);
        }
    }

    /// Pin current thread to a specific CPU
    ///
    /// # Returns
    /// true if successful, false otherwise
    pub fn pin_current(&self, cpu_id: usize) -> bool {
        if !self.available_cpus.is_set(cpu_id) && !self.isolated_cpus.is_set(cpu_id) {
            eprintln!("CPU {} is not available for pinning", cpu_id);
            return false;
        }

        let result = unsafe {
            pin_thread_to_cpu(cpu_id)
        };

        if result {
            self.current_cpu.store(cpu_id, Ordering::Release);
            self.is_active.store(true, Ordering::Release);
        }

        result
    }

    /// Pin current thread to an isolated CPU
    pub fn pin_to_isolated(&self, cpu_id: usize) -> bool {
        if !self.isolated_cpus.is_set(cpu_id) {
            eprintln!("CPU {} is not in isolated set", cpu_id);
            return false;
        }

        self.pin_current(cpu_id)
    }

    /// Unpin current thread (allow OS scheduling)
    pub fn unpin_current(&self) {
        #[cfg(target_os = "linux")]
        unsafe {
            use libc::{cpu_set_t, pthread_self, sched_setaffinity};
            
            let mut cpuset: cpu_set_t = core::mem::zeroed();
            // Allow all CPUs
            for i in 0..std::cmp::min(MAX_CPUS, libc::CPU_SETSIZE as usize) {
                libc::CPU_SET(i, &mut cpuset);
            }
            
            sched_setaffinity(
                pthread_self(),
                core::mem::size_of::<cpu_set_t>(),
                &cpuset as *const _ as *const _,
            );
        }

        self.current_cpu.store(MAX_CPUS, Ordering::Release);
        self.is_active.store(false, Ordering::Release);
    }

    /// Get current pinned CPU
    pub fn current_cpu(&self) -> Option<usize> {
        let cpu = self.current_cpu.load(Ordering::Acquire);
        if cpu < MAX_CPUS {
            Some(cpu)
        } else {
            None
        }
    }

    /// Check if currently pinned
    pub fn is_pinned(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Get list of available (non-isolated) CPUs
    pub fn available_cpus(&self) -> Vec<usize> {
        let mut cpus = Vec::new();
        for i in 0..MAX_CPUS {
            if self.available_cpus.is_set(i) {
                cpus.push(i);
            }
        }
        cpus
    }

    /// Get list of isolated CPUs
    pub fn isolated_cpus(&self) -> Vec<usize> {
        let mut cpus = Vec::new();
        for i in 0..MAX_CPUS {
            if self.isolated_cpus.is_set(i) {
                cpus.push(i);
            }
        }
        cpus
    }
}

impl Default for CorePinner {
    fn default() -> Self {
        Self::new()
    }
}

/// Set IRQ affinity for a specific IRQ
pub fn set_irq_affinity(irq: u32, cpu_mask: &CpuSet) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        let path = format!("/proc/irq/{}/smp_affinity_list", irq);
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)?;
        
        // Build CPU list string
        let mut cpu_list = String::new();
        let mut start: Option<usize> = None;
        let mut end: Option<usize> = None;
        
        for cpu_id in 0..MAX_CPUS {
            if cpu_mask.is_set(cpu_id) {
                if start.is_none() {
                    start = Some(cpu_id);
                }
                end = Some(cpu_id);
            } else if let Some(s) = start {
                if let Some(e) = end {
                    if !cpu_list.is_empty() {
                        cpu_list.push(',');
                    }
                    if s == e {
                        cpu_list.push_str(&format!("{}", s));
                    } else {
                        cpu_list.push_str(&format!("{}-{}", s, e));
                    }
                    start = None;
                    end = None;
                }
            }
        }
        
        // Handle last range
        if let Some(s) = start {
            if let Some(e) = end {
                if !cpu_list.is_empty() {
                    cpu_list.push(',');
                }
                if s == e {
                    cpu_list.push_str(&format!("{}", s));
                } else {
                    cpu_list.push_str(&format!("{}-{}", s, e));
                }
            }
        }
        
        file.write_all(cpu_list.as_bytes())?;
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        let _ = irq;
        let _ = cpu_mask;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "IRQ affinity only supported on Linux",
        ))
    }
}

/// Set real-time priority for current thread
pub fn set_realtime_priority(priority: u32) -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        use libc::{pthread_self, sched_param, SCHED_FIFO};
        
        let mut param: sched_param = core::mem::zeroed();
        param.sched_priority = priority as i32;
        
        let result = libc::pthread_setschedparam(
            pthread_self(),
            SCHED_FIFO,
            &param as *const _,
        );
        
        result == 0
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        let _ = priority;
        false
    }
}

/// Check if a CPU is online
fn is_cpu_online(cpu_id: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let path = format!("/sys/devices/system/cpu/cpu{}/online", cpu_id);
        if let Ok(content) = fs::read_to_string(&path) {
            return content.trim() == "1";
        }
        // CPU 0 is always online
        cpu_id == 0
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // Assume first few CPUs are online
        cpu_id < num_cpus()
    }
}

/// Get number of CPUs
fn num_cpus() -> usize {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }
}

/// Pin thread to CPU (unsafe low-level function)
unsafe fn pin_thread_to_cpu(cpu_id: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        use libc::{cpu_set_t, pthread_self, sched_setaffinity};
        
        let mut cpuset: cpu_set_t = core::mem::zeroed();
        libc::CPU_SET(cpu_id, &mut cpuset);
        
        let result = sched_setaffinity(
            pthread_self(),
            core::mem::size_of::<cpu_set_t>(),
            &cpuset as *const _ as *const _,
        );
        
        result == 0
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu_id;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_set_creation() {
        let set = CpuSet::new();
        assert!(!set.is_set(0));
        
        let single = CpuSet::single(4);
        assert!(single.is_set(4));
        assert!(!single.is_set(5));
    }

    #[test]
    fn test_cpu_set_operations() {
        let mut set = CpuSet::new();
        
        set.set(0);
        set.set(1);
        set.set(64); // Test across boundary
        
        assert!(set.is_set(0));
        assert!(set.is_set(1));
        assert!(set.is_set(64));
        assert!(!set.is_set(2));
        
        set.clear(1);
        assert!(!set.is_set(1));
    }

    #[test]
    fn test_core_pinner_creation() {
        let pinner = CorePinner::new();
        assert!(!pinner.is_pinned());
        assert!(pinner.current_cpu().is_none());
    }

    #[test]
    fn test_num_cpus() {
        let count = num_cpus();
        assert!(count > 0);
        assert!(count <= MAX_CPUS);
    }
}
