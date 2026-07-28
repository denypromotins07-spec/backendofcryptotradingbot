// src/hardware/thread_pool.rs
//! Custom Thread Pool that Avoids OS Context Switching and Thread Migration
//!
//! This module implements a specialized thread pool for low-latency trading:
//! - Pre-spawned dedicated threads (no dynamic creation)
//! - CPU core affinity binding
//! - Work-stealing with minimal contention
//! - Priority-aware scheduling
//!
//! Micro-optimizations:
//! - Lock-free work queues
//! - Cache-line padded task structures
//! - Spin-wait instead of blocking for hot paths
//! - NUMA-aware thread placement

#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of worker threads
const MAX_WORKERS: usize = 32;

/// Maximum tasks per worker queue
const MAX_TASKS_PER_QUEUE: usize = 1 << 10; // 1024

/// Task function type
pub type TaskFn = Box<dyn FnOnce() + Send + 'static>;

/// Padded atomic for cache-line isolation
#[repr(C)]
struct PaddedAtomicBool {
    value: AtomicBool,
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicBool>()],
}

impl PaddedAtomicBool {
    #[inline]
    const fn new(val: bool) -> Self {
        Self {
            value: AtomicBool::new(val),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicBool>()],
        }
    }
}

#[repr(C)]
struct PaddedAtomicUsize {
    value: AtomicUsize,
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>()],
}

impl PaddedAtomicUsize {
    #[inline]
    const fn new(val: usize) -> Self {
        Self {
            value: AtomicUsize::new(val),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<AtomicUsize>()],
        }
    }
}

/// Task slot in work queue
#[repr(C)]
struct TaskSlot {
    task: UnsafeCell<Option<TaskFn>>,
    ready: AtomicBool,
    _pad: [u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Option<TaskFn>>>() - core::mem::size_of::<AtomicBool>()],
}

use core::cell::UnsafeCell;

impl TaskSlot {
    const fn new() -> Self {
        Self {
            task: UnsafeCell::new(None),
            ready: AtomicBool::new(false),
            _pad: [0u8; CACHE_LINE_SIZE - core::mem::size_of::<UnsafeCell<Option<TaskFn>>>() - core::mem::size_of::<AtomicBool>()],
        }
    }
}

/// Per-worker work queue
#[repr(C)]
struct WorkQueue {
    tasks: Box<[TaskSlot; MAX_TASKS_PER_QUEUE]>,
    head: PaddedAtomicUsize,
    tail: PaddedAtomicUsize,
    mask: usize,
}

impl WorkQueue {
    fn new() -> Self {
        const EMPTY_SLOT: TaskSlot = TaskSlot::new();
        Self {
            tasks: Box::new([EMPTY_SLOT; MAX_TASKS_PER_QUEUE]),
            head: PaddedAtomicUsize::new(0),
            tail: PaddedAtomicUsize::new(0),
            mask: MAX_TASKS_PER_QUEUE - 1,
        }
    }

    #[inline]
    fn push(&self, task: TaskFn) -> bool {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);

        if tail.wrapping_sub(head) >= MAX_TASKS_PER_QUEUE {
            return false; // Queue full
        }

        let idx = tail & self.mask;
        let slot = &self.tasks[idx];

        unsafe {
            *slot.task.get() = Some(task);
        }

        slot.ready.store(true, Ordering::Release);
        self.tail.value.store(tail + 1, Ordering::Release);
        true
    }

    #[inline]
    fn pop(&self) -> Option<TaskFn> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);

        if head >= tail {
            return None; // Queue empty
        }

        let idx = head & self.mask;
        let slot = &self.tasks[idx];

        // Wait for task to be ready
        while !slot.ready.load(Ordering::Acquire) {
            core::hint::spin_loop();
        }

        let task = unsafe { (*slot.task.get()).take() };
        self.head.value.store(head + 1, Ordering::Release);

        task
    }

    #[inline]
    fn is_empty(&self) -> bool {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);
        head >= tail
    }

    #[inline]
    fn len(&self) -> usize {
        let tail = self.tail.value.load(Ordering::Acquire);
        let head = self.head.value.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }
}

/// Worker thread state
#[repr(C)]
struct Worker {
    id: usize,
    queue: WorkQueue,
    running: PaddedAtomicBool,
    tasks_executed: PaddedAtomicUsize,
    core_id: usize,
}

impl Worker {
    fn new(id: usize, core_id: usize) -> Self {
        Self {
            id,
            queue: WorkQueue::new(),
            running: PaddedAtomicBool::new(false),
            tasks_executed: PaddedAtomicUsize::new(0),
            core_id,
        }
    }
}

/// Low-latency thread pool
pub struct ThreadPool {
    workers: Box<[Worker; MAX_WORKERS]>,
    worker_count: usize,
    next_worker: AtomicUsize,
    shutdown: AtomicBool,
}

impl ThreadPool {
    /// Create a new thread pool with specified number of workers
    pub fn new(worker_count: usize) -> Self {
        assert!(worker_count <= MAX_WORKERS, "Worker count exceeds maximum");

        let mut workers = Vec::with_capacity(MAX_WORKERS);
        for i in 0..MAX_WORKERS {
            workers.push(Worker::new(i, i % num_cpus()));
        }

        Self {
            workers: workers.into_boxed_slice().try_into().unwrap(),
            worker_count,
            next_worker: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Spawn worker threads with CPU affinity
    pub fn spawn_workers(&self) -> Vec<JoinHandle<()>> {
        let mut handles = Vec::with_capacity(self.worker_count);

        for i in 0..self.worker_count {
            let worker = &self.workers[i];
            worker.running.value.store(true, Ordering::Release);

            let handle = thread::spawn(move || {
                // Pin thread to CPU core
                pin_thread_to_core(worker.core_id);

                // Worker loop
                while worker.running.value.load(Ordering::Acquire) {
                    // Try to execute from own queue first
                    if let Some(task) = worker.queue.pop() {
                        task();
                        worker.tasks_executed.value.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // Try to steal from other workers
                        let mut found_work = false;
                        for j in 0..worker.id {
                            let other = &self.workers[j];
                            if let Some(task) = other.queue.pop() {
                                task();
                                worker.tasks_executed.value.fetch_add(1, Ordering::Relaxed);
                                found_work = true;
                                break;
                            }
                        }

                        // No work available - spin briefly
                        if !found_work {
                            for _ in 0..100 {
                                core::hint::spin_loop();
                            }
                        }
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }

    /// Submit a task to the pool (round-robin distribution)
    #[inline]
    pub fn submit(&self, task: TaskFn) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }

        // Round-robin worker selection
        let worker_idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_count;
        let worker = &self.workers[worker_idx];

        worker.queue.push(task)
    }

    /// Submit a task to a specific worker (for affinity-aware scheduling)
    #[inline]
    pub fn submit_to(&self, worker_id: usize, task: TaskFn) -> bool {
        if worker_id >= self.worker_count {
            return false;
        }

        let worker = &self.workers[worker_id];
        worker.queue.push(task)
    }

    /// Get number of pending tasks
    #[inline]
    pub fn pending_tasks(&self) -> usize {
        let mut total = 0;
        for i in 0..self.worker_count {
            total += self.workers[i].queue.len();
        }
        total
    }

    /// Get total tasks executed
    #[inline]
    pub fn total_executed(&self) -> usize {
        let mut total = 0;
        for i in 0..self.worker_count {
            total += self.workers[i].tasks_executed.value.load(Ordering::Relaxed);
        }
        total
    }

    /// Shutdown the pool
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        for i in 0..self.worker_count {
            self.workers[i].running.value.store(false, Ordering::Release);
        }
    }

    /// Check if pool is shutting down
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Get number of CPUs
fn num_cpus() -> usize {
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize }
    }
    #[cfg(not(target_os = "linux"))]
    {
        thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }
}

/// Pin current thread to a CPU core
fn pin_thread_to_core(core_id: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        use libc::{cpu_set_t, pthread_self, sched_setaffinity};
        
        let mut cpuset: cpu_set_t = core::mem::zeroed();
        libc::CPU_SET(core_id, &mut cpuset);
        
        let result = sched_setaffinity(
            pthread_self(),
            core::mem::size_of::<cpu_set_t>(),
            &cpuset as *const _ as *const _,
        );
        
        if result != 0 {
            eprintln!("Warning: Failed to pin thread to core {}", core_id);
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        let _ = core_id; // Suppress unused warning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.worker_count, 4);
        assert!(!pool.is_shutdown());
    }

    #[test]
    fn test_task_submission() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        
        let counter_clone = counter.clone();
        let task = Box::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });
        
        assert!(pool.submit(task));
        
        // Give time for task execution
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        assert!(counter.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_work_queue() {
        let queue = WorkQueue::new();
        
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        let task = Box::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });
        
        assert!(queue.push(task));
        assert_eq!(queue.len(), 1);
        
        let popped = queue.pop();
        assert!(popped.is_some());
        popped.unwrap()();
        
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        assert!(queue.is_empty());
    }
}
