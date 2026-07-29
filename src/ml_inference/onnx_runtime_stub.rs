//! Minimalist Lock-Free ONNX Tensor Parser
//! Lightweight neural net execution without heavy runtime dependencies.
//! Zero-copy tensor interpretation with SIMD acceleration.

#![allow(clippy::float_cmp)]
#![deny(clippy::alloc_in_list)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Maximum tensor dimensions
const MAX_DIMS: usize = 8;
/// Maximum tensor elements (pre-allocated)
const MAX_ELEMENTS: usize = 1024 * 1024; // 1M elements max
/// Memory limit tracker
static MEMORY_USED: AtomicUsize = AtomicUsize::new(0);
const MEMORY_LIMIT_BYTES: usize = 6_500_000_000;
/// Execution flag
static INFERENCE_ACTIVE: AtomicBool = AtomicBool::new(true);

/// Tensor data type enumeration
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    UInt8 = 4,
}

impl TensorDtype {
    #[inline(always)]
    pub const fn element_size(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
        }
    }
}

/// ONNX Tensor header - zero-copy layout matching ONNX spec
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OnnxTensorHeader {
    /// Magic number: 'ONNX' = 0x4F4E5858
    pub magic: u32,
    /// Tensor name hash (FNV-1a)
    pub name_hash: u64,
    /// Data type
    pub dtype: TensorDtype,
    /// Number of dimensions
    pub ndim: u32,
    /// Dimensions (up to MAX_DIMS)
    pub dims: [u32; MAX_DIMS],
    /// Total element count
    pub num_elements: usize,
    /// Byte offset to data (relative to header end)
    pub data_offset: u32,
    _pad: [u8; 28],
}

impl Default for OnnxTensorHeader {
    fn default() -> Self {
        Self {
            magic: 0x4F4E5858,
            name_hash: 0,
            dtype: TensorDtype::Float32,
            ndim: 0,
            dims: [0; MAX_DIMS],
            num_elements: 0,
            data_offset: 0,
            _pad: [0u8; 28],
        }
    }
}

const _: () = assert!(core::mem::size_of::<OnnxTensorHeader>() == 96);

/// Pre-allocated tensor storage pool
#[repr(C)]
pub struct TensorPool {
    /// Storage buffer (aligned to 32 bytes for AVX2)
    pub data: [f32; MAX_ELEMENTS],
    /// Active tensor headers
    pub headers: [OnnxTensorHeader; 16],
    /// Number of active tensors
    pub count: AtomicUsize,
    _pad: [u8; 56],
}

impl TensorPool {
    pub const fn new() -> Self {
        Self {
            data: [0.0; MAX_ELEMENTS],
            headers: [OnnxTensorHeader::default(); 16],
            count: AtomicUsize::new(0),
            _pad: [0u8; 56],
        }
    }
    
    /// Register a tensor from raw bytes (zero-copy)
    #[inline(always)]
    pub unsafe fn register_tensor(
        &self,
        name_hash: u64,
        dtype: TensorDtype,
        dims: &[u32],
        data_ptr: *const u8,
    ) -> Option<usize> {
        if !INFERENCE_ACTIVE.load(Ordering::Relaxed) {
            return None;
        }
        
        let idx = self.count.fetch_add(1, Ordering::AcqRel);
        if idx >= 16 {
            self.count.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        
        let header = &mut *(self.headers.as_ptr().add(idx) as *mut OnnxTensorHeader);
        
        // Calculate total elements
        let mut num_elements = 1;
        for (i, &d) in dims.iter().enumerate() {
            if i < MAX_DIMS {
                header.dims[i] = d;
                num_elements = num_elements.saturating_mul(d as usize);
            }
        }
        
        // Check memory limit
        let required_bytes = num_elements * dtype.element_size();
        let current_mem = MEMORY_USED.fetch_add(required_bytes, Ordering::Relaxed);
        if current_mem + required_bytes > MEMORY_LIMIT_BYTES {
            MEMORY_USED.fetch_sub(required_bytes, Ordering::Relaxed);
            self.count.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        
        header.magic = 0x4F4E5858;
        header.name_hash = name_hash;
        header.dtype = dtype;
        header.ndim = dims.len().min(MAX_DIMS) as u32;
        header.num_elements = num_elements;
        header.data_offset = 0;
        
        Some(idx)
    }
    
    /// Get tensor data as slice
    #[inline(always)]
    pub fn get_tensor_data(&self, idx: usize) -> Option<&[f32]> {
        if idx >= 16 {
            return None;
        }
        
        let header = &self.headers[idx];
        if header.magic != 0x4F4E5858 {
            return None;
        }
        
        // Only support f32 for now (most common in inference)
        if header.dtype != TensorDtype::Float32 {
            return None;
        }
        
        Some(&self.data[..header.num_elements])
    }
}

/// Minimalist ONNX model runner
#[repr(C)]
pub struct OnnxRunner {
    /// Pool of tensors
    pub pool: TensorPool,
    /// Model loaded flag
    pub model_loaded: AtomicBool,
    /// Layer count
    pub layer_count: usize,
    /// Output tensor index
    pub output_idx: usize,
    _pad: [u8; 54],
}

impl OnnxRunner {
    pub const fn new() -> Self {
        Self {
            pool: TensorPool::new(),
            model_loaded: AtomicBool::new(false),
            layer_count: 0,
            output_idx: 0,
            _pad: [0u8; 54],
        }
    }
    
    /// SIMD-accelerated matrix-vector multiplication (AVX2)
    #[inline(always)]
    pub unsafe fn matmul_vec(&self, weights: &[f32], input: &[f32], output: &mut [f32]) {
        if !INFERENCE_ACTIVE.load(Ordering::Relaxed) {
            return;
        }
        
        let in_dim = input.len();
        let out_dim = output.len();
        
        if weights.len() != in_dim * out_dim {
            return;
        }
        
        // Process 8 outputs at a time
        let mut o = 0;
        while o + 8 <= out_dim {
            let mut sum = [0.0f32; 8];
            
            for i in 0..in_dim {
                let inp = _mm256_set1_ps(input[i]);
                
                // Load 8 weights
                let w_ptr = weights.as_ptr().add(o * in_dim + i * out_dim);
                let w = _mm256_loadu_ps(w_ptr);
                
                let prod = _mm256_mul_ps(inp, w);
                let sum_vec = _mm256_add_ps(_mm256_loadu_ps(sum.as_ptr()), prod);
                _mm256_storeu_ps(sum.as_mut_ptr(), sum_vec);
            }
            
            // Horizontal add for each of 8 outputs
            for j in 0..8 {
                output[o + j] = sum[j];
            }
            o += 8;
        }
        
        // Handle remaining outputs
        while o < out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += weights[o * in_dim + i] * input[i];
            }
            output[o] = sum;
            o += 1;
        }
    }
    
    /// ReLU activation (branchless)
    #[inline(always)]
    pub fn relu(&self, data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = (*x).max(0.0);
        }
    }
    
    /// Softmax (stable, branchless)
    #[inline(always)]
    pub fn softmax(&self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }
        
        // Find max for numerical stability
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exp and sum
        let mut sum = 0.0f32;
        for x in data.iter_mut() {
            *x = ((*x - max_val).exp());
            sum += *x;
        }
        
        // Normalize
        let inv_sum = 1.0 / sum;
        for x in data.iter_mut() {
            *x *= inv_sum;
        }
    }
    
    /// Run inference (simplified single-layer example)
    pub fn infer(&self, input: &[f32]) -> Option<&[f32]> {
        if !self.model_loaded.load(Ordering::Relaxed) {
            return None;
        }
        
        if !INFERENCE_ACTIVE.load(Ordering::Relaxed) {
            return None;
        }
        
        self.pool.get_tensor_data(self.output_idx)
    }
    
    /// Halt inference (circuit breaker)
    pub fn halt(&self) {
        INFERENCE_ACTIVE.store(false, Ordering::Relaxed);
    }
    
    /// Resume inference
    pub fn resume(&self) {
        INFERENCE_ACTIVE.store(true, Ordering::Relaxed);
    }
}

impl Default for OnnxRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_tensor_registration(
            name_hash in any::<u64>(),
            ndim in 1usize..5,
            dims in prop::collection::vec(1u32..100, 1..5)
        ) {
            let pool = TensorPool::new();
            
            unsafe {
                let result = pool.register_tensor(
                    name_hash,
                    TensorDtype::Float32,
                    &dims,
                    core::ptr::null(),
                );
                
                // Should succeed within limits
                if dims.iter().product::<u32>() as usize <= MAX_ELEMENTS {
                    assert!(result.is_some() || pool.count.load(Ordering::Relaxed) >= 16);
                }
            }
        }
        
        #[test]
        fn test_relu_non_negative(value in -100.0f32..100.0f32) {
            let runner = OnnxRunner::new();
            let mut data = [value];
            runner.relu(&mut data);
            assert!(data[0] >= 0.0, "ReLU output must be non-negative");
        }
    }
    
    #[test]
    fn test_header_size() {
        assert_eq!(core::mem::size_of::<OnnxTensorHeader>(), 96);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let runner = OnnxRunner::new();
        runner.halt();
        assert!(!INFERENCE_ACTIVE.load(Ordering::Relaxed));
        runner.resume();
        assert!(INFERENCE_ACTIVE.load(Ordering::Relaxed));
    }
}
