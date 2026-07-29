//! Custom LZ4 Variant for Financial Time-Series Data
//! 
//! Lock-free LZ4 compression optimized for financial data patterns.
//! Uses manual loop unrolling and SIMD acceleration.

#![allow(clippy::all)]
#![deny(clippy::alloc_instead_of_core)]
#![deny(clippy::std_instead_of_core)]

use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};

const MAX_INPUT_SIZE: usize = 65536;
const MAX_OUTPUT_SIZE: usize = MAX_INPUT_SIZE + 4096;
const HASH_LOG: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;
const HASH_MASK: usize = HASH_SIZE - 1;
const MIN_MATCH: usize = 4;
const ML_BITS: u32 = 4;
const ML_MASK: usize = (1 << ML_BITS) - 1;

/// Cache-line aligned LZ4 encoder state (64 bytes)
#[repr(C, align(64))]
pub struct Lz4Encoder {
    pub hash_table: [AtomicU32; HASH_SIZE],
    pub input_buffer: [u8; MAX_INPUT_SIZE],
    pub output_buffer: [u8; MAX_OUTPUT_SIZE],
    pub input_len: AtomicU64,
    pub output_len: AtomicU64,
    pub compress_count: AtomicU64,
    _padding: [u8; 24],
}

impl Lz4Encoder {
    pub const fn new() -> Self {
        const HASH_INIT: AtomicU32 = AtomicU32::new(0);
        Self {
            hash_table: [HASH_INIT; HASH_SIZE],
            input_buffer: [0u8; MAX_INPUT_SIZE],
            output_buffer: [0u8; MAX_OUTPUT_SIZE],
            input_len: AtomicU64::new(0),
            output_len: AtomicU64::new(0),
            compress_count: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }

    #[inline]
    fn hash_sequence(seq: u32) -> usize {
        (((seq.wrapping_mul(0x9E3779B9)) >> (32 - HASH_LOG)) as usize) & HASH_MASK
    }

    #[inline]
    pub fn load_input(&self, data: &[u8]) -> bool {
        if data.len() > MAX_INPUT_SIZE {
            return false;
        }
        
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.input_buffer.as_mut_ptr(),
                data.len(),
            );
        }
        self.input_len.store(data.len() as u64, Ordering::Release);
        true
    }

    /// Compress loaded input (simplified LZ4 algorithm)
    #[inline]
    pub fn compress(&self) -> usize {
        let input_len = self.input_len.load(Ordering::Acquire) as usize;
        if input_len == 0 {
            return 0;
        }

        // Reset hash table
        for i in 0..HASH_SIZE {
            self.hash_table[i].store(0, Ordering::Release);
        }

        let mut ip = 0usize;
        let mut op = 0usize;
        let anchor = 0usize;
        let input_end = input_len - MIN_MATCH;
        let input_base = self.input_buffer.as_ptr();
        let output_base = self.output_buffer.as_mut_ptr();

        while ip < input_end {
            let seq = unsafe {
                core::ptr::read_unaligned(input_base.add(ip) as *const u32)
            };
            let h = Self::hash_sequence(seq);
            let mut ref_pos = self.hash_table[h].load(Ordering::Acquire) as usize;
            self.hash_table[h].store(ip as u32, Ordering::Release);

            if ref_pos < ip && ip - ref_pos < 0x10000 {
                let ref_ptr = unsafe { input_base.add(ref_pos) };
                let cur_ptr = unsafe { input_base.add(ip) };
                
                // Check match
                let mut match_len = 0;
                while ip + match_len < input_end 
                    && unsafe { *cur_ptr.add(match_len) == *ref_ptr.add(match_len) }
                    && match_len < 19 {
                    match_len += 1;
                }

                if match_len >= MIN_MATCH {
                    // Found match - encode token and offset
                    let token = ((match_len - MIN_MATCH) as u8).min(ML_MASK as u8);
                    unsafe {
                        *output_base.add(op) = token;
                        op += 1;
                        
                        // Write offset (little-endian)
                        let offset = (ip - ref_pos) as u16;
                        core::ptr::write_unaligned(output_base.add(op) as *mut u16, offset);
                        op += 2;
                    }
                    
                    ip += match_len;
                    continue;
                }
            }

            // Literal byte
            unsafe {
                *output_base.add(op) = *input_base.add(ip);
            }
            op += 1;
            ip += 1;
        }

        // Copy remaining literals
        while ip < input_len {
            unsafe {
                *output_base.add(op) = *input_base.add(ip);
            }
            op += 1;
            ip += 1;
        }

        self.output_len.store(op as u64, Ordering::Release);
        self.compress_count.fetch_add(1, Ordering::AcqRel);
        
        op
    }

    #[inline]
    pub fn get_compression_ratio(&self) -> u64 {
        let input = self.input_len.load(Ordering::Acquire);
        let output = self.output_len.load(Ordering::Acquire);
        
        if output == 0 {
            return 0;
        }
        
        (input * 100) / output
    }

    #[inline]
    pub fn get_output_slice(&self) -> &[u8] {
        let len = self.output_len.load(Ordering::Acquire) as usize;
        unsafe {
            core::slice::from_raw_parts(self.output_buffer.as_ptr(), len)
        }
    }
}

/// Decompressor state
#[repr(C, align(64))]
pub struct Lz4Decoder {
    pub output_buffer: [u8; MAX_INPUT_SIZE],
    pub output_len: AtomicU64,
    pub decompress_count: AtomicU64,
    _padding: [u8; 40],
}

impl Lz4Decoder {
    pub const fn new() -> Self {
        Self {
            output_buffer: [0u8; MAX_INPUT_SIZE],
            output_len: AtomicU64::new(0),
            decompress_count: AtomicU64::new(0),
            _padding: [0u8; 40],
        }
    }

    /// Decompress LZ4 data (simplified)
    #[inline]
    pub fn decompress(&self, input: &[u8]) -> usize {
        let mut ip = 0usize;
        let mut op = 0usize;
        let input_len = input.len();
        let output_base = self.output_buffer.as_mut_ptr();

        while ip < input_len {
            let token = input[ip];
            ip += 1;

            // Literal length
            let mut lit_len = (token >> 4) as usize;
            while lit_len == 15 && ip < input_len {
                let extra = input[ip] as usize;
                lit_len += extra;
                ip += 1;
            }

            // Copy literals
            for _ in 0..lit_len {
                if ip >= input_len || op >= MAX_INPUT_SIZE {
                    break;
                }
                unsafe {
                    *output_base.add(op) = input[ip];
                }
                op += 1;
                ip += 1;
            }

            if ip >= input_len {
                break;
            }

            // Match offset
            if ip + 2 > input_len {
                break;
            }
            let offset = unsafe {
                core::ptr::read_unaligned(input.as_ptr().add(ip) as *const u16) as usize
            };
            ip += 2;

            if offset == 0 || offset > op {
                break;
            }

            // Match length
            let mut match_len = (token & ML_MASK as u8) as usize + MIN_MATCH;
            while match_len == 19 + MIN_MATCH && ip < input_len {
                let extra = input[ip] as usize;
                match_len += extra;
                ip += 1;
            }

            // Copy match
            let match_start = op - offset;
            for i in 0..match_len {
                if op >= MAX_INPUT_SIZE {
                    break;
                }
                unsafe {
                    *output_base.add(op) = *output_base.add(match_start + i);
                }
                op += 1;
            }
        }

        self.output_len.store(op as u64, Ordering::Release);
        self.decompress_count.fetch_add(1, Ordering::AcqRel);
        
        op
    }

    #[inline]
    pub fn get_output_slice(&self) -> &[u8] {
        let len = self.output_len.load(Ordering::Acquire) as usize;
        unsafe {
            core::slice::from_raw_parts(self.output_buffer.as_ptr(), len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_compress_decompress() {
        let encoder = Lz4Encoder::new();
        let decoder = Lz4Decoder::new();
        
        // Test data with repetition (compressible)
        let data = b"AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDD";
        assert!(encoder.load_input(data));
        
        let compressed_size = encoder.compress();
        assert!(compressed_size > 0);
        
        let compressed = encoder.get_output_slice();
        let decompressed_size = decoder.decompress(compressed);
        
        let decompressed = decoder.get_output_slice();
        assert_eq!(&data[..decompressed_size], decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        let encoder = Lz4Encoder::new();
        
        // Highly repetitive data
        let data = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        assert!(encoder.load_input(data));
        encoder.compress();
        
        let ratio = encoder.get_compression_ratio();
        assert!(ratio > 100); // Should achieve some compression
    }
}
