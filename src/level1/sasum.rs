//! Computes the sum of absolute values of elements in a single precision vector. 
//!
//! This function implements the BLAS [`sasum`] routine, returning sum(|x[i]|) over 
//! `n` elements of the input vector `x` with a specified stride. 
//!
//! # Arguments 
//! - `n`    : Number of elements to sum. 
//! - `x`    : Input slice containing vector elements 
//! - `incx` : Stride between consecutive elements of `x` 
//!
//! # Returns 
//! - `f32` sum of absolute values of selected vector elements. 
//!
//! # Notes 
//! - For `incx == 1`, [`sasum`] uses unrolled NEON SIMD instructions for optimized 
//!   performance on AArch64. 
//! - For non-unit strides the function falls back to a scalar loop
//! - If `n == 0 || incx == 0`, returns `0.0f32`
//! 
//! # Author 
//! Deval Deliwala 


use core::arch::aarch64::{ 
    vld1q_f32, vdupq_n_f32, vaddq_f32, vaddvq_f32, vabsq_f32,
}; 
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline]
pub fn sasum(n: usize, x: &[f32], incx: isize) -> f32 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    unsafe {
        // fast path 
        if incx == 1 {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0;
            while i + 16 <= n {
                let v0 = vabsq_f32(vld1q_f32(x.as_ptr().add(i)));
                let v1 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 4)));
                let v2 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 8)));
                let v3 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 12)));

                acc0 = vaddq_f32(acc0, v0);
                acc1 = vaddq_f32(acc1, v1);
                acc2 = vaddq_f32(acc2, v2);
                acc3 = vaddq_f32(acc3, v3);

                i += 16;
            }

            while i + 4 <= n { 
                let v = vabsq_f32(vld1q_f32(x.as_ptr().add(i))); 
                acc0  = vaddq_f32(acc0, v); 

                i += 4; 
            }

            res += vaddvq_f32(acc0) 
                 + vaddvq_f32(acc1) 
                 + vaddvq_f32(acc2) 
                 + vaddvq_f32(acc3);

            while i < n {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;

            let mut idx = if incx > 0 { 0usize } else { (n - 1) * step } as isize;
            let delta   = if incx > 0 { step as isize } else { -(step as isize) };

            for _ in 0..n {
                let i = idx as usize;
                res += (*x.get_unchecked(i)).abs();
                idx += delta;
            }
        }
    }

    res
}
