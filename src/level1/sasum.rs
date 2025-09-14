//! Computes the sum of absolute values of elements in a single precision vector. 
//!
//! This function implements the BLAS [`sasum`] routine, returning sum(|x[i]|) over 
//! `n` elements of the input vector `x` with a specified stride. 
//!
//! # Arguments 
//! - `n`    (usize)  : Number of elements to sum. 
//! - `x`    (&[f32]) : Input slice containing vector elements 
//! - `incx` (usize)  : Stride between consecutive elements of `x` 
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{ 
    vld1q_f32, 
    vdupq_n_f32,
    vaddq_f32, 
    vaddvq_f32,
    vabsq_f32,
}; 
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline]
#[cfg(target_arch = "aarch64")]
pub fn sasum(
    n       : usize, 
    x       : &[f32],
    incx    : usize
) -> f32 {
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

                // acc += |x| 
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
            let mut ix = 0; 
            for _ in 0..n {
                res += (*x.get_unchecked(ix)).abs();
                
                ix += incx; 
            }
        }
    }

    res
}
