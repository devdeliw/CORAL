//! Computes the sum of absolute values of elements in a complex double precision vector. 
//!
//! This function implements the BLAS [`dzasum`] routine, returning sum(|Re(x[i])| + |Im(x[i])|) 
//! over `n` elements of the input complex vector `x` with a specified stride. 
//!
//! # Arguments 
//! - `n`    : Number of elements to sum. 
//! - `x`    : Input slice containing interleaved complex vector elements 
//!            `[re0, im0, re1, im1, ...]` 
//! - `incx` : Stride between consecutive complex elements of `x` 
//!            (measured in complex numbers; every step advances two scalar idxs) 
//!
//! # Returns 
//! - `f64` sum of absolute values of the real and imaginary parts of selected vector elements. 
//!
//! # Notes 
//! - For `incx == 1`, [`dzasum`] uses unrolled NEON SIMD instructions for optimized 
//!   performance on AArch64. 
//! - For non-unit strides the function falls back to a scalar loop
//! - If `n == 0 || incx == 0`, returns `0.0f64`
//! 
//! # Author 
//! Deval Deliwala

use core::arch::aarch64::{ 
    vld1q_f64, vdupq_n_f64, vaddq_f64, vaddvq_f64, vabsq_f64,
}; 
use crate::level1::assert_length_helpers::required_len_ok_cplx; 


#[inline]
pub fn dzasum(n: usize, x: &[f64], incx: isize) -> f64 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");

    unsafe {
        // fast path 
        if incx == 1 {
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            let mut acc2 = vdupq_n_f64(0.0);
            let mut acc3 = vdupq_n_f64(0.0);

            let mut i = 0;
            let end = 2 * n;
            while i + 8 <= end {
                let v0 = vabsq_f64(vld1q_f64(x.as_ptr().add(i)));
                let v1 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 2)));
                let v2 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 4)));
                let v3 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 6)));

                acc0 = vaddq_f64(acc0, v0);
                acc1 = vaddq_f64(acc1, v1);
                acc2 = vaddq_f64(acc2, v2);
                acc3 = vaddq_f64(acc3, v3);

                i += 8;
            }

            while i + 2 <= end { 
                let v = vabsq_f64(vld1q_f64(x.as_ptr().add(i))); 
                acc0  = vaddq_f64(acc0, v); 

                i += 2; 
                
            }

            res += vaddvq_f64(acc0) 
                 + vaddvq_f64(acc1)
                 + vaddvq_f64(acc2)
                 + vaddvq_f64(acc3); 

            while i < end {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;

            let mut idx = if incx > 0 { 0usize } else { 2 * (n - 1) * step } as isize;
            let delta   = if incx > 0 { (2 * step) as isize } else { -((2 * step) as isize) };

            for _ in 0..n {
                let i = idx as usize;
                let re = *x.get_unchecked(i);
                let im = *x.get_unchecked(i + 1);
                res += re.abs() + im.abs();
                idx += delta;
            }
        }
    }

    res
}


