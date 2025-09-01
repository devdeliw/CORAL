//! Computes the dot product of two double precision vectors.
//!
//! This function implements the BLAS [`ddot`] routine, returning
//! sum(x[i] * y[i]) over `n` elements of the input vectors `x` and `y`
//! with specified strides.
//!
//! # Arguments
//! - `n`    : Number of elements in the vectors.
//! - `x`    : Input slice containing the first vector.
//! - `incx` : Stride between consecutive elements of `x`.
//! - `y`    : Input slice containing the second vector.
//! - `incy` : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - `f64` dot product of the selected vector elements.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`ddot`] uses unrolled NEON SIMD instructions
//!   for optimized performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0`, the function returns `0.0f64`.
//!
//! # Author
//! Deval Deliwala


use core::arch::aarch64::{ 
    vld1q_f64, vdupq_n_f64, vfmaq_f64, vaddvq_f64, vaddq_f64, 
}; 
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline] 
pub fn ddot(n: usize, x: &[f64], incx: isize, y: &[f64], incy: isize) -> f64 { 
    // quick return 
    if n == 0 { return 0.0; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_ptr(); 
    let py = y.as_ptr(); 

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe { 
            let mut acc0 = vdupq_n_f64(0.0); 
            let mut acc1 = vdupq_n_f64(0.0); 
            let mut acc2 = vdupq_n_f64(0.0); 
            let mut acc3 = vdupq_n_f64(0.0); 

            let mut i = 0usize; 

            while i + 8 <= n { 
                let ax0 = vld1q_f64(px.add(i)); 
                let ax1 = vld1q_f64(px.add(i + 2)); 
                let ax2 = vld1q_f64(px.add(i + 4)); 
                let ax3 = vld1q_f64(px.add(i + 6)); 

                let ay0 = vld1q_f64(py.add(i)); 
                let ay1 = vld1q_f64(py.add(i + 2)); 
                let ay2 = vld1q_f64(py.add(i + 4)); 
                let ay3 = vld1q_f64(py.add(i + 6)); 

                acc0 = vfmaq_f64(acc0, ax0, ay0); 
                acc1 = vfmaq_f64(acc1, ax1, ay1); 
                acc2 = vfmaq_f64(acc2, ax2, ay2); 
                acc3 = vfmaq_f64(acc3, ax3, ay3); 

                i += 8; 
            }

            while i + 2 <= n { 
                let ax = vld1q_f64(px.add(i)); 
                let ay = vld1q_f64(py.add(i));
                acc0   = vfmaq_f64(acc0, ax, ay); 

                i += 2; 
            }

            let accv    = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)); 
            let mut acc = vaddvq_f64(accv); 

            // tail 
            while i < n { 
                acc += *px.add(i) * *py.add(i); 

                i += 1
            } 

            return acc; 
        } 
    }
    // non unit stride 
    unsafe { 
        let mut ix: isize = if incx >= 0 { 0 } else { ((n - 1) as isize) * (-incx) }; 
        let mut iy: isize = if incy >= 0 { 0 } else { ((n - 1) as isize) * (-incy) }; 

        let mut acc = 0.0f64; 
        for _ in 0..n { 
            acc += *px.offset(ix) * *py.offset(iy); 
            ix += incx; 
            iy += incy; 
        }

        acc
    }
}


