//! Computes the dot product of two double precision vectors.
//!
//! This function implements the BLAS [`ddot`] routine, returning
//! sum(x[i] * y[i]) over `n` elements of the input vectors `x` and `y`
//! with specified strides.
//!
//! # Arguments
//! - `n`    (usize)  : Number of elements in the vectors.
//! - `x`    (&[f64]) : Input slice containing the first vector.
//! - `incx` (usize)  : Stride between consecutive elements of `x`.
//! - `y`    (&[f64]) : Input slice containing the second vector.
//! - `incy` (usize)  : Stride between consecutive elements of `y`.
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{ 
    vld1q_f64, 
    vdupq_n_f64, 
    vfmaq_f64, 
    vaddvq_f64, 
    vaddq_f64, 
}; 
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn ddot(
    n       : usize, 
    x       : &[f64], 
    incx    : usize, 
    y       : &[f64], 
    incy    : usize
) -> f64 { 
    // quick return 
    if n == 0 { return 0.0; }

    debug_assert!(incx > 0 && incy > 0, "increments must be nonzero");
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
            let mut acc4 = vdupq_n_f64(0.0); 
            let mut acc5 = vdupq_n_f64(0.0); 
            let mut acc6 = vdupq_n_f64(0.0); 
            let mut acc7 = vdupq_n_f64(0.0); 

            let mut i = 0usize; 

            while i + 16 <= n { 
                let ax0 = vld1q_f64(px.add(i + 0)); 
                let ax1 = vld1q_f64(px.add(i + 2)); 
                let ax2 = vld1q_f64(px.add(i + 4)); 
                let ax3 = vld1q_f64(px.add(i + 6)); 
                let ax4 = vld1q_f64(px.add(i + 8)); 
                let ax5 = vld1q_f64(px.add(i + 10)); 
                let ax6 = vld1q_f64(px.add(i + 12)); 
                let ax7 = vld1q_f64(px.add(i + 14)); 

                let ay0 = vld1q_f64(py.add(i + 0)); 
                let ay1 = vld1q_f64(py.add(i + 2)); 
                let ay2 = vld1q_f64(py.add(i + 4)); 
                let ay3 = vld1q_f64(py.add(i + 6)); 
                let ay4 = vld1q_f64(py.add(i + 8)); 
                let ay5 = vld1q_f64(py.add(i + 10)); 
                let ay6 = vld1q_f64(py.add(i + 12)); 
                let ay7 = vld1q_f64(py.add(i + 14)); 

                // acc += ax * ay
                acc0 = vfmaq_f64(acc0, ax0, ay0); 
                acc1 = vfmaq_f64(acc1, ax1, ay1); 
                acc2 = vfmaq_f64(acc2, ax2, ay2); 
                acc3 = vfmaq_f64(acc3, ax3, ay3); 
                acc4 = vfmaq_f64(acc4, ax4, ay4); 
                acc5 = vfmaq_f64(acc5, ax5, ay5); 
                acc6 = vfmaq_f64(acc6, ax6, ay6); 
                acc7 = vfmaq_f64(acc7, ax7, ay7); 

                i += 16; 
            }

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
 
            let accv1   = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)); 
            let accv2   = vaddq_f64(vaddq_f64(acc4, acc5), vaddq_f64(acc6, acc7)); 
            let accv    = vaddq_f64(accv1, accv2); 
            let mut acc = vaddvq_f64(accv); // total 

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
        let mut ix  = 0;
        let mut iy  = 0;
        let mut acc = 0.0; 

        for _ in 0..n { 
            acc += *px.add(ix) * *py.add(iy); 
            ix += incx; 
            iy += incy; 
        }

        acc
    }
}

