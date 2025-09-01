//! Swaps elements of two double precision vectors.
//!
//! This function implements the BLAS [`dswap`] routine, exchanging elements of 
//! two input vectors `x` and `y` over `n` entries with specified strides.
//!
//! # Arguments 
//! - `n`    : Number of elements to swap. 
//! - `x`    : First input/output slice containing vector elements. 
//! - `incx` : Stride between consecutive elements of `x`. 
//! - `y`    : Second input/output slice containing vector elements. 
//! - `incy` : Stride between consecutive elements of `y`. 
//!
//! # Returns 
//! - Nothing. The contents of `x` and `y` are swapped in place.
//!
//! # Notes 
//! - For `incx == 1 && incy == 1`, [`dswap`] uses unrolled NEON SIMD instructions 
//!   for optimized performance on AArch64. 
//! - For non-unit or negative strides, the function falls back to scalar iteration. 
//! - If `n == 0`, the function returns immediately without modifying input slices.
//!
//! # Author 
//! Deval Deliwala


use core::arch::aarch64::{
    vld1q_f64, vst1q_f64
};
use crate::level1::assert_length_helpers::required_len_ok;


#[inline(always)]
pub fn dswap(n: usize, x: &mut [f64], incx: isize, y: &mut [f64], incy: isize) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");
    
    unsafe { 
        // fast path 
        if incx == 1 && incy == 1 { 
            let px = x.as_mut_ptr(); 
            let py = y.as_mut_ptr(); 

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

                vst1q_f64(py.add(i), ax0);
                vst1q_f64(py.add(i + 2), ax1);
                vst1q_f64(py.add(i + 4), ax2);
                vst1q_f64(py.add(i + 6), ax3);

                vst1q_f64(px.add(i), ay0);
                vst1q_f64(px.add(i + 2), ay1);
                vst1q_f64(px.add(i + 4), ay2);
                vst1q_f64(px.add(i + 6), ay3);

                i += 8; 
            }

            while i + 2 <= n { 
                let ax = vld1q_f64(px.add(i)); 
                let ay = vld1q_f64(py.add(i)); 
                vst1q_f64(px.add(i), ay);
                vst1q_f64(py.add(i), ax);

                i += 2; 
            }

            // tail 
            while i < n { 
                let a = *px.add(i); 
                *px.add(i) = *py.add(i); 
                *py.add(i) = a; 
                
                i += 1; 
            }
            return; 
        } 

        // non unit stride 
        let px = x.as_mut_ptr(); 
        let py = y.as_mut_ptr(); 

        let stepx = if incx > 0 { incx as usize } else { (-incx) as usize }; 
        let stepy = if incy > 0 { incy as usize } else { (-incy) as usize }; 

        let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx }; 
        let mut iy = if incy >= 0 { 0usize } else { (n - 1) * stepy }; 

        for _ in 0..n { 
            let a = *px.add(ix); 
            *px.add(ix) = *py.add(iy); 
            *py.add(iy) = a; 

            if incx >= 0 { ix += stepx } else { ix = ix.wrapping_sub(stepx) };
            if incy >= 0 { iy += stepy } else { iy = iy.wrapping_sub(stepy) }; 
        }
    }
}


