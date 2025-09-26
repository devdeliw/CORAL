//! SWAP Swaps elements of two single precision vectors.
//!
//! This function implements the BLAS [`sswap`] routine, exchanging elements of 
//! two input vectors `x` and `y` over `n` entries with specified strides.
//!
//! # Arguments 
//! - `n`    (usize)      : Number of elements to swap. 
//! - `x`    (&mut [f32]) : First input/output slice containing vector elements. 
//! - `incx` (usize)      : Stride between consecutive elements of `x`. 
//! - `y`    (&mut [f32]) : Second input/output slice containing vector elements. 
//! - `incy` (usize)      : Stride between consecutive elements of `y`. 
//!
//! # Returns 
//! - Nothing. The contents of `x` and `y` are swapped in place.
//!
//! # Notes 
//! - For `incx == 1 && incy == 1`, [`sswap`] uses unrolled NEON SIMD instructions 
//!   for optimized performance on AArch64. 
//! - For non-unit or negative strides, the function falls back to scalar iteration. 
//! - If `n == 0`, the function returns immediately without modifying input slices.
//!
//! # Author 
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f32, 
    vst1q_f32
};
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline(always)]
#[cfg(target_arch = "aarch64")] 
pub fn sswap(
    n       : usize,
    x       : &mut [f32], 
    incx    : usize, 
    y       : &mut [f32], 
    incy    : usize
) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx > 0 && incy > 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");
    
    unsafe { 
        // fast path 
        if incx == 1 && incy == 1 { 
            let px = x.as_mut_ptr(); 
            let py = y.as_mut_ptr(); 

            let mut i = 0; 

            while i + 16 <= n { 
                let ax0 = vld1q_f32(px.add(i)); 
                let ax1 = vld1q_f32(px.add(i + 4)); 
                let ax2 = vld1q_f32(px.add(i + 8)); 
                let ax3 = vld1q_f32(px.add(i + 12)); 

                let ay0 = vld1q_f32(py.add(i)); 
                let ay1 = vld1q_f32(py.add(i + 4)); 
                let ay2 = vld1q_f32(py.add(i + 8)); 
                let ay3 = vld1q_f32(py.add(i + 12)); 

                // swap
                vst1q_f32(py.add(i), ax0);
                vst1q_f32(py.add(i + 4),  ax1);
                vst1q_f32(py.add(i + 8),  ax2);
                vst1q_f32(py.add(i + 12), ax3);

                vst1q_f32(px.add(i), ay0);
                vst1q_f32(px.add(i + 4),  ay1);
                vst1q_f32(px.add(i + 8),  ay2);
                vst1q_f32(px.add(i + 12), ay3);

                i += 16; 
            }

            while i + 4 <= n { 
                let ax = vld1q_f32(px.add(i)); 
                let ay = vld1q_f32(py.add(i)); 
                vst1q_f32(px.add(i), ay);
                vst1q_f32(py.add(i), ax);

                i += 4; 
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

        let mut ix = 0; 
        let mut iy = 0; 

        for _ in 0..n { 
            let a = *px.add(ix);
            *px.add(ix) = *py.add(iy); 
            *py.add(iy) = a; 

            ix += incx; 
            iy += incy; 
        }
    }
}


