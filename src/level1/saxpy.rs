//! Performs a single precision AXPY operation: y := alpha * x + y.
//!
//! This function implements the BLAS [`saxpy`] routine, updating the vector `y`
//! by adding `alpha * x` elementwise over `n` entries with specified strides.
//!
//! # Arguments
//! - `n`     : Number of elements to process.
//! - `alpha` : Scalar multiplier for `x`.
//! - `x`     : Input slice containing vector elements.
//! - `incx`  : Stride between consecutive elements of `x`.
//! - `y`     : Input/output slice containing vector elements, updated in place.
//! - `incy`  : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place as `y[i] = alpha * x[i] + y[i]`.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`saxpy`] uses unrolled NEON SIMD instructions
//!   for optimized performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `alpha == 0.0`, the function returns immediately; no slice modification.
//!
//! # Author
//! Deval Deliwala


use core::arch::aarch64::{ 
    vld1q_f32, vdupq_n_f32, vfmaq_f32, vst1q_f32 
}; 
use crate::level1::assert_length_helpers::required_len_ok;


#[inline(always)]
pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: isize, y: &mut [f32], incy: isize) { 
    // quick return 
    if n == 0 || alpha == 0.0 { 
        return; 
    } 

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    unsafe { 
        let av = vdupq_n_f32(alpha); 

        // fast path 
        if incx == 1 && incy == 1 { 
            let mut i = 0; 
            while i + 16 <= n { 
                let x0 = vld1q_f32(x.as_ptr().add(i)); 
                let x1 = vld1q_f32(x.as_ptr().add(i + 4)); 
                let x2 = vld1q_f32(x.as_ptr().add(i + 8)); 
                let x3 = vld1q_f32(x.as_ptr().add(i + 12));

                let y0 = vld1q_f32(y.as_ptr().add(i)); 
                let y1 = vld1q_f32(y.as_ptr().add(i + 4)); 
                let y2 = vld1q_f32(y.as_ptr().add(i + 8)); 
                let y3 = vld1q_f32(y.as_ptr().add(i + 12)); 

                let r0 = vfmaq_f32(y0, av, x0); 
                let r1 = vfmaq_f32(y1, av, x1); 
                let r2 = vfmaq_f32(y2, av, x2); 
                let r3 = vfmaq_f32(y3, av, x3); 

                vst1q_f32(y.as_mut_ptr().add(i), r0);
                vst1q_f32(y.as_mut_ptr().add(i + 4), r1);
                vst1q_f32(y.as_mut_ptr().add(i + 8), r2);
                vst1q_f32(y.as_mut_ptr().add(i + 12), r3);

                i += 16; 
            }

            while i + 8 <= n { 
                let x0 = vld1q_f32(x.as_ptr().add(i)); 
                let x1 = vld1q_f32(x.as_ptr().add(i + 4)); 

                let y0 = vld1q_f32(y.as_ptr().add(i)); 
                let y1 = vld1q_f32(y.as_ptr().add(i + 4)); 

                let r0 = vfmaq_f32(y0, av, x0); 
                let r1 = vfmaq_f32(y1, av, x1); 

                vst1q_f32(y.as_mut_ptr().add(i), r0);
                vst1q_f32(y.as_mut_ptr().add(i + 4), r1);

                i += 8; 
            }

            // tail 
            while i < n { 
                let x0 = *x.as_ptr().add(i); 
                let y0 = y.as_mut_ptr().add(i); 
    
                *y0 += alpha * x0; 

                i += 1; 

            }
        } else { 
            // non unit stride 
            let stepx = if incx > 0 { incx as usize } else { (-incx) as usize }; 
            let stepy = if incy > 0 { incy as usize } else { (-incy) as usize }; 

            let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx }; 
            let mut iy = if incy >= 0 { 0usize } else { (n - 1) * stepy }; 

            for _ in 0..n { 
                let x0 = *x.as_ptr().add(ix); 
                let y0 = y.as_mut_ptr().add(iy); 

                *y0 += alpha * x0; 

                ix += stepx; 
                iy += stepy;  
            }
            
        }
    }
}
