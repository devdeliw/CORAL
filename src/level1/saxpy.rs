//! Performs a single precision AXPY operation: y := alpha * x + y.
//!
//! This function implements the BLAS [`saxpy`] routine, updating the vector `y`
//! by adding `alpha * x` elementwise over `n` entries with specified strides.
//!
//! # Arguments
//! - `n`     (usize)     : Number of elements to process.
//! - `alpha` (f32)       : Scalar multiplier for `x`.
//! - `x`     (&[f32])    : Input slice containing vector elements.
//! - `incx`  (usize)     : Stride between consecutive elements of `x`.
//! - `y`     (&mut [f32] : Input/output slice containing vector elements, updated in place.
//! - `incy`  (usize)     : Stride between consecutive elements of `y`.
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{ 
    vld1q_f32, 
    vdupq_n_f32, 
    vfmaq_f32, 
    vst1q_f32 
}; 
use crate::level1::assert_length_helpers::required_len_ok;


#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn saxpy(
    n       : usize, 
    alpha   : f32, 
    x       : &[f32], 
    incx    : usize, 
    y       : &mut [f32], 
    incy    : usize
) { 
    // quick return 
    if n == 0 || alpha == 0.0 { 
        return; 
    } 

    debug_assert!(incx > 0 && incy > 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    unsafe { 
        let av = vdupq_n_f32(alpha); 

        // fast path 
        if incx == 1 && incy == 1 { 
            let mut i = 0; 
            while i + 16 <= n { 
                let x0 = vld1q_f32(x.as_ptr().add(i + 0)); 
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

                vst1q_f32(y.as_mut_ptr().add(i + 0), r0);
                vst1q_f32(y.as_mut_ptr().add(i + 4), r1);
                vst1q_f32(y.as_mut_ptr().add(i + 8), r2);
                vst1q_f32(y.as_mut_ptr().add(i + 12), r3);

                i += 16; 
            }

            while i + 8 <= n { 
                let x0 = vld1q_f32(x.as_ptr().add(i + 0)); 
                let x1 = vld1q_f32(x.as_ptr().add(i + 4)); 

                let y0 = vld1q_f32(y.as_ptr().add(i)); 
                let y1 = vld1q_f32(y.as_ptr().add(i + 4)); 

                let r0 = vfmaq_f32(y0, av, x0); 
                let r1 = vfmaq_f32(y1, av, x1); 

                vst1q_f32(y.as_mut_ptr().add(i + 0), r0);
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
            let mut ix = 0; 
            let mut iy = 0; 

            for _ in 0..n { 
                let x0 = *x.as_ptr().add(ix); 
                let y0 = y.as_mut_ptr().add(iy); 

                *y0 += alpha * x0; 

                ix += incx; 
                iy += incy;  
            }
        }
    }
}
