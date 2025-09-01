//! Scales a complex single precision vector by a real scalar.
//!
//! This function implements the BLAS [`csscal`] routine, multiplying each complex element 
//! of the input vector `x` by the real scalar `alpha` over `n` entries with a specified stride.
//!
//! # Arguments 
//! - `n`     : Number of complex elements to scale. 
//! - `alpha` : Real scalar multiplier. 
//! - `x`     : Input/output slice containing interleaved complex vector elements 
//!             `[re0, im0, re1, im1, ...]`. 
//! - `incx`  : Stride between consecutive complex elements of `x` 
//!             (measured in complex numbers; every step advances two scalar idxs). 
//!
//! # Returns 
//! - Nothing. The contents of `x` are updated in place as `x[i] = alpha * x[i]`. 
//!
//! # Notes 
//! - For `incx == 1`, [`csscal`] uses NEON SIMD instructions for optimized performance on AArch64. 
//! - For non unit strides, the function falls back to a scalar loop. 
//! - If `n == 0` or `incx <= 0`, the function returns immediately; no slice modification. 
//!
//! # Author 
//! Deval Deliwala


use core::arch::aarch64::{ 
    vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32
}; 


pub fn csscal(n: usize, alpha: f32, x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let a_v   = vdupq_n_f32(alpha);
            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v   = vld1q_f32(p);
                let out = vmulq_f32(v, a_v);

                vst1q_f32(p, out);

                p = p.add(4);
                i += 4;
            }
            while i < 2 * n {
                *p        = *p * alpha;
                *p.add(1) = *p.add(1) * alpha;

                p = p.add(2);
                i += 2;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let step    = 2 * incx as usize;
            let mut idx = 0usize;
            let p       = x.as_mut_ptr();

            for _ in 0..n {
                *p.add(idx)     = *p.add(idx) * alpha;
                *p.add(idx + 1) = *p.add(idx + 1) * alpha;

                idx += step;
            }
        }
    }
}
