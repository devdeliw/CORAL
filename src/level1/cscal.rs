//! Scales a complex single precision vector by a complex scalar.
//!
//! This function implements the BLAS [`cscal`] routine, multiplying each complex element 
//! of the input vector `x` by the complex scalar `alpha` over `n` entries with a specified stride.
//!
//! # Arguments 
//! - `n`     : Number of complex elements to scale. 
//! - `alpha` : Complex scalar multiplier given as `[real, imag]`. 
//! - `x`     : Input/output slice containing interleaved complex vector elements 
//!             `[re0, im0, re1, im1, ...]`. 
//! - `incx`  : Stride between consecutive complex elements of `x` 
//!             (measured in complex numbers; every step advances two scalar idxs). 
//!
//! # Returns 
//! - Nothing. The contents of `x` are updated in place as `x[i] = alpha * x[i]`. 
//!
//! # Notes 
//! - For `incx == 1`, [`cscal`] uses NEON SIMD instructions for optimized performance on AArch64. 
//! - For non unit strides, the function falls back to a scalar loop. 
//! - If `n == 0` or `incx <= 0`, the function returns immediately; no slice modification. 
//!
//! # Author 
//! Deval Deliwala


use core::arch::aarch64::{
    vdupq_n_f32, vld1q_f32, vst1q_f32, vmulq_f32,
    vrev64q_f32, vfmsq_f32, vfmaq_f32, vzip1q_f32, 
    vzip2q_f32, vcombine_f32, vget_low_f32,
};


#[inline(always)]
pub fn cscal(n: usize, alpha: [f32; 2], x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    let a_real = alpha[0];
    let a_imag = alpha[1];

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();

            let a_real_v = vdupq_n_f32(a_real);
            let a_imag_v = vdupq_n_f32(a_imag);

            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v_main = vld1q_f32(p);
                let v_swap = vrev64q_f32(v_main);

                let re = vfmsq_f32(vmulq_f32(v_main, a_real_v), v_swap, a_imag_v);
                let im = vfmaq_f32(vmulq_f32(v_swap, a_real_v), v_main, a_imag_v);

                let zip_lo = vzip1q_f32(re, im);
                let zip_hi = vzip2q_f32(re, im);
                let out = vcombine_f32(vget_low_f32(zip_lo), vget_low_f32(zip_hi));

                vst1q_f32(p, out);

                p = p.add(4);
                i += 4;
            }

            // tail
            while i < 2 * n {
                let real = *p;
                let imag = *p.add(1);
                *p        = a_real * real - a_imag * imag;
                *p.add(1) = a_real * imag + a_imag * real;
                p = p.add(2);
                i += 2;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let step = 2 * incx as usize;
            let mut idx = 0usize;
            let p = x.as_mut_ptr();
            for _ in 0..n {
                let real = *p.add(idx);
                let imag = *p.add(idx + 1);
                *p.add(idx)     = a_real * real - a_imag * imag;
                *p.add(idx + 1) = a_real * imag + a_imag * real;
                idx += step;
            }
        }
    }
}
