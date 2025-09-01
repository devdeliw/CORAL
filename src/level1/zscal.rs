//! Scales a complex double precision vector by a complex scalar.
//!
//! This function implements the BLAS [`zscal`] routine, multiplying each complex element 
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
//! - For `incx == 1`, [`zscal`] uses NEON SIMD instructions for optimized performance on AArch64. 
//! - For non unit strides, the function falls back to a scalar loop. 
//! - If `n == 0` or `incx <= 0`, the function returns immediately; no slice modification. 
//!
//! # Author 
//! Deval Deliwala


use core::arch::aarch64::{
    vdupq_n_f64, vld1q_f64, vst1q_f64, vmulq_f64,
    vextq_f64, vfmsq_f64, vfmaq_f64, vzip1q_f64
};


#[inline(always)]
pub fn zscal(n: usize, alpha: [f64; 2], x: &mut [f64], incx: isize) {
    if n == 0 || incx <= 0 { return; }

    let ar = alpha[0];
    let ai = alpha[1];

    // fast path 
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let ar_v = vdupq_n_f64(ar);
            let ai_v = vdupq_n_f64(ai);

            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v0  = vld1q_f64(p);     
                let s0  = vextq_f64(v0, v0, 1); 
                let re0 = vfmsq_f64(vmulq_f64(v0, ar_v), s0, ai_v);
                let im0 = vfmaq_f64(vmulq_f64(s0, ar_v), v0, ai_v);
                let out0 = vzip1q_f64(re0, im0);
                vst1q_f64(p, out0);

                let v1  = vld1q_f64(p.add(2));  
                let s1  = vextq_f64(v1, v1, 1); 
                let re1 = vfmsq_f64(vmulq_f64(v1, ar_v), s1, ai_v);
                let im1 = vfmaq_f64(vmulq_f64(s1, ar_v), v1, ai_v);
                let out1 = vzip1q_f64(re1, im1); 
                vst1q_f64(p.add(2), out1);

                p = p.add(4);
                i += 4;
            }
            while i + 2 <= 2 * n {
                let v   = vld1q_f64(p);         
                let s   = vextq_f64(v, v, 1);   
                let re  = vfmsq_f64(vmulq_f64(v, ar_v), s, ai_v);
                let im  = vfmaq_f64(vmulq_f64(s, ar_v), v, ai_v);
                let out = vzip1q_f64(re, im);   
                vst1q_f64(p, out);
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
                let re = *p.add(idx);
                let im = *p.add(idx + 1);
                *p.add(idx)     = ar * re - ai * im;
                *p.add(idx + 1) = ar * im + ai * re;
                idx += step;
            }
        }
    }
}
