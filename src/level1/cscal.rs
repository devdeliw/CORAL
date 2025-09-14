//! Scales a complex single precision vector by a complex scalar.
//!
//! This function implements the BLAS [`cscal`] routine, multiplying each complex element 
//! of the input vector `x` by the complex scalar `alpha` over `n` entries with a specified stride.
//!
//! # Arguments 
//! - `n`     (usize)      : Number of complex elements to scale. 
//! - `alpha` ([f32; 2])   : Complex scalar multiplier given as `[real, imag]`. 
//! - `x`     (&mut [f32]) : Input/output slice containing interleaved complex vector elements 
//!                        | `[re0, im0, re1, im1, ...]`. 
//! - `incx`  (usize)      : Stride between consecutive complex elements of `x` 
//!                          (measured in complex numbers; every step advances two scalar idxs). 
//!
//! # Returns 
//! - Nothing. The contents of `x` are updated in place as `x[i] = alpha * x[i]`. 
//!
//! # Notes 
//! - For `incx == 1`, [`cscal`] uses NEON SIMD instructions for optimized performance on AArch64. 
//! - For non unit strides, the function falls back to a scalar loop. 
//! - If `n == 0` or `incx == 0`, the function returns immediately; no slice modification. 
//!
//! # Author 
//! Deval Deliwala

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vdupq_n_f32,
    vmulq_f32,
    vfmsq_f32, 
    vfmaq_f32,
    vld2q_f32, 
    vst2q_f32, 
    float32x4x2_t
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn cscal(
    n       : usize, 
    alpha   : [f32; 2],
    x       : &mut [f32], 
    incx    : usize
) {
    // quick return
    if n == 0 || incx == 0 { return; }

    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");


    let a_real = alpha[0];
    let a_imag = alpha[1];

    // fast path 
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();

            let ar = vdupq_n_f32(a_real);
            let ai = vdupq_n_f32(a_imag);

            let mut i = 0;
            while i + 4 <= n {
                let xi = vld2q_f32(p);
                let xi_re = xi.0; // [r0, r1, r2, r3]
                let xi_im = xi.1; // [i0, i1, i2, i3] 

                // re = ar xr - ai xi
                // im = ar xi + ai xr
                let re = vfmsq_f32(vmulq_f32(xi_re, ar), xi_im, ai);
                let im = vfmaq_f32(vmulq_f32(xi_im, ar), xi_re, ai); 

                vst2q_f32(p, float32x4x2_t(re, im));

                p = p.add(8);   
                i += 4;
            }

            let mut k = i;
            while k < n {
                let re = *p;
                let im = *p.add(1);

                *p        = a_real * re - a_imag * im;
                *p.add(1) = a_real * im + a_imag * re;

                p = p.add(2);
                k += 1;
            }
        }
    } else {
        unsafe {
            let p = x.as_mut_ptr();
            let mut idx = 0;

            for _ in 0..n {
                let re = *p.add(idx);
                let im = *p.add(idx + 1);

                *p.add(idx + 0) = a_real * re - a_imag * im;
                *p.add(idx + 1) = a_real * im + a_imag * re;

                idx += incx * 2;
            }
        }
    }
}
