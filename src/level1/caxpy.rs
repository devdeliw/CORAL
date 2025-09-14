//! Performs a complex single precision AXPY operation: y := alpha * x + y.
//!
//! This function implements the BLAS [`caxpy`] routine, updating the vector `y`
//! by adding `alpha * x` elementwise over `n` complex entries with specified strides.
//!
//! # Arguments
//! - `n`     (usize)      : Number of complex elements to process.
//! - `alpha` ([f32; 2])   : Complex scalar multiplier given as `[real, imag]`.
//! - `x`     (&[f32])     : Input slice containing interleaved complex vector elements
//!                        | `[re0, im0, re1, im1, ...]`.
//! - `incx`  (usize)      : Stride between consecutive complex elements of `x`
//!                        | (measured in complex numbers; every step advances two scalar idxs).
//! - `y`     (&mut [f32]) : Input/output slice containing interleaved complex vector elements,
//!                        | updated in place.
//! - `incy`  (usize)      : Stride between consecutive complex elements of `y`
//!                          (measured in complex numbers; every step advances two scalar idxs).
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place as `y[i] = alpha * x[i] + y[i]`.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`caxpy`] uses unrolled NEON SIMD instructions
//!   for optimized performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `alpha == [0.0, 0.0]`, the function returns immediately; no slice modification.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{ 
    vdupq_n_f32,
    vfmaq_f32, 
    vfmsq_f32,
    vld2q_f32, 
    vst2q_f32, 
    float32x4x2_t
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn caxpy(
    n       : usize, 
    alpha   : [f32; 2], 
    x       : &[f32], 
    incx    : usize, 
    y       : &mut [f32], 
    incy    : usize
) { 
    let ar = alpha[0]; // real part 
    let ai = alpha[1]; // imag part 

    // quick return 
    if n == 0 || (ar == 0.0  && ai == 0.0) { 
        return; 
    }

    debug_assert!(incx > 0 && incy > 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy");

    unsafe { 
        let ar_v = vdupq_n_f32(ar); 
        let ai_v = vdupq_n_f32(ai);

        // fast path 
        if incx == 1 && incy == 1 { 
            let mut i = 0usize; 
            while i + 8 <= n { 
                // first four 
                let p0 = 2 * i; 

                let x01 = vld2q_f32(x.as_ptr().add(p0)); 
                let y01 = vld2q_f32(y.as_ptr().add(p0)); 

                let xr0     = x01.0; 
                let xi0     = x01.1; 
                let mut yr0 = y01.0; 
                let mut yi0 = y01.1; 

                // y += (ar_v xr0 - ai_v xri) + 
                //      (ar_v xri + ai_v xr0) 
                yr0 = vfmaq_f32(yr0, ar_v, xr0); 
                yr0 = vfmsq_f32(yr0, ai_v, xi0); 
                yi0 = vfmaq_f32(yi0, ar_v, xi0); 
                yi0 = vfmaq_f32(yi0, ai_v, xr0);

                // store
                vst2q_f32(y.as_mut_ptr().add(p0), float32x4x2_t(yr0, yi0));
                
                // second four 
                let p1  = p0 + 8; 
                let x23 = vld2q_f32(x.as_ptr().add(p1)); 
                let y23 = vld2q_f32(y.as_ptr().add(p1)); 
                
                let xr1     = x23.0; 
                let xi1     = x23.1; 
                let mut yr1 = y23.0; 
                let mut yi1 = y23.1;

                yr1 = vfmaq_f32(yr1, ar_v, xr1); 
                yr1 = vfmsq_f32(yr1, ai_v, xi1); 
                yi1 = vfmaq_f32(yi1, ar_v, xi1); 
                yi1 = vfmaq_f32(yi1, ai_v, xr1); 

                // store 
                vst2q_f32(y.as_mut_ptr().add(p1), float32x4x2_t(yr1, yi1));

                i += 8; 
            }
            while i + 4 <= n {
                let p = 2 * i;

                let x01 = vld2q_f32(x.as_ptr().add(p));
                let y01 = vld2q_f32(y.as_ptr().add(p));

                let xr     = x01.0; 
                let xi     = x01.1;
                let mut yr = y01.0; 
                let mut yi = y01.1;

                yr = vfmaq_f32(yr, ar_v, xr);
                yr = vfmsq_f32(yr, ai_v, xi);
                yi = vfmaq_f32(yi, ar_v, xi);
                yi = vfmaq_f32(yi, ai_v, xr);

                vst2q_f32(y.as_mut_ptr().add(p), float32x4x2_t(yr, yi));

                i += 4;
            }

            // tail
            while i < n { 
                let p = 2 * i; 
                let xr = *x.as_ptr().add(p); 
                let xi = *x.as_ptr().add(p + 1); 

                let yrp = y.as_mut_ptr().add(p); 
                let yip = y.as_mut_ptr().add(p + 1); 

                *yrp += ar * xr - ai * xi; 
                *yip += ar * xi + ai * xr; 

                i += 1; 
            }
        } else { 
            // non unit stride 
            let px = x.as_ptr(); 
            let py = y.as_mut_ptr(); 

            let mut ix = 0;
            let mut iy = 0;

            for _ in 0..n { 
                let xr = *px.add(ix); 
                let xi = *px.add(ix + 1); 

                let yrp = py.add(iy); 
                let yip = py.add(iy + 1); 

                *yrp += ar * xr - ai * xi; 
                *yip += ar * xi + ai * xr; 

                ix += incx * 2; 
                iy += incy * 2;
            }
        }
    }
}
