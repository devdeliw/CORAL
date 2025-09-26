//! Applies a plane rotation ROT to two complex double precision vectors.
//!
//! This function implements the BLAS [`zdrot`] routine, replacing elements of
//! vectors `x` and `y` with
//! 
//! ```text
//! x[i] := c * x[i] + s * y[i]
//! y[i] := c * y[i] - s * x[i]
//! ```
//!
//! where the rotation is applied elementwise to both the real and imaginary parts
//! of each complex number, over `n` complex entries with specified strides.
//!
//! # Arguments
//! - `n`    (usize)      : Number of complex elements to process.
//! - `x`    (&mut [f64]) : Input/output slice containing interleaved complex vector elements
//!                       | `[re0, im0, re1, im1, ...]`, updated in place.
//! - `incx` (usize)      : Stride between consecutive complex elements of `x`; complex units. 
//! - `y`    (&mut [f64]) : Input/output slice containing interleaved complex vector elements,
//!                       | updated in place.
//! - `incy` (usize)      : Stride between consecutive complex elements of `y`; complex units. 
//! - `c`    (f64)        : Cosine component of the rotation.
//! - `s`    (f64)        : Sine component of the rotation.
//!
//! # Returns
//! - Nothing. The contents of `x` and `y` are updated in place.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`zdrot`] uses unrolled NEON SIMD instructions
//!   for optimized performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0`, the function returns immediately; no slice modification.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f64, 
    vst1q_f64, 
    vdupq_n_f64,
    vfmaq_f64, 
    vmulq_f64, 
    vsubq_f64,
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline]
#[cfg(target_arch = "aarch64")] 
pub fn zdrot(
    n       : usize, 
    x       : &mut [f64], 
    incx    : usize, 
    y       : &mut [f64], 
    incy    : usize, 
    c       : f64,
    s       : f64
) {
    // quick return
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy (complex)");

    let px = x.as_mut_ptr();
    let py = y.as_mut_ptr();

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe {
            let c2 = vdupq_n_f64(c);
            let s2 = vdupq_n_f64(s);

            let len   = 2 * n;
            let mut i = 0;
            while i + 8 <= len {
                let x0 = vld1q_f64(px.add(i + 0));
                let x1 = vld1q_f64(px.add(i + 2));
                let x2 = vld1q_f64(px.add(i + 4));
                let x3 = vld1q_f64(px.add(i + 6));

                let y0 = vld1q_f64(py.add(i + 0));
                let y1 = vld1q_f64(py.add(i + 2));
                let y2 = vld1q_f64(py.add(i + 4));
                let y3 = vld1q_f64(py.add(i + 6));

                let xn0 = vfmaq_f64(vmulq_f64(c2, x0), y0, s2);
                let xn1 = vfmaq_f64(vmulq_f64(c2, x1), y1, s2);
                let xn2 = vfmaq_f64(vmulq_f64(c2, x2), y2, s2);
                let xn3 = vfmaq_f64(vmulq_f64(c2, x3), y3, s2);

                let yn0 = vsubq_f64(vmulq_f64(c2, y0), vmulq_f64(s2, x0));
                let yn1 = vsubq_f64(vmulq_f64(c2, y1), vmulq_f64(s2, x1));
                let yn2 = vsubq_f64(vmulq_f64(c2, y2), vmulq_f64(s2, x2));
                let yn3 = vsubq_f64(vmulq_f64(c2, y3), vmulq_f64(s2, x3));

                vst1q_f64(py.add(i + 0), yn0);
                vst1q_f64(py.add(i + 2), yn1);
                vst1q_f64(py.add(i + 4), yn2);
                vst1q_f64(py.add(i + 6), yn3);

                vst1q_f64(px.add(i + 0), xn0);
                vst1q_f64(px.add(i + 2), xn1);
                vst1q_f64(px.add(i + 4), xn2);
                vst1q_f64(px.add(i + 6), xn3);

                i += 8;
            }

            while i + 2 <= len {
                let xv = vld1q_f64(px.add(i));
                let yv = vld1q_f64(py.add(i));

                let xn = vfmaq_f64(vmulq_f64(c2, xv), yv, s2);
                let yn = vsubq_f64(vmulq_f64(c2, yv), vmulq_f64(s2, xv));

                vst1q_f64(py.add(i), yn);
                vst1q_f64(px.add(i), xn);

                i += 2;
            }

            while i < len {
                let xi  = *px.add(i);
                let yi  = *py.add(i);
                let tmp = c * xi + s * yi;  

                *py.add(i) = c * yi - s * xi;   
                *px.add(i) = tmp;

                i += 1;
            }
        }
        return;
    }

    // non unit stride 
    unsafe {
        let mut ix = 0; 
        let mut iy = 0; 

        for _ in 0..n {
            let xr = *px.add(2*ix + 0);
            let xi = *px.add(2*ix + 1);
            let yr = *py.add(2*iy + 0);
            let yi = *py.add(2*iy + 1);

            let xnr = c * xr + s * yr;
            let xni = c * xi + s * yi;

            let ynr = c * yr - s * xr;
            let yni = c * yi - s * xi;

            *px.add(2*ix + 0) = xnr;
            *px.add(2*ix + 1) = xni;
            *py.add(2*iy + 0) = ynr;
            *py.add(2*iy + 1) = yni;

            ix += incx;
            iy += incy;
        }
    }
}

