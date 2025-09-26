//! Computes the unconjugated dot product DOTU of two complex double precision vectors.
//!
//! ```text
//! sum(x[i] * y[i])
//! ```
//!
//! This function implements the BLAS [`zdotu`] routine over `n` complex elements 
//! of the input vectors `x` and `y` with specified strides.
//!
//! # Arguments
//! - `n`    (usize)  : Number of complex elements in the vectors.
//! - `x`    (&[f64]) : Input slice containing interleaved complex vector elements
//!                   | `[re0, im0, re1, im1, ...]`.
//! - `incx` (usize)  : Stride between consecutive complex elements of `x`; complex units. 
//! - `y`    (&[f64]) : Input slice containing interleaved complex vector elements.
//! - `incy` (usize)  : Stride between consecutive complex elements of `y`; complex units. 
//!
//! # Returns
//! - `[f64; 2]` complex result of the dot product, `[real, imag]`.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`zdotu`] uses unrolled NEON SIMD instructions
//!   for optimized performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0`, the function returns `[0.0, 0.0]`.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vld1q_f64,
    vdupq_n_f64, 
    vfmaq_f64, 
    vfmsq_f64, 
    vaddvq_f64,
    vaddq_f64, 
    vuzp1q_f64, 
    vuzp2q_f64,
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline]
#[cfg(target_arch = "aarch64")]
pub fn zdotu(
    n       : usize,
    x       : &[f64], 
    incx    : usize, 
    y       : &[f64], 
    incy    : usize
) -> [f64; 2] {
    // quick return 
    if n == 0 { return [0.0, 0.0]; }

    debug_assert!(incx > 0 && incy > 0, "increments must be nonzero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_ptr();
    let py = y.as_ptr();

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe {
            let mut acc_re0 = vdupq_n_f64(0.0);
            let mut acc_im0 = vdupq_n_f64(0.0);
            let mut acc_re1 = vdupq_n_f64(0.0);
            let mut acc_im1 = vdupq_n_f64(0.0);

            let mut i = 0usize;
            let len = 2 * n; 

            while i + 8 <= len {
                let x0 = vld1q_f64(px.add(i + 0));
                let x1 = vld1q_f64(px.add(i + 2));
                let x2 = vld1q_f64(px.add(i + 4));
                let x3 = vld1q_f64(px.add(i + 6));
                let y0 = vld1q_f64(py.add(i + 0));
                let y1 = vld1q_f64(py.add(i + 2));
                let y2 = vld1q_f64(py.add(i + 4));
                let y3 = vld1q_f64(py.add(i + 6));

                let x_re_a0 = vuzp1q_f64(x0, x1); 
                let x_im_a0 = vuzp2q_f64(x0, x1); 
                let y_re_a0 = vuzp1q_f64(y0, y1);
                let y_im_a0 = vuzp2q_f64(y0, y1);

                let x_re_a1 = vuzp1q_f64(x2, x3);
                let x_im_a1 = vuzp2q_f64(x2, x3);
                let y_re_a1 = vuzp1q_f64(y2, y3);
                let y_im_a1 = vuzp2q_f64(y2, y3);

                // unconjugated 
                // real += (x_re y_re - x_im y_im) 
                // imag += (x_re y_im + x_im y_re) 
                acc_re0 = vfmaq_f64(acc_re0, x_re_a0, y_re_a0);
                acc_re0 = vfmsq_f64(acc_re0, x_im_a0, y_im_a0); 
                acc_im0 = vfmaq_f64(acc_im0, x_re_a0, y_im_a0); 
                acc_im0 = vfmaq_f64(acc_im0, x_im_a0, y_re_a0); 

                acc_re1 = vfmaq_f64(acc_re1, x_re_a1, y_re_a1);
                acc_re1 = vfmsq_f64(acc_re1, x_im_a1, y_im_a1);
                acc_im1 = vfmaq_f64(acc_im1, x_re_a1, y_im_a1);
                acc_im1 = vfmaq_f64(acc_im1, x_im_a1, y_re_a1);

                i += 8;
            }

            while i + 4 <= len {
                let x0 = vld1q_f64(px.add(i + 0));
                let x1 = vld1q_f64(px.add(i + 2));
                let y0 = vld1q_f64(py.add(i + 0));
                let y1 = vld1q_f64(py.add(i + 2));

                let x_re = vuzp1q_f64(x0, x1);
                let x_im = vuzp2q_f64(x0, x1);
                let y_re = vuzp1q_f64(y0, y1);
                let y_im = vuzp2q_f64(y0, y1);

                acc_re0 = vfmaq_f64(acc_re0, x_re, y_re);
                acc_re0 = vfmsq_f64(acc_re0, x_im, y_im);
                acc_im0 = vfmaq_f64(acc_im0, x_re, y_im);
                acc_im0 = vfmaq_f64(acc_im0, x_im, y_re);

                i += 4;
            }

            let acc_re_v = vaddq_f64(acc_re0, acc_re1);
            let acc_im_v = vaddq_f64(acc_im0, acc_im1);

            let mut real = vaddvq_f64(acc_re_v); // total real 
            let mut imag = vaddvq_f64(acc_im_v); // total imag

            // tail 
            while i < len {
                let xr = *px.add(i + 0);
                let xi = *px.add(i + 1);
                let yr = *py.add(i + 0);
                let yi = *py.add(i + 1);

                real += xr * yr - xi * yi;
                imag += xr * yi + xi * yr;

                i += 2;
            }

            return [real, imag];
        }
    }

    // non unit stride 
    unsafe {
        let mut real = 0.0;
        let mut imag = 0.0;

        let mut ix = 0; 
        let mut iy = 0;


        for _ in 0..n {
            let xr = *px.add(ix + 0);
            let xi = *px.add(ix + 1);
            let yr = *py.add(iy + 0);
            let yi = *py.add(iy + 1);

            real += xr * yr - xi * yi;
            imag += xr * yi + xi * yr;

            ix += incx * 2;
            iy += incy * 2;
        }

        [real, imag]
    }
}

