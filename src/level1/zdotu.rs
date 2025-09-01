//! Computes the unconjugated dot product of two complex double precision vectors.
//!
//! This function implements the BLAS [`zdotu`] routine, returning
//! sum(x[i] * y[i]) over `n` complex elements of the input vectors `x` and `y`
//! with specified strides. No conjugation is applied to either vector.
//!
//! # Arguments
//! - `n`    : Number of complex elements in the vectors.
//! - `x`    : Input slice containing interleaved complex vector elements
//!            `[re0, im0, re1, im1, ...]`.
//! - `incx` : Stride between consecutive complex elements of `x`
//!            (measured in complex numbers; every step advances two scalar idxs).
//! - `y`    : Input slice containing interleaved complex vector elements
//!            `[re0, im0, re1, im1, ...]`.
//! - `incy` : Stride between consecutive complex elements of `y`
//!            (measured in complex numbers; every step advances two scalar idxs).
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


use core::arch::aarch64::{
    vld1q_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vaddvq_f64, vaddq_f64, vuzp1q_f64, vuzp2q_f64,
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline]
pub fn zdotu(n: usize, x: &[f64], incx: isize, y: &[f64], incy: isize) -> [f64; 2] {
    if n == 0 { return [0.0, 0.0]; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
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
            let mut real = vaddvq_f64(acc_re_v);
            let mut imag = vaddvq_f64(acc_im_v);

            // tail 
            while i < len {
                let xr = *px.add(i);
                let xi = *px.add(i + 1);
                let yr = *py.add(i);
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
        let mut real = 0.0f64;
        let mut imag = 0.0f64;

        let stepx: isize = 2 * incx; 
        let stepy: isize = 2 * incy;

        let mut ix: isize = if incx >= 0 { 0 } else { 2 * (n as isize - 1) * -incx };
        let mut iy: isize = if incy >= 0 { 0 } else { 2 * (n as isize - 1) * -incy };

        for _ in 0..n {
            let xr = *px.offset(ix);
            let xi = *px.offset(ix + 1);
            let yr = *py.offset(iy);
            let yi = *py.offset(iy + 1);

            real += xr * yr - xi * yi;
            imag += xr * yi + xi * yr;

            ix += stepx;
            iy += stepy;
        }

        [real, imag]
    }
}

