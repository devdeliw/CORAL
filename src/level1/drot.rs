//! Applies a plane rotation to two double precision vectors.
//!
//! This function implements the BLAS [`drot`] routine, replacing elements of
//! vectors `x` and `y` with
//!
//! x[i] := c * x[i] + s * y[i]
//! y[i] := c * y[i] - s * x[i]
//!
//! over `n` entries with specified strides.
//!
//! # Arguments
//! - `n`    (usize)      : Number of elements to process.
//! - `x`    (&mut [f64]) : Input/output slice containing the first vector, updated in place.
//! - `incx` (usize)      : Stride between consecutive elements of `x`.
//! - `y`    (&mut [f64]) : Input/output slice containing the second vector, updated in place.
//! - `incy` (usize)      : Stride between consecutive elements of `y`.
//! - `c`    (f64)        : Cosine component of the rotation.
//! - `s`    (f64)        : Sine component of the rotation.
//!
//! # Returns
//! - Nothing. The contents of `x` and `y` are updated in place.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`drot`] uses unrolled NEON SIMD instructions
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
    vfmsq_f64,
    vmulq_f64,
};
use crate::level1::assert_length_helpers::required_len_ok;


#[inline]
#[cfg(target_arch = "aarch64")]
pub fn drot(
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
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_mut_ptr();
    let py = y.as_mut_ptr();

    // fast path
    if incx == 1 && incy == 1 {
        unsafe {
            let c2 = vdupq_n_f64(c);
            let s2 = vdupq_n_f64(s);

            let mut i = 0;
            while i + 16 <= n {
                let x0 = vld1q_f64(px.add(i + 0));
                let x1 = vld1q_f64(px.add(i + 2));
                let x2 = vld1q_f64(px.add(i + 4));
                let x3 = vld1q_f64(px.add(i + 6));
                let x4 = vld1q_f64(px.add(i + 8));
                let x5 = vld1q_f64(px.add(i + 10));
                let x6 = vld1q_f64(px.add(i + 12));
                let x7 = vld1q_f64(px.add(i + 14));

                let y0 = vld1q_f64(py.add(i + 0));
                let y1 = vld1q_f64(py.add(i + 2));
                let y2 = vld1q_f64(py.add(i + 4));
                let y3 = vld1q_f64(py.add(i + 6));
                let y4 = vld1q_f64(py.add(i + 8));
                let y5 = vld1q_f64(py.add(i + 10));
                let y6 = vld1q_f64(py.add(i + 12));
                let y7 = vld1q_f64(py.add(i + 14));

                let xn0 = vfmaq_f64(vmulq_f64(c2, x0), y0, s2);
                let xn1 = vfmaq_f64(vmulq_f64(c2, x1), y1, s2);
                let xn2 = vfmaq_f64(vmulq_f64(c2, x2), y2, s2);
                let xn3 = vfmaq_f64(vmulq_f64(c2, x3), y3, s2);
                let xn4 = vfmaq_f64(vmulq_f64(c2, x4), y4, s2);
                let xn5 = vfmaq_f64(vmulq_f64(c2, x5), y5, s2);
                let xn6 = vfmaq_f64(vmulq_f64(c2, x6), y6, s2);
                let xn7 = vfmaq_f64(vmulq_f64(c2, x7), y7, s2);

                let yn0 = vfmsq_f64(vmulq_f64(c2, y0), s2, x0);
                let yn1 = vfmsq_f64(vmulq_f64(c2, y1), s2, x1);
                let yn2 = vfmsq_f64(vmulq_f64(c2, y2), s2, x2);
                let yn3 = vfmsq_f64(vmulq_f64(c2, y3), s2, x3);
                let yn4 = vfmsq_f64(vmulq_f64(c2, y4), s2, x4);
                let yn5 = vfmsq_f64(vmulq_f64(c2, y5), s2, x5);
                let yn6 = vfmsq_f64(vmulq_f64(c2, y6), s2, x6);
                let yn7 = vfmsq_f64(vmulq_f64(c2, y7), s2, x7);

                vst1q_f64(py.add(i + 0), yn0);
                vst1q_f64(py.add(i + 2), yn1);
                vst1q_f64(py.add(i + 4), yn2);
                vst1q_f64(py.add(i + 6), yn3);
                vst1q_f64(py.add(i + 8), yn4);
                vst1q_f64(py.add(i + 10), yn5);
                vst1q_f64(py.add(i + 12), yn6);
                vst1q_f64(py.add(i + 14), yn7);

                vst1q_f64(px.add(i + 0), xn0);
                vst1q_f64(px.add(i + 2), xn1);
                vst1q_f64(px.add(i + 4), xn2);
                vst1q_f64(px.add(i + 6), xn3);
                vst1q_f64(px.add(i + 8), xn4);
                vst1q_f64(px.add(i + 10), xn5);
                vst1q_f64(px.add(i + 12), xn6);
                vst1q_f64(px.add(i + 14), xn7);

                i += 16;
            }

            while i + 2 <= n {
                let xv = vld1q_f64(px.add(i));
                let yv = vld1q_f64(py.add(i));

                let xn = vfmaq_f64(vmulq_f64(c2, xv), yv, s2);
                let yn = vfmsq_f64(vmulq_f64(c2, yv), s2, xv); 

                vst1q_f64(py.add(i), yn);
                vst1q_f64(px.add(i), xn);

                i += 2;
            }

            // tail
            while i < n {
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

    // non-unit stride
    unsafe {
        let mut ix = 0; 
        let mut iy = 0; 

        for _ in 0..n {
            let xi  = *px.add(ix);
            let yi  = *py.add(iy);
            let tmp = c * xi + s * yi;

            *py.add(iy) = c * yi - s * xi;
            *px.add(ix) = tmp;

            ix += incx;
            iy += incy;
        }
    }
}

