//! Performs a complex matrix-vector multiply and accumulation AXPY:
//!
//! ```text
//! y := y + A x
//! ```
//!
//! This function implements an optimized fused double-precision complex
//! BLAS [`zaxpy`] operation using NEON intrinsics on AArch64. It computes the
//! product of a complex matrix `A` (size m x n) and a complex vector `x`
//! (length n), and accumulates the result into the complex vector `y`
//! (length m).
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `x`      (&[f64])     : Input interleaved complex vector.
//!                         | `[re, im, re, im, ...]`.
//! - `incx`   (usize)      : Stride between consecutive complex elements of `x`.
//! - `matrix` (&[f64])     : Complex interleaved matrix `A` of dimensions
//!                         | (`lda` x `n_cols`). 
//! - `lda`    (usize)      : Leading dimension of `A`. Must be >= `n_rows`.
//! - `y`      (&mut [f64]) : Input/output interleaved complex vector.
//! - `incy`   (usize)      : Stride between consecutive complex elements of `y`.
//!
//! # Notes
//! - For unit strides (`incx == 1`, `incy == 1`), the kernel uses
//!   SIMD microkernels with blocking. 
//! - For non-unit strides, it falls back to scalar [`zaxpy`] updates.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float64x2x2_t,
    vdupq_n_f64,
    vld2q_f64,
    vfmaq_f64,
    vfmsq_f64,
    vst2q_f64,
};

use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;
use crate::level1::zaxpy::zaxpy;

const MC: usize = 128;
const NR: usize = 8;

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn zaxpyf(
    n_rows : usize,
    n_cols : usize,
    x      : &[f64],
    incx   : usize,
    matrix : &[f64],
    lda    : usize,
    y      : &mut [f64],
    incy   : usize,
) {
    // quick return
    if n_rows == 0 || n_cols == 0 { return; }

    debug_assert!(incx > 0 && incy > 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n_cols, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n_rows, incy), "y too short for m/incy");
    debug_assert!(lda >= n_rows, "lda must be larger than n_rows (in complexes)");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n_rows, n_cols, lda),
        "complex matrix not large enough given n_rows, n_cols, and lda col stride"
    );

    // fast path
    if incx == 1 && incy == 1 {
        unsafe {
            let row_block = MC;
            let mut row_idx = 0;

            while row_idx < n_rows {
                let mr = core::cmp::min(row_block, n_rows - row_idx);

                let mut col_idx = 0;

                // 8 columns at a time
                while col_idx + NR <= n_cols {
                    // x broadcasts (re, im) for 8 columns
                    let xr0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 0));
                    let xi0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 1));
                    let xr1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 0));
                    let xi1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 1));
                    let xr2 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 2) + 0));
                    let xi2 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 2) + 1));
                    let xr3 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 3) + 0));
                    let xi3 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 3) + 1));
                    let xr4 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 4) + 0));
                    let xi4 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 4) + 1));
                    let xr5 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 5) + 0));
                    let xi5 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 5) + 1));
                    let xr6 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 6) + 0));
                    let xi6 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 6) + 1));
                    let xr7 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 7) + 0));
                    let xi7 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 7) + 1));

                    // pointers to 8 columns in matrix (convert complexes to scalars by ×2)
                    let pa0 = matrix.as_ptr().add(2 * ((col_idx + 0) * lda + row_idx));
                    let pa1 = matrix.as_ptr().add(2 * ((col_idx + 1) * lda + row_idx));
                    let pa2 = matrix.as_ptr().add(2 * ((col_idx + 2) * lda + row_idx));
                    let pa3 = matrix.as_ptr().add(2 * ((col_idx + 3) * lda + row_idx));
                    let pa4 = matrix.as_ptr().add(2 * ((col_idx + 4) * lda + row_idx));
                    let pa5 = matrix.as_ptr().add(2 * ((col_idx + 5) * lda + row_idx));
                    let pa6 = matrix.as_ptr().add(2 * ((col_idx + 6) * lda + row_idx));
                    let pa7 = matrix.as_ptr().add(2 * ((col_idx + 7) * lda + row_idx));

                    // process 4 complexes per chunk (two NEON loads of 2 complexes each)
                    let mut i = 0;
                    while i + 4 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));

                        // first 2 complexes
                        let mut y0: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr0, a0.0);
                        y0.0 = vfmsq_f64(y0.0, xi0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xr0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr1, a1.0);
                        y0.0 = vfmsq_f64(y0.0, xi1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xr1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xi1, a1.0);

                        let a2 = vld2q_f64(pa2.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr2, a2.0);
                        y0.0 = vfmsq_f64(y0.0, xi2, a2.1);
                        y0.1 = vfmaq_f64(y0.1, xr2, a2.1);
                        y0.1 = vfmaq_f64(y0.1, xi2, a2.0);

                        let a3 = vld2q_f64(pa3.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr3, a3.0);
                        y0.0 = vfmsq_f64(y0.0, xi3, a3.1);
                        y0.1 = vfmaq_f64(y0.1, xr3, a3.1);
                        y0.1 = vfmaq_f64(y0.1, xi3, a3.0);

                        let a4 = vld2q_f64(pa4.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr4, a4.0);
                        y0.0 = vfmsq_f64(y0.0, xi4, a4.1);
                        y0.1 = vfmaq_f64(y0.1, xr4, a4.1);
                        y0.1 = vfmaq_f64(y0.1, xi4, a4.0);

                        let a5 = vld2q_f64(pa5.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr5, a5.0);
                        y0.0 = vfmsq_f64(y0.0, xi5, a5.1);
                        y0.1 = vfmaq_f64(y0.1, xr5, a5.1);
                        y0.1 = vfmaq_f64(y0.1, xi5, a5.0);

                        let a6 = vld2q_f64(pa6.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr6, a6.0);
                        y0.0 = vfmsq_f64(y0.0, xi6, a6.1);
                        y0.1 = vfmaq_f64(y0.1, xr6, a6.1);
                        y0.1 = vfmaq_f64(y0.1, xi6, a6.0);

                        let a7 = vld2q_f64(pa7.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr7, a7.0);
                        y0.0 = vfmsq_f64(y0.0, xi7, a7.1);
                        y0.1 = vfmaq_f64(y0.1, xr7, a7.1);
                        y0.1 = vfmaq_f64(y0.1, xi7, a7.0);

                        vst2q_f64(pyi, y0);

                        // next 2 complexes
                        let pyj = pyi.add(4);
                        let mut y1: float64x2x2_t = vld2q_f64(pyj);

                        let b0 = vld2q_f64(pa0.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr0, b0.0);
                        y1.0 = vfmsq_f64(y1.0, xi0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xr0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xi0, b0.0);

                        let b1 = vld2q_f64(pa1.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr1, b1.0);
                        y1.0 = vfmsq_f64(y1.0, xi1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xr1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xi1, b1.0);

                        let b2 = vld2q_f64(pa2.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr2, b2.0);
                        y1.0 = vfmsq_f64(y1.0, xi2, b2.1);
                        y1.1 = vfmaq_f64(y1.1, xr2, b2.1);
                        y1.1 = vfmaq_f64(y1.1, xi2, b2.0);

                        let b3 = vld2q_f64(pa3.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr3, b3.0);
                        y1.0 = vfmsq_f64(y1.0, xi3, b3.1);
                        y1.1 = vfmaq_f64(y1.1, xr3, b3.1);
                        y1.1 = vfmaq_f64(y1.1, xi3, b3.0);

                        let b4 = vld2q_f64(pa4.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr4, b4.0);
                        y1.0 = vfmsq_f64(y1.0, xi4, b4.1);
                        y1.1 = vfmaq_f64(y1.1, xr4, b4.1);
                        y1.1 = vfmaq_f64(y1.1, xi4, b4.0);

                        let b5 = vld2q_f64(pa5.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr5, b5.0);
                        y1.0 = vfmsq_f64(y1.0, xi5, b5.1);
                        y1.1 = vfmaq_f64(y1.1, xr5, b5.1);
                        y1.1 = vfmaq_f64(y1.1, xi5, b5.0);

                        let b6 = vld2q_f64(pa6.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr6, b6.0);
                        y1.0 = vfmsq_f64(y1.0, xi6, b6.1);
                        y1.1 = vfmaq_f64(y1.1, xr6, b6.1);
                        y1.1 = vfmaq_f64(y1.1, xi6, b6.0);

                        let b7 = vld2q_f64(pa7.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr7, b7.0);
                        y1.0 = vfmsq_f64(y1.0, xi7, b7.1);
                        y1.1 = vfmaq_f64(y1.1, xr7, b7.1);
                        y1.1 = vfmaq_f64(y1.1, xi7, b7.0);

                        vst2q_f64(pyj, y1);

                        i += 4;
                    }

                    // 2 complexes at a time
                    while i + 2 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut yv: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                        yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                        yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                        let a2 = vld2q_f64(pa2.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr2, a2.0);
                        yv.0 = vfmsq_f64(yv.0, xi2, a2.1);
                        yv.1 = vfmaq_f64(yv.1, xr2, a2.1);
                        yv.1 = vfmaq_f64(yv.1, xi2, a2.0);

                        let a3 = vld2q_f64(pa3.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr3, a3.0);
                        yv.0 = vfmsq_f64(yv.0, xi3, a3.1);
                        yv.1 = vfmaq_f64(yv.1, xr3, a3.1);
                        yv.1 = vfmaq_f64(yv.1, xi3, a3.0);

                        let a4 = vld2q_f64(pa4.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr4, a4.0);
                        yv.0 = vfmsq_f64(yv.0, xi4, a4.1);
                        yv.1 = vfmaq_f64(yv.1, xr4, a4.1);
                        yv.1 = vfmaq_f64(yv.1, xi4, a4.0);

                        let a5 = vld2q_f64(pa5.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr5, a5.0);
                        yv.0 = vfmsq_f64(yv.0, xi5, a5.1);
                        yv.1 = vfmaq_f64(yv.1, xr5, a5.1);
                        yv.1 = vfmaq_f64(yv.1, xi5, a5.0);

                        let a6 = vld2q_f64(pa6.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr6, a6.0);
                        yv.0 = vfmsq_f64(yv.0, xi6, a6.1);
                        yv.1 = vfmaq_f64(yv.1, xr6, a6.1);
                        yv.1 = vfmaq_f64(yv.1, xi6, a6.0);

                        let a7 = vld2q_f64(pa7.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr7, a7.0);
                        yv.0 = vfmsq_f64(yv.0, xi7, a7.1);
                        yv.1 = vfmaq_f64(yv.1, xr7, a7.1);
                        yv.1 = vfmaq_f64(yv.1, xi7, a7.0);

                        vst2q_f64(pyi, yv);

                        i += 2;
                    }

                    // tail 
                    while i < mr {
                        let off = 2 * (row_idx + i);
                        let mut yr = *y.get_unchecked(off + 0);
                        let mut yi = *y.get_unchecked(off + 1);

                        let xr0s = *x.get_unchecked(2*(col_idx + 0) + 0);
                        let xi0s = *x.get_unchecked(2*(col_idx + 0) + 1);
                        let ar0  = *pa0.add(2*i + 0);
                        let ai0  = *pa0.add(2*i + 1);
                        yr += xr0s * ar0 - xi0s * ai0;
                        yi += xr0s * ai0 + xi0s * ar0;

                        let xr1s = *x.get_unchecked(2*(col_idx + 1) + 0);
                        let xi1s = *x.get_unchecked(2*(col_idx + 1) + 1);
                        let ar1  = *pa1.add(2*i + 0);
                        let ai1  = *pa1.add(2*i + 1);
                        yr += xr1s * ar1 - xi1s * ai1;
                        yi += xr1s * ai1 + xi1s * ar1;

                        let xr2s = *x.get_unchecked(2*(col_idx + 2) + 0);
                        let xi2s = *x.get_unchecked(2*(col_idx + 2) + 1);
                        let ar2  = *pa2.add(2*i + 0);
                        let ai2  = *pa2.add(2*i + 1);
                        yr += xr2s * ar2 - xi2s * ai2;
                        yi += xr2s * ai2 + xi2s * ar2;

                        let xr3s = *x.get_unchecked(2*(col_idx + 3) + 0);
                        let xi3s = *x.get_unchecked(2*(col_idx + 3) + 1);
                        let ar3  = *pa3.add(2*i + 0);
                        let ai3  = *pa3.add(2*i + 1);
                        yr += xr3s * ar3 - xi3s * ai3;
                        yi += xr3s * ai3 + xi3s * ar3;

                        let xr4s = *x.get_unchecked(2*(col_idx + 4) + 0);
                        let xi4s = *x.get_unchecked(2*(col_idx + 4) + 1);
                        let ar4  = *pa4.add(2*i + 0);
                        let ai4  = *pa4.add(2*i + 1);
                        yr += xr4s * ar4 - xi4s * ai4;
                        yi += xr4s * ai4 + xi4s * ar4;

                        let xr5s = *x.get_unchecked(2*(col_idx + 5) + 0);
                        let xi5s = *x.get_unchecked(2*(col_idx + 5) + 1);
                        let ar5  = *pa5.add(2*i + 0);
                        let ai5  = *pa5.add(2*i + 1);
                        yr += xr5s * ar5 - xi5s * ai5;
                        yi += xr5s * ai5 + xi5s * ar5;

                        let xr6s = *x.get_unchecked(2*(col_idx + 6) + 0);
                        let xi6s = *x.get_unchecked(2*(col_idx + 6) + 1);
                        let ar6  = *pa6.add(2*i + 0);
                        let ai6  = *pa6.add(2*i + 1);
                        yr += xr6s * ar6 - xi6s * ai6;
                        yi += xr6s * ai6 + xi6s * ar6;

                        let xr7s = *x.get_unchecked(2*(col_idx + 7) + 0);
                        let xi7s = *x.get_unchecked(2*(col_idx + 7) + 1);
                        let ar7  = *pa7.add(2*i + 0);
                        let ai7  = *pa7.add(2*i + 1);
                        yr += xr7s * ar7 - xi7s * ai7;
                        yi += xr7s * ai7 + xi7s * ar7;

                        *y.get_unchecked_mut(off + 0) = yr;
                        *y.get_unchecked_mut(off + 1) = yi;

                        i += 1;
                    }

                    col_idx += NR;
                }

                if col_idx + 4 <= n_cols {
                    let xr0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 0));
                    let xi0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 1));
                    let xr1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 0));
                    let xi1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 1));
                    let xr2 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 2) + 0));
                    let xi2 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 2) + 1));
                    let xr3 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 3) + 0));
                    let xi3 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 3) + 1));

                    let pa0 = matrix.as_ptr().add(2 * ((col_idx + 0) * lda + row_idx));
                    let pa1 = matrix.as_ptr().add(2 * ((col_idx + 1) * lda + row_idx));
                    let pa2 = matrix.as_ptr().add(2 * ((col_idx + 2) * lda + row_idx));
                    let pa3 = matrix.as_ptr().add(2 * ((col_idx + 3) * lda + row_idx));

                    let mut i = 0;
                    while i + 4 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut y0: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr0, a0.0);
                        y0.0 = vfmsq_f64(y0.0, xi0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xr0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr1, a1.0);
                        y0.0 = vfmsq_f64(y0.0, xi1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xr1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xi1, a1.0);

                        let a2 = vld2q_f64(pa2.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr2, a2.0);
                        y0.0 = vfmsq_f64(y0.0, xi2, a2.1);
                        y0.1 = vfmaq_f64(y0.1, xr2, a2.1);
                        y0.1 = vfmaq_f64(y0.1, xi2, a2.0);

                        let a3 = vld2q_f64(pa3.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr3, a3.0);
                        y0.0 = vfmsq_f64(y0.0, xi3, a3.1);
                        y0.1 = vfmaq_f64(y0.1, xr3, a3.1);
                        y0.1 = vfmaq_f64(y0.1, xi3, a3.0);

                        vst2q_f64(pyi, y0);

                        // next 2 complexes
                        let pyj = pyi.add(4);
                        let mut y1: float64x2x2_t = vld2q_f64(pyj);

                        let b0 = vld2q_f64(pa0.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr0, b0.0);
                        y1.0 = vfmsq_f64(y1.0, xi0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xr0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xi0, b0.0);

                        let b1 = vld2q_f64(pa1.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr1, b1.0);
                        y1.0 = vfmsq_f64(y1.0, xi1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xr1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xi1, b1.0);

                        let b2 = vld2q_f64(pa2.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr2, b2.0);
                        y1.0 = vfmsq_f64(y1.0, xi2, b2.1);
                        y1.1 = vfmaq_f64(y1.1, xr2, b2.1);
                        y1.1 = vfmaq_f64(y1.1, xi2, b2.0);

                        let b3 = vld2q_f64(pa3.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr3, b3.0);
                        y1.0 = vfmsq_f64(y1.0, xi3, b3.1);
                        y1.1 = vfmaq_f64(y1.1, xr3, b3.1);
                        y1.1 = vfmaq_f64(y1.1, xi3, b3.0);

                        vst2q_f64(pyj, y1);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut yv: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                        yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                        yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                        let a2 = vld2q_f64(pa2.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr2, a2.0);
                        yv.0 = vfmsq_f64(yv.0, xi2, a2.1);
                        yv.1 = vfmaq_f64(yv.1, xr2, a2.1);
                        yv.1 = vfmaq_f64(yv.1, xi2, a2.0);

                        let a3 = vld2q_f64(pa3.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr3, a3.0);
                        yv.0 = vfmsq_f64(yv.0, xi3, a3.1);
                        yv.1 = vfmaq_f64(yv.1, xr3, a3.1);
                        yv.1 = vfmaq_f64(yv.1, xi3, a3.0);

                        vst2q_f64(pyi, yv);

                        i += 2;
                    }

                    while i < mr {
                        let off = 2 * (row_idx + i);
                        let mut yr = *y.get_unchecked(off + 0);
                        let mut yi = *y.get_unchecked(off + 1);

                        let xr0s = *x.get_unchecked(2*(col_idx + 0) + 0);
                        let xi0s = *x.get_unchecked(2*(col_idx + 0) + 1);
                        let ar0  = *pa0.add(2*i + 0);
                        let ai0  = *pa0.add(2*i + 1);
                        yr += xr0s * ar0 - xi0s * ai0;
                        yi += xr0s * ai0 + xi0s * ar0;

                        let xr1s = *x.get_unchecked(2*(col_idx + 1) + 0);
                        let xi1s = *x.get_unchecked(2*(col_idx + 1) + 1);
                        let ar1  = *pa1.add(2*i + 0);
                        let ai1  = *pa1.add(2*i + 1);
                        yr += xr1s * ar1 - xi1s * ai1;
                        yi += xr1s * ai1 + xi1s * ar1;

                        let xr2s = *x.get_unchecked(2*(col_idx + 2) + 0);
                        let xi2s = *x.get_unchecked(2*(col_idx + 2) + 1);
                        let ar2  = *pa2.add(2*i + 0);
                        let ai2  = *pa2.add(2*i + 1);
                        yr += xr2s * ar2 - xi2s * ai2;
                        yi += xr2s * ai2 + xi2s * ar2;

                        let xr3s = *x.get_unchecked(2*(col_idx + 3) + 0);
                        let xi3s = *x.get_unchecked(2*(col_idx + 3) + 1);
                        let ar3  = *pa3.add(2*i + 0);
                        let ai3  = *pa3.add(2*i + 1);
                        yr += xr3s * ar3 - xi3s * ai3;
                        yi += xr3s * ai3 + xi3s * ar3;

                        *y.get_unchecked_mut(off + 0) = yr;
                        *y.get_unchecked_mut(off + 1) = yi;

                        i += 1;
                    }

                    col_idx += 4;
                }

                if col_idx + 2 <= n_cols {
                    let xr0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 0));
                    let xi0 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 0) + 1));
                    let xr1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 0));
                    let xi1 = vdupq_n_f64(*x.get_unchecked(2*(col_idx + 1) + 1));

                    let pa0 = matrix.as_ptr().add(2 * ((col_idx + 0) * lda + row_idx));
                    let pa1 = matrix.as_ptr().add(2 * ((col_idx + 1) * lda + row_idx));

                    let mut i = 0;
                    while i + 4 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut y0: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr0, a0.0);
                        y0.0 = vfmsq_f64(y0.0, xi0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xr0, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr1, a1.0);
                        y0.0 = vfmsq_f64(y0.0, xi1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xr1, a1.1);
                        y0.1 = vfmaq_f64(y0.1, xi1, a1.0);

                        vst2q_f64(pyi, y0);

                        // next 2 complexes
                        let pyj = pyi.add(4);
                        let mut y1: float64x2x2_t = vld2q_f64(pyj);

                        let b0 = vld2q_f64(pa0.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr0, b0.0);
                        y1.0 = vfmsq_f64(y1.0, xi0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xr0, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xi0, b0.0);

                        let b1 = vld2q_f64(pa1.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr1, b1.0);
                        y1.0 = vfmsq_f64(y1.0, xi1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xr1, b1.1);
                        y1.1 = vfmaq_f64(y1.1, xi1, b1.0);

                        vst2q_f64(pyj, y1);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut yv: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                        yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                        let a1 = vld2q_f64(pa1.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                        yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                        yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                        vst2q_f64(pyi, yv);

                        i += 2;
                    }

                    while i < mr {
                        let off = 2 * (row_idx + i);
                        let mut yr = *y.get_unchecked(off + 0);
                        let mut yi = *y.get_unchecked(off + 1);

                        let xr0s = *x.get_unchecked(2*(col_idx + 0) + 0);
                        let xi0s = *x.get_unchecked(2*(col_idx + 0) + 1);
                        let ar0  = *pa0.add(2*i + 0);
                        let ai0  = *pa0.add(2*i + 1);
                        yr += xr0s * ar0 - xi0s * ai0;
                        yi += xr0s * ai0 + xi0s * ar0;

                        let xr1s = *x.get_unchecked(2*(col_idx + 1) + 0);
                        let xi1s = *x.get_unchecked(2*(col_idx + 1) + 1);
                        let ar1  = *pa1.add(2*i + 0);
                        let ai1  = *pa1.add(2*i + 1);
                        yr += xr1s * ar1 - xi1s * ai1;
                        yi += xr1s * ai1 + xi1s * ar1;

                        *y.get_unchecked_mut(off + 0) = yr;
                        *y.get_unchecked_mut(off + 1) = yi;

                        i += 1;
                    }

                    col_idx += 2;
                }

                if col_idx < n_cols {
                    let xr = vdupq_n_f64(*x.get_unchecked(2*col_idx + 0));
                    let xi = vdupq_n_f64(*x.get_unchecked(2*col_idx + 1));

                    let pa0 = matrix.as_ptr().add(2 * (col_idx * lda + row_idx));

                    let mut i = 0;
                    while i + 4 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut y0: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        y0.0 = vfmaq_f64(y0.0, xr, a0.0);
                        y0.0 = vfmsq_f64(y0.0, xi, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xr, a0.1);
                        y0.1 = vfmaq_f64(y0.1, xi, a0.0);

                        vst2q_f64(pyi, y0);

                        let pyj = pyi.add(4);
                        let mut y1: float64x2x2_t = vld2q_f64(pyj);

                        let b0 = vld2q_f64(pa0.add(2*i + 4));
                        y1.0 = vfmaq_f64(y1.0, xr, b0.0);
                        y1.0 = vfmsq_f64(y1.0, xi, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xr, b0.1);
                        y1.1 = vfmaq_f64(y1.1, xi, b0.0);

                        vst2q_f64(pyj, y1);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let pyi = y.as_mut_ptr().add(2 * (row_idx + i));
                        let mut yv: float64x2x2_t = vld2q_f64(pyi);

                        let a0 = vld2q_f64(pa0.add(2*i));
                        yv.0 = vfmaq_f64(yv.0, xr, a0.0);
                        yv.0 = vfmsq_f64(yv.0, xi, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xr, a0.1);
                        yv.1 = vfmaq_f64(yv.1, xi, a0.0);

                        vst2q_f64(pyi, yv);

                        i += 2;
                    }

                    while i < mr {
                        let off = 2 * (row_idx + i);
                        let mut yr = *y.get_unchecked(off + 0);
                        let mut yi = *y.get_unchecked(off + 1);

                        let ar = *pa0.add(2*i + 0);
                        let ai = *pa0.add(2*i + 1);
                        let xrs = *x.get_unchecked(2*col_idx + 0);
                        let xis = *x.get_unchecked(2*col_idx + 1);

                        yr += xrs * ar - xis * ai;
                        yi += xrs * ai + xis * ar;

                        *y.get_unchecked_mut(off + 0) = yr;
                        *y.get_unchecked_mut(off + 1) = yi;

                        i += 1;
                    }
                }

                row_idx += mr;
            }

            return;
        }
    } else {
        // non unit stride
        for col_idx in 0..n_cols {
            unsafe {
                let xoff = 2 * (col_idx * incx);
                let xr = *x.get_unchecked(xoff + 0);
                let xi = *x.get_unchecked(xoff + 1);
                if xr != 0.0 || xi != 0.0 {
                    let col_ptr = matrix.as_ptr().add(2 * (col_idx * lda));
                    let col = core::slice::from_raw_parts(col_ptr, 2 * n_rows);
                    zaxpy(n_rows, [xr, xi], col, 1, y, incy);
                }
            }
        }
    }
}

