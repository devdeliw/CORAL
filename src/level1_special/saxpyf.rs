//! Performs a matrix-vector multiply and accumulation AXPY:
//! ```text
//! y := y + A x
//! ```
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `x`      (&[f32])     : Input vector of length `n_cols`.
//! - `incx`   (usize)      : Stride between consecutive elements of `x`.
//! - `matrix` (&[f32])     : Matrix `A` of dimension (`lda` x `n_cols`).
//! - `lda`    (usize)      : Leading dimension of `A`. Must be >= `n_rows`.
//! - `y`      (&mut [f32]) : Input/output vector of length `n_rows`.
//! - `incy`   (usize)      : Stride between consecutive elements of `y`.
//!
//! # Notes
//! - For unit strides, uses NEON micro-kernels with MR x NR paneling.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vld1q_f32,
    vdupq_n_f32,
    vfmaq_f32,
    vst1q_f32,
};

use crate::level1::assert_length_helpers::required_len_ok;
use crate::level2::assert_length_helpers::required_len_ok_matrix;
use crate::level1::saxpy::saxpy;

const MR:  usize = 128;  // rows per panel
const NR:  usize = 128;  // cols per panel

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn saxpyf(
    n_rows : usize,
    n_cols : usize,
    x      : &[f32],
    incx   : usize,
    matrix : &[f32],
    lda    : usize,
    y      : &mut [f32],
    incy   : usize,
) {
    // quick return
    if n_rows == 0 || n_cols == 0 { return; }

    debug_assert!(incx > 0 && incy > 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n_cols, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n_rows, incy), "y too short for m/incy");
    debug_assert!(lda >= n_rows, "lda must be larger than n_rows");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n_rows, n_cols, lda),
        "matrix not large enough given n_rows, n_cols, and lda col stride"
    );

    // fast path
    if incx == 1 && incy == 1 {
        unsafe {
            let row_block = MR;

            let mut row_idx = 0;
            while row_idx < n_rows {
                let mr = core::cmp::min(row_block, n_rows - row_idx);

                // sweep NR-wide panels of cols
                let mut col_blk = 0;
                while col_blk < n_cols {
                    let nb_eff = core::cmp::min(NR, n_cols - col_blk);

                    let mut j = 0;
                    while j + 8 <= nb_eff {
                        let x0 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 0));
                        let x1 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 1));
                        let x2 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 2));
                        let x3 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 3));
                        let x4 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 4));
                        let x5 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 5));
                        let x6 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 6));
                        let x7 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 7));

                        let pa0 = matrix.as_ptr().add((col_blk + j + 0) * lda + row_idx);
                        let pa1 = matrix.as_ptr().add((col_blk + j + 1) * lda + row_idx);
                        let pa2 = matrix.as_ptr().add((col_blk + j + 2) * lda + row_idx);
                        let pa3 = matrix.as_ptr().add((col_blk + j + 3) * lda + row_idx);
                        let pa4 = matrix.as_ptr().add((col_blk + j + 4) * lda + row_idx);
                        let pa5 = matrix.as_ptr().add((col_blk + j + 5) * lda + row_idx);
                        let pa6 = matrix.as_ptr().add((col_blk + j + 6) * lda + row_idx);
                        let pa7 = matrix.as_ptr().add((col_blk + j + 7) * lda + row_idx);

                        let mut i = 0;

                        // 16 rows at a time
                        while i + 16 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add( 0));
                            let mut y1 = vld1q_f32(py.add( 4));
                            let mut y2 = vld1q_f32(py.add( 8));
                            let mut y3 = vld1q_f32(py.add(12));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);
                            let mut a2p = pa2.add(i);
                            let mut a3p = pa3.add(i);
                            let mut a4p = pa4.add(i);
                            let mut a5p = pa5.add(i);
                            let mut a6p = pa6.add(i);
                            let mut a7p = pa7.add(i);

                            // col 0..7
                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y2 = vfmaq_f32(y2, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y3 = vfmaq_f32(y3, x0, vld1q_f32(a0p));

                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y2 = vfmaq_f32(y2, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y3 = vfmaq_f32(y3, x1, vld1q_f32(a1p));

                            y0 = vfmaq_f32(y0, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y1 = vfmaq_f32(y1, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y2 = vfmaq_f32(y2, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y3 = vfmaq_f32(y3, x2, vld1q_f32(a2p));

                            y0 = vfmaq_f32(y0, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y1 = vfmaq_f32(y1, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y2 = vfmaq_f32(y2, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y3 = vfmaq_f32(y3, x3, vld1q_f32(a3p));

                            y0 = vfmaq_f32(y0, x4, vld1q_f32(a4p)); a4p = a4p.add(4);
                            y1 = vfmaq_f32(y1, x4, vld1q_f32(a4p)); a4p = a4p.add(4);
                            y2 = vfmaq_f32(y2, x4, vld1q_f32(a4p)); a4p = a4p.add(4);
                            y3 = vfmaq_f32(y3, x4, vld1q_f32(a4p));

                            y0 = vfmaq_f32(y0, x5, vld1q_f32(a5p)); a5p = a5p.add(4);
                            y1 = vfmaq_f32(y1, x5, vld1q_f32(a5p)); a5p = a5p.add(4);
                            y2 = vfmaq_f32(y2, x5, vld1q_f32(a5p)); a5p = a5p.add(4);
                            y3 = vfmaq_f32(y3, x5, vld1q_f32(a5p));

                            y0 = vfmaq_f32(y0, x6, vld1q_f32(a6p)); a6p = a6p.add(4);
                            y1 = vfmaq_f32(y1, x6, vld1q_f32(a6p)); a6p = a6p.add(4);
                            y2 = vfmaq_f32(y2, x6, vld1q_f32(a6p)); a6p = a6p.add(4);
                            y3 = vfmaq_f32(y3, x6, vld1q_f32(a6p));

                            y0 = vfmaq_f32(y0, x7, vld1q_f32(a7p)); a7p = a7p.add(4);
                            y1 = vfmaq_f32(y1, x7, vld1q_f32(a7p)); a7p = a7p.add(4);
                            y2 = vfmaq_f32(y2, x7, vld1q_f32(a7p)); a7p = a7p.add(4);
                            y3 = vfmaq_f32(y3, x7, vld1q_f32(a7p));

                            vst1q_f32(py.add( 0), y0);
                            vst1q_f32(py.add( 4), y1);
                            vst1q_f32(py.add( 8), y2);
                            vst1q_f32(py.add(12), y3);

                            i += 16;
                        }

                        // 8 rows at a time
                        while i + 8 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add(0));
                            let mut y1 = vld1q_f32(py.add(4));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);
                            let mut a2p = pa2.add(i);
                            let mut a3p = pa3.add(i);
                            let mut a4p = pa4.add(i);
                            let mut a5p = pa5.add(i);
                            let mut a6p = pa6.add(i);
                            let mut a7p = pa7.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p));
                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p));
                            y0 = vfmaq_f32(y0, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y1 = vfmaq_f32(y1, x2, vld1q_f32(a2p));
                            y0 = vfmaq_f32(y0, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y1 = vfmaq_f32(y1, x3, vld1q_f32(a3p));
                            y0 = vfmaq_f32(y0, x4, vld1q_f32(a4p)); a4p = a4p.add(4);
                            y1 = vfmaq_f32(y1, x4, vld1q_f32(a4p));
                            y0 = vfmaq_f32(y0, x5, vld1q_f32(a5p)); a5p = a5p.add(4);
                            y1 = vfmaq_f32(y1, x5, vld1q_f32(a5p));
                            y0 = vfmaq_f32(y0, x6, vld1q_f32(a6p)); a6p = a6p.add(4);
                            y1 = vfmaq_f32(y1, x6, vld1q_f32(a6p));
                            y0 = vfmaq_f32(y0, x7, vld1q_f32(a7p)); a7p = a7p.add(4);
                            y1 = vfmaq_f32(y1, x7, vld1q_f32(a7p));

                            vst1q_f32(py.add(0), y0);
                            vst1q_f32(py.add(4), y1);

                            i += 8;
                        }

                        // 4 rows at a time
                        while i + 4 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);
                            let mut y0 = vld1q_f32(py);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(pa0.add(i)));
                            y0 = vfmaq_f32(y0, x1, vld1q_f32(pa1.add(i)));
                            y0 = vfmaq_f32(y0, x2, vld1q_f32(pa2.add(i)));
                            y0 = vfmaq_f32(y0, x3, vld1q_f32(pa3.add(i)));
                            y0 = vfmaq_f32(y0, x4, vld1q_f32(pa4.add(i)));
                            y0 = vfmaq_f32(y0, x5, vld1q_f32(pa5.add(i)));
                            y0 = vfmaq_f32(y0, x6, vld1q_f32(pa6.add(i)));
                            y0 = vfmaq_f32(y0, x7, vld1q_f32(pa7.add(i)));

                            vst1q_f32(py, y0);
                            i += 4;
                        }

                        // tail over rows
                        while i < mr {
                            let dst = y.as_mut_ptr().add(row_idx + i);
                            let mut acc = *dst;

                            acc += *x.get_unchecked(col_blk + j + 0) * *pa0.add(i);
                            acc += *x.get_unchecked(col_blk + j + 1) * *pa1.add(i);
                            acc += *x.get_unchecked(col_blk + j + 2) * *pa2.add(i);
                            acc += *x.get_unchecked(col_blk + j + 3) * *pa3.add(i);
                            acc += *x.get_unchecked(col_blk + j + 4) * *pa4.add(i);
                            acc += *x.get_unchecked(col_blk + j + 5) * *pa5.add(i);
                            acc += *x.get_unchecked(col_blk + j + 6) * *pa6.add(i);
                            acc += *x.get_unchecked(col_blk + j + 7) * *pa7.add(i);

                            *dst = acc;
                            i += 1;
                        }

                        j += 8;
                    }

                    while j + 4 <= nb_eff {
                        let x0 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 0));
                        let x1 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 1));
                        let x2 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 2));
                        let x3 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 3));

                        let pa0 = matrix.as_ptr().add((col_blk + j + 0) * lda + row_idx);
                        let pa1 = matrix.as_ptr().add((col_blk + j + 1) * lda + row_idx);
                        let pa2 = matrix.as_ptr().add((col_blk + j + 2) * lda + row_idx);
                        let pa3 = matrix.as_ptr().add((col_blk + j + 3) * lda + row_idx);

                        let mut i = 0;

                        while i + 16 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add( 0));
                            let mut y1 = vld1q_f32(py.add( 4));
                            let mut y2 = vld1q_f32(py.add( 8));
                            let mut y3 = vld1q_f32(py.add(12));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);
                            let mut a2p = pa2.add(i);
                            let mut a3p = pa3.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y2 = vfmaq_f32(y2, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y3 = vfmaq_f32(y3, x0, vld1q_f32(a0p));

                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y2 = vfmaq_f32(y2, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y3 = vfmaq_f32(y3, x1, vld1q_f32(a1p));

                            y0 = vfmaq_f32(y0, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y1 = vfmaq_f32(y1, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y2 = vfmaq_f32(y2, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y3 = vfmaq_f32(y3, x2, vld1q_f32(a2p));

                            y0 = vfmaq_f32(y0, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y1 = vfmaq_f32(y1, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y2 = vfmaq_f32(y2, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y3 = vfmaq_f32(y3, x3, vld1q_f32(a3p));

                            vst1q_f32(py.add( 0), y0);
                            vst1q_f32(py.add( 4), y1);
                            vst1q_f32(py.add( 8), y2);
                            vst1q_f32(py.add(12), y3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add(0));
                            let mut y1 = vld1q_f32(py.add(4));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);
                            let mut a2p = pa2.add(i);
                            let mut a3p = pa3.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p));
                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p));
                            y0 = vfmaq_f32(y0, x2, vld1q_f32(a2p)); a2p = a2p.add(4);
                            y1 = vfmaq_f32(y1, x2, vld1q_f32(a2p));
                            y0 = vfmaq_f32(y0, x3, vld1q_f32(a3p)); a3p = a3p.add(4);
                            y1 = vfmaq_f32(y1, x3, vld1q_f32(a3p));

                            vst1q_f32(py.add(0), y0);
                            vst1q_f32(py.add(4), y1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);
                            let mut y0 = vld1q_f32(py);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(pa0.add(i)));
                            y0 = vfmaq_f32(y0, x1, vld1q_f32(pa1.add(i)));
                            y0 = vfmaq_f32(y0, x2, vld1q_f32(pa2.add(i)));
                            y0 = vfmaq_f32(y0, x3, vld1q_f32(pa3.add(i)));

                            vst1q_f32(py, y0);

                            i += 4;
                        }

                        while i < mr {
                            let dst = y.as_mut_ptr().add(row_idx + i);
                            let mut acc = *dst;

                            acc += *x.get_unchecked(col_blk + j + 0) * *pa0.add(i);
                            acc += *x.get_unchecked(col_blk + j + 1) * *pa1.add(i);
                            acc += *x.get_unchecked(col_blk + j + 2) * *pa2.add(i);
                            acc += *x.get_unchecked(col_blk + j + 3) * *pa3.add(i);

                            *dst = acc;
                            i += 1;
                        }

                        j += 4;
                    }

                    while j + 2 <= nb_eff {
                        let x0 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 0));
                        let x1 = vdupq_n_f32(*x.get_unchecked(col_blk + j + 1));

                        let pa0 = matrix.as_ptr().add((col_blk + j + 0) * lda + row_idx);
                        let pa1 = matrix.as_ptr().add((col_blk + j + 1) * lda + row_idx);

                        let mut i = 0;

                        while i + 16 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add( 0));
                            let mut y1 = vld1q_f32(py.add( 4));
                            let mut y2 = vld1q_f32(py.add( 8));
                            let mut y3 = vld1q_f32(py.add(12));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y2 = vfmaq_f32(y2, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y3 = vfmaq_f32(y3, x0, vld1q_f32(a0p));

                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y2 = vfmaq_f32(y2, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y3 = vfmaq_f32(y3, x1, vld1q_f32(a1p));

                            vst1q_f32(py.add( 0), y0);
                            vst1q_f32(py.add( 4), y1);
                            vst1q_f32(py.add( 8), y2);
                            vst1q_f32(py.add(12), y3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add(0));
                            let mut y1 = vld1q_f32(py.add(4));

                            let mut a0p = pa0.add(i);
                            let mut a1p = pa1.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p));

                            y0 = vfmaq_f32(y0, x1, vld1q_f32(a1p)); a1p = a1p.add(4);
                            y1 = vfmaq_f32(y1, x1, vld1q_f32(a1p));

                            vst1q_f32(py.add(0), y0);
                            vst1q_f32(py.add(4), y1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);
                            let mut y0 = vld1q_f32(py);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(pa0.add(i)));
                            y0 = vfmaq_f32(y0, x1, vld1q_f32(pa1.add(i)));

                            vst1q_f32(py, y0);
                            i += 4;
                        }

                        while i < mr {
                            let dst = y.as_mut_ptr().add(row_idx + i);
                            let mut acc = *dst;
                            acc += *x.get_unchecked(col_blk + j + 0) * *pa0.add(i);
                            acc += *x.get_unchecked(col_blk + j + 1) * *pa1.add(i);
                            *dst = acc;
                            i += 1;
                        }

                        j += 2;
                    }

                    if j < nb_eff {
                        let x0 = vdupq_n_f32(*x.get_unchecked(col_blk + j));
                        let pa0 = matrix.as_ptr().add((col_blk + j) * lda + row_idx);

                        let mut i = 0;

                        while i + 16 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);

                            let mut y0 = vld1q_f32(py.add( 0));
                            let mut y1 = vld1q_f32(py.add( 4));
                            let mut y2 = vld1q_f32(py.add( 8));
                            let mut y3 = vld1q_f32(py.add(12));

                            let mut a0p = pa0.add(i);

                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y2 = vfmaq_f32(y2, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y3 = vfmaq_f32(y3, x0, vld1q_f32(a0p));

                            vst1q_f32(py.add( 0), y0);
                            vst1q_f32(py.add( 4), y1);
                            vst1q_f32(py.add( 8), y2);
                            vst1q_f32(py.add(12), y3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);
                            let mut y0 = vld1q_f32(py.add(0));
                            let mut y1 = vld1q_f32(py.add(4));

                            let mut a0p = pa0.add(i);
                            y0 = vfmaq_f32(y0, x0, vld1q_f32(a0p)); a0p = a0p.add(4);
                            y1 = vfmaq_f32(y1, x0, vld1q_f32(a0p));

                            vst1q_f32(py.add(0), y0);
                            vst1q_f32(py.add(4), y1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let py = y.as_mut_ptr().add(row_idx + i);
                            let y0 = vfmaq_f32(vld1q_f32(py), x0, vld1q_f32(pa0.add(i)));
                            vst1q_f32(py, y0);

                            i += 4;
                        }

                        while i < mr {
                            let dst = y.as_mut_ptr().add(row_idx + i);
                            *dst += *x.get_unchecked(col_blk + j) * *pa0.add(i);

                            i += 1;
                        }
                    }

                    col_blk += nb_eff;
                }

                row_idx += mr;
            }

            return;
        }
    }

    // non-unit stride
    for col_idx in 0..n_cols {
        unsafe {
            let scaled = *x.get_unchecked(col_idx * incx);
            if scaled != 0.0 {
                let col_ptr = matrix.as_ptr().add(col_idx * lda);
                let col = core::slice::from_raw_parts(col_ptr, n_rows);
                saxpy(n_rows, scaled, col, 1, y, incy);
            }
        }
    }
}

