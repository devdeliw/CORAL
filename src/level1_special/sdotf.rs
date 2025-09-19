//! Computes fused column dots: out := out + A^T x
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m).
//! - `n_cols` (usize)      : Number of columns (n).
//! - `matrix` (&[f32])     : Column-major A with dims (`lda` x `n_cols`).
//! - `lda`    (usize)      : Leading dimension (>= `n_rows`).
//! - `x`      (&[f32])     : Vector of length `n_rows` with stride `incx`.
//! - `incx`   (usize)      : Stride for `x`.
//! - `out`    (&mut [f32]) : Output of length `n_cols`, accumulated in place.
//!
//! # Notes
//! - Fast path when `incx == 1` uses NEON + blocking (MR x NR panels, 8-wide micro-kernel).
//! - Otherwise falls back to scalar strided dots.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vld1q_f32,
    vdupq_n_f32,
    vfmaq_f32,
    vaddvq_f32,
};

use crate::level1::assert_length_helpers::required_len_ok;
use crate::level2::assert_length_helpers::required_len_ok_matrix;
use crate::level1::sdot::sdot;

const MR:  usize = 8;  // rows per panel
const NR:  usize = 8;  // cols per panel

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn sdotf(
    n_rows : usize,
    n_cols : usize,
    matrix : &[f32],
    lda    : usize,
    x      : &[f32],
    incx   : usize,
    out    : &mut [f32],
) {
    // quick return
    if n_rows == 0 || n_cols == 0 { return; }

    debug_assert!(incx > 0, "incx must be non-zero");
    debug_assert!(lda >= n_rows, "lda must be >= n_rows");
    debug_assert!(required_len_ok(x.len(), n_rows, incx), "x too short for n_rows/incx");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n_rows, n_cols, lda),
        "matrix not large enough for n_rows, n_cols, lda"
    );
    debug_assert!(out.len() >= n_cols, "out too small for n_cols");

    // fast path
    if incx == 1 {
        unsafe {
            let a_ptr = matrix.as_ptr();
            let x_ptr = x.as_ptr();
            let row_block = MR;

            // sweep NR-wide panels of c ols 
            let mut col_blk = 0;
            while col_blk < n_cols {
                let nb_eff = core::cmp::min(NR, n_cols - col_blk);

                let mut j = 0;
                while j + 8 <= nb_eff {
                    // column pointers for the 8-wide micro-kernel
                    let pa0 = a_ptr.add((col_blk + j + 0) * lda);
                    let pa1 = a_ptr.add((col_blk + j + 1) * lda);
                    let pa2 = a_ptr.add((col_blk + j + 2) * lda);
                    let pa3 = a_ptr.add((col_blk + j + 3) * lda);
                    let pa4 = a_ptr.add((col_blk + j + 4) * lda);
                    let pa5 = a_ptr.add((col_blk + j + 5) * lda);
                    let pa6 = a_ptr.add((col_blk + j + 6) * lda);
                    let pa7 = a_ptr.add((col_blk + j + 7) * lda);

                    let mut sum0 = 0.0; let mut sum1 = 0.0; let mut sum2 = 0.0; let mut sum3 = 0.0;
                    let mut sum4 = 0.0; let mut sum5 = 0.0; let mut sum6 = 0.0; let mut sum7 = 0.0;

                    // sweep MR-high panels of rows
                    let mut row_idx = 0;
                    while row_idx < n_rows {
                        let mr = core::cmp::min(row_block, n_rows - row_idx);

                        let mut acc0 = vdupq_n_f32(0.0);
                        let mut acc1 = vdupq_n_f32(0.0);
                        let mut acc2 = vdupq_n_f32(0.0);
                        let mut acc3 = vdupq_n_f32(0.0);
                        let mut acc4 = vdupq_n_f32(0.0);
                        let mut acc5 = vdupq_n_f32(0.0);
                        let mut acc6 = vdupq_n_f32(0.0);
                        let mut acc7 = vdupq_n_f32(0.0);

                        let mut i = 0;

                        // 16 rows at a time
                        while i + 16 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i +  0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i +  4));
                            let x2 = vld1q_f32(x_ptr.add(row_idx + i +  8));
                            let x3 = vld1q_f32(x_ptr.add(row_idx + i + 12));

                            let a00 = vld1q_f32(pa0.add(row_idx + i +  0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i +  4));
                            let a02 = vld1q_f32(pa0.add(row_idx + i +  8));
                            let a03 = vld1q_f32(pa0.add(row_idx + i + 12));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);
                            acc0 = vfmaq_f32(acc0, a02, x2);
                            acc0 = vfmaq_f32(acc0, a03, x3);

                            let b00 = vld1q_f32(pa1.add(row_idx + i +  0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i +  4));
                            let b02 = vld1q_f32(pa1.add(row_idx + i +  8));
                            let b03 = vld1q_f32(pa1.add(row_idx + i + 12));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);
                            acc1 = vfmaq_f32(acc1, b02, x2);
                            acc1 = vfmaq_f32(acc1, b03, x3);

                            let c00 = vld1q_f32(pa2.add(row_idx + i +  0));
                            let c01 = vld1q_f32(pa2.add(row_idx + i +  4));
                            let c02 = vld1q_f32(pa2.add(row_idx + i +  8));
                            let c03 = vld1q_f32(pa2.add(row_idx + i + 12));
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc2 = vfmaq_f32(acc2, c01, x1);
                            acc2 = vfmaq_f32(acc2, c02, x2);
                            acc2 = vfmaq_f32(acc2, c03, x3);

                            let d00 = vld1q_f32(pa3.add(row_idx + i +  0));
                            let d01 = vld1q_f32(pa3.add(row_idx + i +  4));
                            let d02 = vld1q_f32(pa3.add(row_idx + i +  8));
                            let d03 = vld1q_f32(pa3.add(row_idx + i + 12));
                            acc3 = vfmaq_f32(acc3, d00, x0);
                            acc3 = vfmaq_f32(acc3, d01, x1);
                            acc3 = vfmaq_f32(acc3, d02, x2);
                            acc3 = vfmaq_f32(acc3, d03, x3);

                            let e00 = vld1q_f32(pa4.add(row_idx + i +  0));
                            let e01 = vld1q_f32(pa4.add(row_idx + i +  4));
                            let e02 = vld1q_f32(pa4.add(row_idx + i +  8));
                            let e03 = vld1q_f32(pa4.add(row_idx + i + 12));
                            acc4 = vfmaq_f32(acc4, e00, x0);
                            acc4 = vfmaq_f32(acc4, e01, x1);
                            acc4 = vfmaq_f32(acc4, e02, x2);
                            acc4 = vfmaq_f32(acc4, e03, x3);

                            let f00 = vld1q_f32(pa5.add(row_idx + i +  0));
                            let f01 = vld1q_f32(pa5.add(row_idx + i +  4));
                            let f02 = vld1q_f32(pa5.add(row_idx + i +  8));
                            let f03 = vld1q_f32(pa5.add(row_idx + i + 12));
                            acc5 = vfmaq_f32(acc5, f00, x0);
                            acc5 = vfmaq_f32(acc5, f01, x1);
                            acc5 = vfmaq_f32(acc5, f02, x2);
                            acc5 = vfmaq_f32(acc5, f03, x3);

                            let g00 = vld1q_f32(pa6.add(row_idx + i +  0));
                            let g01 = vld1q_f32(pa6.add(row_idx + i +  4));
                            let g02 = vld1q_f32(pa6.add(row_idx + i +  8));
                            let g03 = vld1q_f32(pa6.add(row_idx + i + 12));
                            acc6 = vfmaq_f32(acc6, g00, x0);
                            acc6 = vfmaq_f32(acc6, g01, x1);
                            acc6 = vfmaq_f32(acc6, g02, x2);
                            acc6 = vfmaq_f32(acc6, g03, x3);

                            let h00 = vld1q_f32(pa7.add(row_idx + i +  0));
                            let h01 = vld1q_f32(pa7.add(row_idx + i +  4));
                            let h02 = vld1q_f32(pa7.add(row_idx + i +  8));
                            let h03 = vld1q_f32(pa7.add(row_idx + i + 12));
                            acc7 = vfmaq_f32(acc7, h00, x0);
                            acc7 = vfmaq_f32(acc7, h01, x1);
                            acc7 = vfmaq_f32(acc7, h02, x2);
                            acc7 = vfmaq_f32(acc7, h03, x3);

                            i += 16;
                        }

                        // 8 rows at a time
                        while i + 8 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i + 0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i + 4));

                            let a00 = vld1q_f32(pa0.add(row_idx + i + 0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i + 4));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);

                            let b00 = vld1q_f32(pa1.add(row_idx + i + 0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i + 4));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);

                            let c00 = vld1q_f32(pa2.add(row_idx + i + 0));
                            let c01 = vld1q_f32(pa2.add(row_idx + i + 4));
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc2 = vfmaq_f32(acc2, c01, x1);

                            let d00 = vld1q_f32(pa3.add(row_idx + i + 0));
                            let d01 = vld1q_f32(pa3.add(row_idx + i + 4));
                            acc3 = vfmaq_f32(acc3, d00, x0);
                            acc3 = vfmaq_f32(acc3, d01, x1);

                            let e00 = vld1q_f32(pa4.add(row_idx + i + 0));
                            let e01 = vld1q_f32(pa4.add(row_idx + i + 4));
                            acc4 = vfmaq_f32(acc4, e00, x0);
                            acc4 = vfmaq_f32(acc4, e01, x1);

                            let f00 = vld1q_f32(pa5.add(row_idx + i + 0));
                            let f01 = vld1q_f32(pa5.add(row_idx + i + 4));
                            acc5 = vfmaq_f32(acc5, f00, x0);
                            acc5 = vfmaq_f32(acc5, f01, x1);

                            let g00 = vld1q_f32(pa6.add(row_idx + i + 0));
                            let g01 = vld1q_f32(pa6.add(row_idx + i + 4));
                            acc6 = vfmaq_f32(acc6, g00, x0);
                            acc6 = vfmaq_f32(acc6, g01, x1);

                            let h00 = vld1q_f32(pa7.add(row_idx + i + 0));
                            let h01 = vld1q_f32(pa7.add(row_idx + i + 4));
                            acc7 = vfmaq_f32(acc7, h00, x0);
                            acc7 = vfmaq_f32(acc7, h01, x1);

                            i += 8;
                        }

                        // 4 rows at a time
                        while i + 4 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i));

                            let a00 = vld1q_f32(pa0.add(row_idx + i));
                            let b00 = vld1q_f32(pa1.add(row_idx + i));
                            let c00 = vld1q_f32(pa2.add(row_idx + i));
                            let d00 = vld1q_f32(pa3.add(row_idx + i));
                            let e00 = vld1q_f32(pa4.add(row_idx + i));
                            let f00 = vld1q_f32(pa5.add(row_idx + i));
                            let g00 = vld1q_f32(pa6.add(row_idx + i));
                            let h00 = vld1q_f32(pa7.add(row_idx + i));

                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc3 = vfmaq_f32(acc3, d00, x0);
                            acc4 = vfmaq_f32(acc4, e00, x0);
                            acc5 = vfmaq_f32(acc5, f00, x0);
                            acc6 = vfmaq_f32(acc6, g00, x0);
                            acc7 = vfmaq_f32(acc7, h00, x0);

                            i += 4;
                        }

                        // horizontal add + scalar tail
                        sum0 += vaddvq_f32(acc0);
                        sum1 += vaddvq_f32(acc1);
                        sum2 += vaddvq_f32(acc2);
                        sum3 += vaddvq_f32(acc3);
                        sum4 += vaddvq_f32(acc4);
                        sum5 += vaddvq_f32(acc5);
                        sum6 += vaddvq_f32(acc6);
                        sum7 += vaddvq_f32(acc7);

                        while i < mr {
                            let xi = *x_ptr.add(row_idx + i);
                            sum0 += *pa0.add(row_idx + i) * xi;
                            sum1 += *pa1.add(row_idx + i) * xi;
                            sum2 += *pa2.add(row_idx + i) * xi;
                            sum3 += *pa3.add(row_idx + i) * xi;
                            sum4 += *pa4.add(row_idx + i) * xi;
                            sum5 += *pa5.add(row_idx + i) * xi;
                            sum6 += *pa6.add(row_idx + i) * xi;
                            sum7 += *pa7.add(row_idx + i) * xi;
                            i += 1;
                        }

                        row_idx += mr;
                    }

                    *out.get_unchecked_mut(col_blk + j + 0) += sum0;
                    *out.get_unchecked_mut(col_blk + j + 1) += sum1;
                    *out.get_unchecked_mut(col_blk + j + 2) += sum2;
                    *out.get_unchecked_mut(col_blk + j + 3) += sum3;
                    *out.get_unchecked_mut(col_blk + j + 4) += sum4;
                    *out.get_unchecked_mut(col_blk + j + 5) += sum5;
                    *out.get_unchecked_mut(col_blk + j + 6) += sum6;
                    *out.get_unchecked_mut(col_blk + j + 7) += sum7;

                    j += 8;
                }

                while j + 4 <= nb_eff {
                    let pa0 = a_ptr.add((col_blk + j + 0) * lda);
                    let pa1 = a_ptr.add((col_blk + j + 1) * lda);
                    let pa2 = a_ptr.add((col_blk + j + 2) * lda);
                    let pa3 = a_ptr.add((col_blk + j + 3) * lda);

                    let mut sum0 = 0.0; 
                    let mut sum1 = 0.0; 
                    let mut sum2 = 0.0; 
                    let mut sum3 = 0.0;

                    let mut row_idx = 0;
                    while row_idx < n_rows {
                        let mr = core::cmp::min(row_block, n_rows - row_idx);

                        let mut acc0 = vdupq_n_f32(0.0);
                        let mut acc1 = vdupq_n_f32(0.0);
                        let mut acc2 = vdupq_n_f32(0.0);
                        let mut acc3 = vdupq_n_f32(0.0);

                        let mut i = 0;

                        while i + 16 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i +  0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i +  4));
                            let x2 = vld1q_f32(x_ptr.add(row_idx + i +  8));
                            let x3 = vld1q_f32(x_ptr.add(row_idx + i + 12));

                            let a00 = vld1q_f32(pa0.add(row_idx + i +  0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i +  4));
                            let a02 = vld1q_f32(pa0.add(row_idx + i +  8));
                            let a03 = vld1q_f32(pa0.add(row_idx + i + 12));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);
                            acc0 = vfmaq_f32(acc0, a02, x2);
                            acc0 = vfmaq_f32(acc0, a03, x3);

                            let b00 = vld1q_f32(pa1.add(row_idx + i +  0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i +  4));
                            let b02 = vld1q_f32(pa1.add(row_idx + i +  8));
                            let b03 = vld1q_f32(pa1.add(row_idx + i + 12));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);
                            acc1 = vfmaq_f32(acc1, b02, x2);
                            acc1 = vfmaq_f32(acc1, b03, x3);

                            let c00 = vld1q_f32(pa2.add(row_idx + i +  0));
                            let c01 = vld1q_f32(pa2.add(row_idx + i +  4));
                            let c02 = vld1q_f32(pa2.add(row_idx + i +  8));
                            let c03 = vld1q_f32(pa2.add(row_idx + i + 12));
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc2 = vfmaq_f32(acc2, c01, x1);
                            acc2 = vfmaq_f32(acc2, c02, x2);
                            acc2 = vfmaq_f32(acc2, c03, x3);

                            let d00 = vld1q_f32(pa3.add(row_idx + i +  0));
                            let d01 = vld1q_f32(pa3.add(row_idx + i +  4));
                            let d02 = vld1q_f32(pa3.add(row_idx + i +  8));
                            let d03 = vld1q_f32(pa3.add(row_idx + i + 12));
                            acc3 = vfmaq_f32(acc3, d00, x0);
                            acc3 = vfmaq_f32(acc3, d01, x1);
                            acc3 = vfmaq_f32(acc3, d02, x2);
                            acc3 = vfmaq_f32(acc3, d03, x3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i + 0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i + 4));

                            let a00 = vld1q_f32(pa0.add(row_idx + i + 0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i + 4));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);

                            let b00 = vld1q_f32(pa1.add(row_idx + i + 0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i + 4));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);

                            let c00 = vld1q_f32(pa2.add(row_idx + i + 0));
                            let c01 = vld1q_f32(pa2.add(row_idx + i + 4));
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc2 = vfmaq_f32(acc2, c01, x1);

                            let d00 = vld1q_f32(pa3.add(row_idx + i + 0));
                            let d01 = vld1q_f32(pa3.add(row_idx + i + 4));
                            acc3 = vfmaq_f32(acc3, d00, x0);
                            acc3 = vfmaq_f32(acc3, d01, x1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i));

                            let a00 = vld1q_f32(pa0.add(row_idx + i));
                            let b00 = vld1q_f32(pa1.add(row_idx + i));
                            let c00 = vld1q_f32(pa2.add(row_idx + i));
                            let d00 = vld1q_f32(pa3.add(row_idx + i));

                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc2 = vfmaq_f32(acc2, c00, x0);
                            acc3 = vfmaq_f32(acc3, d00, x0);

                            i += 4;
                        }

                        sum0 += vaddvq_f32(acc0);
                        sum1 += vaddvq_f32(acc1);
                        sum2 += vaddvq_f32(acc2);
                        sum3 += vaddvq_f32(acc3);

                        while i < mr {
                            let xi = *x_ptr.add(row_idx + i);
                            sum0 += *pa0.add(row_idx + i) * xi;
                            sum1 += *pa1.add(row_idx + i) * xi;
                            sum2 += *pa2.add(row_idx + i) * xi;
                            sum3 += *pa3.add(row_idx + i) * xi;

                            i += 1;
                        }

                        row_idx += mr;
                    }

                    *out.get_unchecked_mut(col_blk + j + 0) += sum0;
                    *out.get_unchecked_mut(col_blk + j + 1) += sum1;
                    *out.get_unchecked_mut(col_blk + j + 2) += sum2;
                    *out.get_unchecked_mut(col_blk + j + 3) += sum3;

                    j += 4;
                }

                while j + 2 <= nb_eff {
                    let pa0 = a_ptr.add((col_blk + j + 0) * lda);
                    let pa1 = a_ptr.add((col_blk + j + 1) * lda);

                    let mut sum0 = 0.0; let mut sum1 = 0.0;

                    let mut row_idx = 0;
                    while row_idx < n_rows {
                        let mr = core::cmp::min(row_block, n_rows - row_idx);

                        let mut acc0 = vdupq_n_f32(0.0);
                        let mut acc1 = vdupq_n_f32(0.0);

                        let mut i = 0;

                        while i + 16 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i +  0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i +  4));
                            let x2 = vld1q_f32(x_ptr.add(row_idx + i +  8));
                            let x3 = vld1q_f32(x_ptr.add(row_idx + i + 12));

                            let a00 = vld1q_f32(pa0.add(row_idx + i +  0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i +  4));
                            let a02 = vld1q_f32(pa0.add(row_idx + i +  8));
                            let a03 = vld1q_f32(pa0.add(row_idx + i + 12));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);
                            acc0 = vfmaq_f32(acc0, a02, x2);
                            acc0 = vfmaq_f32(acc0, a03, x3);

                            let b00 = vld1q_f32(pa1.add(row_idx + i +  0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i +  4));
                            let b02 = vld1q_f32(pa1.add(row_idx + i +  8));
                            let b03 = vld1q_f32(pa1.add(row_idx + i + 12));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);
                            acc1 = vfmaq_f32(acc1, b02, x2);
                            acc1 = vfmaq_f32(acc1, b03, x3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i + 0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i + 4));

                            let a00 = vld1q_f32(pa0.add(row_idx + i + 0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i + 4));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);

                            let b00 = vld1q_f32(pa1.add(row_idx + i + 0));
                            let b01 = vld1q_f32(pa1.add(row_idx + i + 4));
                            acc1 = vfmaq_f32(acc1, b00, x0);
                            acc1 = vfmaq_f32(acc1, b01, x1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i));

                            let a00 = vld1q_f32(pa0.add(row_idx + i));
                            let b00 = vld1q_f32(pa1.add(row_idx + i));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc1 = vfmaq_f32(acc1, b00, x0);

                            i += 4;
                        }

                        sum0 += vaddvq_f32(acc0);
                        sum1 += vaddvq_f32(acc1);

                        while i < mr {
                            let xi = *x_ptr.add(row_idx + i);
                            sum0 += *pa0.add(row_idx + i) * xi;
                            sum1 += *pa1.add(row_idx + i) * xi;
                            i += 1;
                        }

                        row_idx += mr;
                    }

                    *out.get_unchecked_mut(col_blk + j + 0) += sum0;
                    *out.get_unchecked_mut(col_blk + j + 1) += sum1;

                    j += 2;
                }

                if j < nb_eff {
                    let pa0 = a_ptr.add((col_blk + j) * lda);

                    let mut sum0 = 0.0;

                    let mut row_idx = 0;
                    while row_idx < n_rows {
                        let mr = core::cmp::min(row_block, n_rows - row_idx);

                        let mut acc0 = vdupq_n_f32(0.0);
                        let mut i = 0;

                        while i + 16 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i +  0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i +  4));
                            let x2 = vld1q_f32(x_ptr.add(row_idx + i +  8));
                            let x3 = vld1q_f32(x_ptr.add(row_idx + i + 12));

                            let a00 = vld1q_f32(pa0.add(row_idx + i +  0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i +  4));
                            let a02 = vld1q_f32(pa0.add(row_idx + i +  8));
                            let a03 = vld1q_f32(pa0.add(row_idx + i + 12));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);
                            acc0 = vfmaq_f32(acc0, a02, x2);
                            acc0 = vfmaq_f32(acc0, a03, x3);

                            i += 16;
                        }

                        while i + 8 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i + 0));
                            let x1 = vld1q_f32(x_ptr.add(row_idx + i + 4));
                            let a00 = vld1q_f32(pa0.add(row_idx + i + 0));
                            let a01 = vld1q_f32(pa0.add(row_idx + i + 4));
                            acc0 = vfmaq_f32(acc0, a00, x0);
                            acc0 = vfmaq_f32(acc0, a01, x1);

                            i += 8;
                        }

                        while i + 4 <= mr {
                            let x0 = vld1q_f32(x_ptr.add(row_idx + i));
                            let a00 = vld1q_f32(pa0.add(row_idx + i));
                            acc0 = vfmaq_f32(acc0, a00, x0);

                            i += 4;
                        }

                        sum0 += vaddvq_f32(acc0);

                        while i < mr {
                            let xi = *x_ptr.add(row_idx + i);
                            sum0 += *pa0.add(row_idx + i) * xi;

                            i += 1;
                        }

                        row_idx += mr;
                    }

                    *out.get_unchecked_mut(col_blk + j) += sum0;
                }

                col_blk += nb_eff;
            }

            return;
        }
    }

    // non-unit stride fallback
    for col_idx in 0..n_cols {
        unsafe {
            let col_ptr = matrix.as_ptr().add(col_idx * lda);
            let col = core::slice::from_raw_parts(col_ptr, n_rows);
            let sum = sdot(n_rows, col, 1, x, incx);
            *out.get_unchecked_mut(col_idx) += sum;
        }
    }
}

