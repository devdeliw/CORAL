//! Computes fused column dots (conjugated): out := out + conj(A)^T x
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of complex rows (m).
//! - `n_cols` (usize)      : Number of columns (n).
//! - `matrix` (&[f64])     : Column-major A (interleaved) with dims (`lda` x `n_cols`).
//! - `lda`    (usize)      : Leading dimension in complex elements (>= `n_rows`).
//! - `x`      (&[f64])     : Complex vector of length `n_rows` (interleaved) with stride `incx`.
//! - `incx`   (usize)      : Stride for `x` in complex elements.
//! - `out`    (&mut [f64]) : Complex output (interleaved) of length `n_cols`, accumulated in place.
//!
//! # Notes
//! - Fast path when `incx == 1` uses NEON + blocking (NR=8, MC=128).
//! - Otherwise falls back to level-1 `zdotc` per column.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vld1q_f64,
    vdupq_n_f64,
    vfmaq_f64,
    vfmsq_f64,
    vaddvq_f64,
    vuzp1q_f64,
    vuzp2q_f64,
};

use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;
use crate::level1::zdotc::zdotc;

const MC: usize = 128;
const NR: usize = 8;

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn zdotcf(
    n_rows : usize,
    n_cols : usize,
    matrix : &[f64],
    lda    : usize,
    x      : &[f64],
    incx   : usize,
    out    : &mut [f64],
) {
    if n_rows == 0 || n_cols == 0 { return; }

    debug_assert!(incx > 0, "incx must be non-zero");
    debug_assert!(lda >= n_rows, "lda must be >= n_rows");
    debug_assert!(required_len_ok_cplx(x.len(), n_rows, incx), "x too short for n_rows/incx");
    debug_assert!(out.len() >= 2 * n_cols, "out too small for n_cols");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n_rows, n_cols, lda),
        "matrix not large enough for n_rows, n_cols, lda"
    );

    if incx == 1 {
        unsafe {
            let a_ptr = matrix.as_ptr();
            let x_ptr = x.as_ptr();

            let row_block = MC;
            let mut col_idx = 0;

            // 8 columns at a time
            while col_idx + NR <= n_cols {
                let pa0 = a_ptr.add(2 * (col_idx * lda));
                let pa1 = a_ptr.add(2 * ((col_idx + 1) * lda));
                let pa2 = a_ptr.add(2 * ((col_idx + 2) * lda));
                let pa3 = a_ptr.add(2 * ((col_idx + 3) * lda));
                let pa4 = a_ptr.add(2 * ((col_idx + 4) * lda));
                let pa5 = a_ptr.add(2 * ((col_idx + 5) * lda));
                let pa6 = a_ptr.add(2 * ((col_idx + 6) * lda));
                let pa7 = a_ptr.add(2 * ((col_idx + 7) * lda));

                let mut sum_re0 = 0.0; let mut sum_im0 = 0.0;
                let mut sum_re1 = 0.0; let mut sum_im1 = 0.0;
                let mut sum_re2 = 0.0; let mut sum_im2 = 0.0;
                let mut sum_re3 = 0.0; let mut sum_im3 = 0.0;
                let mut sum_re4 = 0.0; let mut sum_im4 = 0.0;
                let mut sum_re5 = 0.0; let mut sum_im5 = 0.0;
                let mut sum_re6 = 0.0; let mut sum_im6 = 0.0;
                let mut sum_re7 = 0.0; let mut sum_im7 = 0.0;

                let mut row_idx = 0;
                while row_idx < n_rows {
                    let mr = core::cmp::min(row_block, n_rows - row_idx);

                    let mut acc_re0 = vdupq_n_f64(0.0); let mut acc_im0 = vdupq_n_f64(0.0);
                    let mut acc_re1 = vdupq_n_f64(0.0); let mut acc_im1 = vdupq_n_f64(0.0);
                    let mut acc_re2 = vdupq_n_f64(0.0); let mut acc_im2 = vdupq_n_f64(0.0);
                    let mut acc_re3 = vdupq_n_f64(0.0); let mut acc_im3 = vdupq_n_f64(0.0);
                    let mut acc_re4 = vdupq_n_f64(0.0); let mut acc_im4 = vdupq_n_f64(0.0);
                    let mut acc_re5 = vdupq_n_f64(0.0); let mut acc_im5 = vdupq_n_f64(0.0);
                    let mut acc_re6 = vdupq_n_f64(0.0); let mut acc_im6 = vdupq_n_f64(0.0);
                    let mut acc_re7 = vdupq_n_f64(0.0); let mut acc_im7 = vdupq_n_f64(0.0);

                    let col0 = pa0.add(2 * row_idx);
                    let col1 = pa1.add(2 * row_idx);
                    let col2 = pa2.add(2 * row_idx);
                    let col3 = pa3.add(2 * row_idx);
                    let col4 = pa4.add(2 * row_idx);
                    let col5 = pa5.add(2 * row_idx);
                    let col6 = pa6.add(2 * row_idx);
                    let col7 = pa7.add(2 * row_idx);

                    let mut i = 0;

                    // 8 complex
                    while i + 8 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let x4 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 4)));
                        let x5 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 5)));
                        let x6 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 6)));
                        let x7 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 7)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);
                        let xr2 = vuzp1q_f64(x4, x5); let xi2 = vuzp2q_f64(x4, x5);
                        let xr3 = vuzp1q_f64(x6, x7); let xi3 = vuzp2q_f64(x6, x7);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let a2 = vld1q_f64(col0.add(2 * (i + 2)));
                        let a3 = vld1q_f64(col0.add(2 * (i + 3)));
                        let a4 = vld1q_f64(col0.add(2 * (i + 4)));
                        let a5 = vld1q_f64(col0.add(2 * (i + 5)));
                        let a6 = vld1q_f64(col0.add(2 * (i + 6)));
                        let a7 = vld1q_f64(col0.add(2 * (i + 7)));
                        let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                        let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                        let ar2 = vuzp1q_f64(a4, a5); let ai2 = vuzp2q_f64(a4, a5);
                        let ar3 = vuzp1q_f64(a6, a7); let ai3 = vuzp2q_f64(a6, a7);
                        acc_re0 = vfmaq_f64(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f64(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f64(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f64(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f64(acc_im0, ai1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ar2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ai2, xi2);
                        acc_im0 = vfmaq_f64(acc_im0, ar2, xi2);
                        acc_im0 = vfmsq_f64(acc_im0, ai2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ar3, xr3);
                        acc_re0 = vfmaq_f64(acc_re0, ai3, xi3);
                        acc_im0 = vfmaq_f64(acc_im0, ar3, xi3);
                        acc_im0 = vfmsq_f64(acc_im0, ai3, xr3);

                        let b0 = vld1q_f64(col1.add(2 * (i + 0)));
                        let b1 = vld1q_f64(col1.add(2 * (i + 1)));
                        let b2 = vld1q_f64(col1.add(2 * (i + 2)));
                        let b3 = vld1q_f64(col1.add(2 * (i + 3)));
                        let b4 = vld1q_f64(col1.add(2 * (i + 4)));
                        let b5 = vld1q_f64(col1.add(2 * (i + 5)));
                        let b6 = vld1q_f64(col1.add(2 * (i + 6)));
                        let b7 = vld1q_f64(col1.add(2 * (i + 7)));
                        let br0 = vuzp1q_f64(b0, b1); let bi0 = vuzp2q_f64(b0, b1);
                        let br1 = vuzp1q_f64(b2, b3); let bi1 = vuzp2q_f64(b2, b3);
                        let br2 = vuzp1q_f64(b4, b5); let bi2 = vuzp2q_f64(b4, b5);
                        let br3 = vuzp1q_f64(b6, b7); let bi3 = vuzp2q_f64(b6, b7);
                        acc_re1 = vfmaq_f64(acc_re1, br0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, bi0, xi0);
                        acc_im1 = vfmaq_f64(acc_im1, br0, xi0);
                        acc_im1 = vfmsq_f64(acc_im1, bi0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, br1, xr1);
                        acc_re1 = vfmaq_f64(acc_re1, bi1, xi1);
                        acc_im1 = vfmaq_f64(acc_im1, br1, xi1);
                        acc_im1 = vfmsq_f64(acc_im1, bi1, xr1);
                        acc_re1 = vfmaq_f64(acc_re1, br2, xr2);
                        acc_re1 = vfmaq_f64(acc_re1, bi2, xi2);
                        acc_im1 = vfmaq_f64(acc_im1, br2, xi2);
                        acc_im1 = vfmsq_f64(acc_im1, bi2, xr2);
                        acc_re1 = vfmaq_f64(acc_re1, br3, xr3);
                        acc_re1 = vfmaq_f64(acc_re1, bi3, xi3);
                        acc_im1 = vfmaq_f64(acc_im1, br3, xi3);
                        acc_im1 = vfmsq_f64(acc_im1, bi3, xr3);

                        let c0 = vld1q_f64(col2.add(2 * (i + 0)));
                        let c1 = vld1q_f64(col2.add(2 * (i + 1)));
                        let c2 = vld1q_f64(col2.add(2 * (i + 2)));
                        let c3 = vld1q_f64(col2.add(2 * (i + 3)));
                        let c4 = vld1q_f64(col2.add(2 * (i + 4)));
                        let c5 = vld1q_f64(col2.add(2 * (i + 5)));
                        let c6 = vld1q_f64(col2.add(2 * (i + 6)));
                        let c7 = vld1q_f64(col2.add(2 * (i + 7)));
                        let cr0 = vuzp1q_f64(c0, c1); let ci0 = vuzp2q_f64(c0, c1);
                        let cr1 = vuzp1q_f64(c2, c3); let ci1 = vuzp2q_f64(c2, c3);
                        let cr2 = vuzp1q_f64(c4, c5); let ci2 = vuzp2q_f64(c4, c5);
                        let cr3 = vuzp1q_f64(c6, c7); let ci3 = vuzp2q_f64(c6, c7);
                        acc_re2 = vfmaq_f64(acc_re2, cr0, xr0);
                        acc_re2 = vfmaq_f64(acc_re2, ci0, xi0);
                        acc_im2 = vfmaq_f64(acc_im2, cr0, xi0);
                        acc_im2 = vfmsq_f64(acc_im2, ci0, xr0);
                        acc_re2 = vfmaq_f64(acc_re2, cr1, xr1);
                        acc_re2 = vfmaq_f64(acc_re2, ci1, xi1);
                        acc_im2 = vfmaq_f64(acc_im2, cr1, xi1);
                        acc_im2 = vfmsq_f64(acc_im2, ci1, xr1);
                        acc_re2 = vfmaq_f64(acc_re2, cr2, xr2);
                        acc_re2 = vfmaq_f64(acc_re2, ci2, xi2);
                        acc_im2 = vfmaq_f64(acc_im2, cr2, xi2);
                        acc_im2 = vfmsq_f64(acc_im2, ci2, xr2);
                        acc_re2 = vfmaq_f64(acc_re2, cr3, xr3);
                        acc_re2 = vfmaq_f64(acc_re2, ci3, xi3);
                        acc_im2 = vfmaq_f64(acc_im2, cr3, xi3);
                        acc_im2 = vfmsq_f64(acc_im2, ci3, xr3);

                        let d0 = vld1q_f64(col3.add(2 * (i + 0)));
                        let d1 = vld1q_f64(col3.add(2 * (i + 1)));
                        let d2 = vld1q_f64(col3.add(2 * (i + 2)));
                        let d3 = vld1q_f64(col3.add(2 * (i + 3)));
                        let d4 = vld1q_f64(col3.add(2 * (i + 4)));
                        let d5 = vld1q_f64(col3.add(2 * (i + 5)));
                        let d6 = vld1q_f64(col3.add(2 * (i + 6)));
                        let d7 = vld1q_f64(col3.add(2 * (i + 7)));
                        let dr0 = vuzp1q_f64(d0, d1); let di0 = vuzp2q_f64(d0, d1);
                        let dr1 = vuzp1q_f64(d2, d3); let di1 = vuzp2q_f64(d2, d3);
                        let dr2 = vuzp1q_f64(d4, d5); let di2 = vuzp2q_f64(d4, d5);
                        let dr3 = vuzp1q_f64(d6, d7); let di3 = vuzp2q_f64(d6, d7);
                        acc_re3 = vfmaq_f64(acc_re3, dr0, xr0);
                        acc_re3 = vfmaq_f64(acc_re3, di0, xi0);
                        acc_im3 = vfmaq_f64(acc_im3, dr0, xi0);
                        acc_im3 = vfmsq_f64(acc_im3, di0, xr0);
                        acc_re3 = vfmaq_f64(acc_re3, dr1, xr1);
                        acc_re3 = vfmaq_f64(acc_re3, di1, xi1);
                        acc_im3 = vfmaq_f64(acc_im3, dr1, xi1);
                        acc_im3 = vfmsq_f64(acc_im3, di1, xr1);
                        acc_re3 = vfmaq_f64(acc_re3, dr2, xr2);
                        acc_re3 = vfmaq_f64(acc_re3, di2, xi2);
                        acc_im3 = vfmaq_f64(acc_im3, dr2, xi2);
                        acc_im3 = vfmsq_f64(acc_im3, di2, xr2);
                        acc_re3 = vfmaq_f64(acc_re3, dr3, xr3);
                        acc_re3 = vfmaq_f64(acc_re3, di3, xi3);
                        acc_im3 = vfmaq_f64(acc_im3, dr3, xi3);
                        acc_im3 = vfmsq_f64(acc_im3, di3, xr3);

                        let e0 = vld1q_f64(col4.add(2 * (i + 0)));
                        let e1 = vld1q_f64(col4.add(2 * (i + 1)));
                        let e2 = vld1q_f64(col4.add(2 * (i + 2)));
                        let e3 = vld1q_f64(col4.add(2 * (i + 3)));
                        let e4 = vld1q_f64(col4.add(2 * (i + 4)));
                        let e5 = vld1q_f64(col4.add(2 * (i + 5)));
                        let e6 = vld1q_f64(col4.add(2 * (i + 6)));
                        let e7 = vld1q_f64(col4.add(2 * (i + 7)));
                        let er0 = vuzp1q_f64(e0, e1); let ei0 = vuzp2q_f64(e0, e1);
                        let er1 = vuzp1q_f64(e2, e3); let ei1 = vuzp2q_f64(e2, e3);
                        let er2 = vuzp1q_f64(e4, e5); let ei2 = vuzp2q_f64(e4, e5);
                        let er3 = vuzp1q_f64(e6, e7); let ei3 = vuzp2q_f64(e6, e7);
                        acc_re4 = vfmaq_f64(acc_re4, er0, xr0);
                        acc_re4 = vfmaq_f64(acc_re4, ei0, xi0);
                        acc_im4 = vfmaq_f64(acc_im4, er0, xi0);
                        acc_im4 = vfmsq_f64(acc_im4, ei0, xr0);
                        acc_re4 = vfmaq_f64(acc_re4, er1, xr1);
                        acc_re4 = vfmaq_f64(acc_re4, ei1, xi1);
                        acc_im4 = vfmaq_f64(acc_im4, er1, xi1);
                        acc_im4 = vfmsq_f64(acc_im4, ei1, xr1);
                        acc_re4 = vfmaq_f64(acc_re4, er2, xr2);
                        acc_re4 = vfmaq_f64(acc_re4, ei2, xi2);
                        acc_im4 = vfmaq_f64(acc_im4, er2, xi2);
                        acc_im4 = vfmsq_f64(acc_im4, ei2, xr2);
                        acc_re4 = vfmaq_f64(acc_re4, er3, xr3);
                        acc_re4 = vfmaq_f64(acc_re4, ei3, xi3);
                        acc_im4 = vfmaq_f64(acc_im4, er3, xi3);
                        acc_im4 = vfmsq_f64(acc_im4, ei3, xr3);

                        let f0 = vld1q_f64(col5.add(2 * (i + 0)));
                        let f1 = vld1q_f64(col5.add(2 * (i + 1)));
                        let f2 = vld1q_f64(col5.add(2 * (i + 2)));
                        let f3 = vld1q_f64(col5.add(2 * (i + 3)));
                        let f4 = vld1q_f64(col5.add(2 * (i + 4)));
                        let f5 = vld1q_f64(col5.add(2 * (i + 5)));
                        let f6 = vld1q_f64(col5.add(2 * (i + 6)));
                        let f7 = vld1q_f64(col5.add(2 * (i + 7)));
                        let fr0 = vuzp1q_f64(f0, f1); let fi0 = vuzp2q_f64(f0, f1);
                        let fr1 = vuzp1q_f64(f2, f3); let fi1 = vuzp2q_f64(f2, f3);
                        let fr2 = vuzp1q_f64(f4, f5); let fi2 = vuzp2q_f64(f4, f5);
                        let fr3 = vuzp1q_f64(f6, f7); let fi3 = vuzp2q_f64(f6, f7);
                        acc_re5 = vfmaq_f64(acc_re5, fr0, xr0);
                        acc_re5 = vfmaq_f64(acc_re5, fi0, xi0);
                        acc_im5 = vfmaq_f64(acc_im5, fr0, xi0);
                        acc_im5 = vfmsq_f64(acc_im5, fi0, xr0);
                        acc_re5 = vfmaq_f64(acc_re5, fr1, xr1);
                        acc_re5 = vfmaq_f64(acc_re5, fi1, xi1);
                        acc_im5 = vfmaq_f64(acc_im5, fr1, xi1);
                        acc_im5 = vfmsq_f64(acc_im5, fi1, xr1);
                        acc_re5 = vfmaq_f64(acc_re5, fr2, xr2);
                        acc_re5 = vfmaq_f64(acc_re5, fi2, xi2);
                        acc_im5 = vfmaq_f64(acc_im5, fr2, xi2);
                        acc_im5 = vfmsq_f64(acc_im5, fi2, xr2);
                        acc_re5 = vfmaq_f64(acc_re5, fr3, xr3);
                        acc_re5 = vfmaq_f64(acc_re5, fi3, xi3);
                        acc_im5 = vfmaq_f64(acc_im5, fr3, xi3);
                        acc_im5 = vfmsq_f64(acc_im5, fi3, xr3);

                        let g0 = vld1q_f64(col6.add(2 * (i + 0)));
                        let g1 = vld1q_f64(col6.add(2 * (i + 1)));
                        let g2 = vld1q_f64(col6.add(2 * (i + 2)));
                        let g3 = vld1q_f64(col6.add(2 * (i + 3)));
                        let g4 = vld1q_f64(col6.add(2 * (i + 4)));
                        let g5 = vld1q_f64(col6.add(2 * (i + 5)));
                        let g6 = vld1q_f64(col6.add(2 * (i + 6)));
                        let g7 = vld1q_f64(col6.add(2 * (i + 7)));
                        let gr0 = vuzp1q_f64(g0, g1); let gi0 = vuzp2q_f64(g0, g1);
                        let gr1 = vuzp1q_f64(g2, g3); let gi1 = vuzp2q_f64(g2, g3);
                        let gr2 = vuzp1q_f64(g4, g5); let gi2 = vuzp2q_f64(g4, g5);
                        let gr3 = vuzp1q_f64(g6, g7); let gi3 = vuzp2q_f64(g6, g7);
                        acc_re6 = vfmaq_f64(acc_re6, gr0, xr0);
                        acc_re6 = vfmaq_f64(acc_re6, gi0, xi0);
                        acc_im6 = vfmaq_f64(acc_im6, gr0, xi0);
                        acc_im6 = vfmsq_f64(acc_im6, gi0, xr0);
                        acc_re6 = vfmaq_f64(acc_re6, gr1, xr1);
                        acc_re6 = vfmaq_f64(acc_re6, gi1, xi1);
                        acc_im6 = vfmaq_f64(acc_im6, gr1, xi1);
                        acc_im6 = vfmsq_f64(acc_im6, gi1, xr1);
                        acc_re6 = vfmaq_f64(acc_re6, gr2, xr2);
                        acc_re6 = vfmaq_f64(acc_re6, gi2, xi2);
                        acc_im6 = vfmaq_f64(acc_im6, gr2, xi2);
                        acc_im6 = vfmsq_f64(acc_im6, gi2, xr2);
                        acc_re6 = vfmaq_f64(acc_re6, gr3, xr3);
                        acc_re6 = vfmaq_f64(acc_re6, gi3, xi3);
                        acc_im6 = vfmaq_f64(acc_im6, gr3, xi3);
                        acc_im6 = vfmsq_f64(acc_im6, gi3, xr3);

                        let h0 = vld1q_f64(col7.add(2 * (i + 0)));
                        let h1 = vld1q_f64(col7.add(2 * (i + 1)));
                        let h2 = vld1q_f64(col7.add(2 * (i + 2)));
                        let h3 = vld1q_f64(col7.add(2 * (i + 3)));
                        let h4 = vld1q_f64(col7.add(2 * (i + 4)));
                        let h5 = vld1q_f64(col7.add(2 * (i + 5)));
                        let h6 = vld1q_f64(col7.add(2 * (i + 6)));
                        let h7 = vld1q_f64(col7.add(2 * (i + 7)));
                        let hr0 = vuzp1q_f64(h0, h1); let hi0 = vuzp2q_f64(h0, h1);
                        let hr1 = vuzp1q_f64(h2, h3); let hi1 = vuzp2q_f64(h2, h3);
                        let hr2 = vuzp1q_f64(h4, h5); let hi2 = vuzp2q_f64(h4, h5);
                        let hr3 = vuzp1q_f64(h6, h7); let hi3 = vuzp2q_f64(h6, h7);
                        acc_re7 = vfmaq_f64(acc_re7, hr0, xr0);
                        acc_re7 = vfmaq_f64(acc_re7, hi0, xi0);
                        acc_im7 = vfmaq_f64(acc_im7, hr0, xi0);
                        acc_im7 = vfmsq_f64(acc_im7, hi0, xr0);
                        acc_re7 = vfmaq_f64(acc_re7, hr1, xr1);
                        acc_re7 = vfmaq_f64(acc_re7, hi1, xi1);
                        acc_im7 = vfmaq_f64(acc_im7, hr1, xi1);
                        acc_im7 = vfmsq_f64(acc_im7, hi1, xr1);
                        acc_re7 = vfmaq_f64(acc_re7, hr2, xr2);
                        acc_re7 = vfmaq_f64(acc_re7, hi2, xi2);
                        acc_im7 = vfmaq_f64(acc_im7, hr2, xi2);
                        acc_im7 = vfmsq_f64(acc_im7, hi2, xr2);
                        acc_re7 = vfmaq_f64(acc_re7, hr3, xr3);
                        acc_re7 = vfmaq_f64(acc_re7, hi3, xi3);
                        acc_im7 = vfmaq_f64(acc_im7, hr3, xi3);
                        acc_im7 = vfmsq_f64(acc_im7, hi3, xr3);

                        i += 8;
                    }

                    // 4 complex
                    while i + 4 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);

                        macro_rules! upd4 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f64($col.add(2 * (i + 0)));
                                let a1 = vld1q_f64($col.add(2 * (i + 1)));
                                let a2 = vld1q_f64($col.add(2 * (i + 2)));
                                let a3 = vld1q_f64($col.add(2 * (i + 3)));
                                let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                                let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                                $re = vfmaq_f64($re, ar0, xr0);
                                $re = vfmaq_f64($re, ai0, xi0);
                                $im = vfmaq_f64($im, ar0, xi0);
                                $im = vfmsq_f64($im, ai0, xr0);
                                $re = vfmaq_f64($re, ar1, xr1);
                                $re = vfmaq_f64($re, ai1, xi1);
                                $im = vfmaq_f64($im, ar1, xi1);
                                $im = vfmsq_f64($im, ai1, xr1);
                            }};
                        }

                        upd4!(col0, acc_re0, acc_im0);
                        upd4!(col1, acc_re1, acc_im1);
                        upd4!(col2, acc_re2, acc_im2);
                        upd4!(col3, acc_re3, acc_im3);
                        upd4!(col4, acc_re4, acc_im4);
                        upd4!(col5, acc_re5, acc_im5);
                        upd4!(col6, acc_re6, acc_im6);
                        upd4!(col7, acc_re7, acc_im7);

                        i += 4;
                    }

                    // 2 complex
                    while i + 2 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let xr = vuzp1q_f64(x0, x1);
                        let xi = vuzp2q_f64(x0, x1);

                        macro_rules! upd2 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f64($col.add(2 * (i + 0)));
                                let a1 = vld1q_f64($col.add(2 * (i + 1)));
                                let ar = vuzp1q_f64(a0, a1); let ai = vuzp2q_f64(a0, a1);
                                $re = vfmaq_f64($re, ar, xr);
                                $re = vfmaq_f64($re, ai, xi);
                                $im = vfmaq_f64($im, ar, xi);
                                $im = vfmsq_f64($im, ai, xr);
                            }};
                        }

                        upd2!(col0, acc_re0, acc_im0);
                        upd2!(col1, acc_re1, acc_im1);
                        upd2!(col2, acc_re2, acc_im2);
                        upd2!(col3, acc_re3, acc_im3);
                        upd2!(col4, acc_re4, acc_im4);
                        upd2!(col5, acc_re5, acc_im5);
                        upd2!(col6, acc_re6, acc_im6);
                        upd2!(col7, acc_re7, acc_im7);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f64(acc_re0); sum_im0 += vaddvq_f64(acc_im0);
                    sum_re1 += vaddvq_f64(acc_re1); sum_im1 += vaddvq_f64(acc_im1);
                    sum_re2 += vaddvq_f64(acc_re2); sum_im2 += vaddvq_f64(acc_im2);
                    sum_re3 += vaddvq_f64(acc_re3); sum_im3 += vaddvq_f64(acc_im3);
                    sum_re4 += vaddvq_f64(acc_re4); sum_im4 += vaddvq_f64(acc_im4);
                    sum_re5 += vaddvq_f64(acc_re5); sum_im5 += vaddvq_f64(acc_im5);
                    sum_re6 += vaddvq_f64(acc_re6); sum_im6 += vaddvq_f64(acc_im6);
                    sum_re7 += vaddvq_f64(acc_re7); sum_im7 += vaddvq_f64(acc_im7);

                    while i < mr {
                        let xr = *x_ptr.add(2 * (row_idx + i));
                        let xi = *x_ptr.add(2 * (row_idx + i) + 1);

                        let a_r = *col0.add(2 * i); let a_i = *col0.add(2 * i + 1);
                        let b_r = *col1.add(2 * i); let b_i = *col1.add(2 * i + 1);
                        let c_r = *col2.add(2 * i); let c_i = *col2.add(2 * i + 1);
                        let d_r = *col3.add(2 * i); let d_i = *col3.add(2 * i + 1);
                        let e_r = *col4.add(2 * i); let e_i = *col4.add(2 * i + 1);
                        let f_r = *col5.add(2 * i); let f_i = *col5.add(2 * i + 1);
                        let g_r = *col6.add(2 * i); let g_i = *col6.add(2 * i + 1);
                        let h_r = *col7.add(2 * i); let h_i = *col7.add(2 * i + 1);

                        sum_re0 += a_r * xr + a_i * xi; sum_im0 += a_r * xi - a_i * xr;
                        sum_re1 += b_r * xr + b_i * xi; sum_im1 += b_r * xi - b_i * xr;
                        sum_re2 += c_r * xr + c_i * xi; sum_im2 += c_r * xi - c_i * xr;
                        sum_re3 += d_r * xr + d_i * xi; sum_im3 += d_r * xi - d_i * xr;
                        sum_re4 += e_r * xr + e_i * xi; sum_im4 += e_r * xi - e_i * xr;
                        sum_re5 += f_r * xr + f_i * xi; sum_im5 += f_r * xi - f_i * xr;
                        sum_re6 += g_r * xr + g_i * xi; sum_im6 += g_r * xi - g_i * xr;
                        sum_re7 += h_r * xr + h_i * xi; sum_im7 += h_r * xi - h_i * xr;

                        i += 1;
                    }

                    row_idx += mr;
                }

                let po = out.as_mut_ptr().add(2 * col_idx);
                *po.add(0)  += sum_re0; *po.add(1)  += sum_im0;
                *po.add(2)  += sum_re1; *po.add(3)  += sum_im1;
                *po.add(4)  += sum_re2; *po.add(5)  += sum_im2;
                *po.add(6)  += sum_re3; *po.add(7)  += sum_im3;
                *po.add(8)  += sum_re4; *po.add(9)  += sum_im4;
                *po.add(10) += sum_re5; *po.add(11) += sum_im5;
                *po.add(12) += sum_re6; *po.add(13) += sum_im6;
                *po.add(14) += sum_re7; *po.add(15) += sum_im7;

                col_idx += NR;
            }

            // 4 columns
            if col_idx + 4 <= n_cols {
                let pa0 = a_ptr.add(2 * (col_idx * lda));
                let pa1 = a_ptr.add(2 * ((col_idx + 1) * lda));
                let pa2 = a_ptr.add(2 * ((col_idx + 2) * lda));
                let pa3 = a_ptr.add(2 * ((col_idx + 3) * lda));

                let mut sum_re0 = 0.0; let mut sum_im0 = 0.0;
                let mut sum_re1 = 0.0; let mut sum_im1 = 0.0;
                let mut sum_re2 = 0.0; let mut sum_im2 = 0.0;
                let mut sum_re3 = 0.0; let mut sum_im3 = 0.0;

                let mut row_idx = 0;
                while row_idx < n_rows {
                    let mr = core::cmp::min(row_block, n_rows - row_idx);

                    let mut acc_re0 = vdupq_n_f64(0.0); let mut acc_im0 = vdupq_n_f64(0.0);
                    let mut acc_re1 = vdupq_n_f64(0.0); let mut acc_im1 = vdupq_n_f64(0.0);
                    let mut acc_re2 = vdupq_n_f64(0.0); let mut acc_im2 = vdupq_n_f64(0.0);
                    let mut acc_re3 = vdupq_n_f64(0.0); let mut acc_im3 = vdupq_n_f64(0.0);

                    let col0 = pa0.add(2 * row_idx);
                    let col1 = pa1.add(2 * row_idx);
                    let col2 = pa2.add(2 * row_idx);
                    let col3 = pa3.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let x4 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 4)));
                        let x5 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 5)));
                        let x6 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 6)));
                        let x7 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 7)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);
                        let xr2 = vuzp1q_f64(x4, x5); let xi2 = vuzp2q_f64(x4, x5);
                        let xr3 = vuzp1q_f64(x6, x7); let xi3 = vuzp2q_f64(x6, x7);

                        macro_rules! upd8 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f64($col.add(2 * (i + 0)));
                                let a1 = vld1q_f64($col.add(2 * (i + 1)));
                                let a2 = vld1q_f64($col.add(2 * (i + 2)));
                                let a3 = vld1q_f64($col.add(2 * (i + 3)));
                                let a4 = vld1q_f64($col.add(2 * (i + 4)));
                                let a5 = vld1q_f64($col.add(2 * (i + 5)));
                                let a6 = vld1q_f64($col.add(2 * (i + 6)));
                                let a7 = vld1q_f64($col.add(2 * (i + 7)));
                                let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                                let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                                let ar2 = vuzp1q_f64(a4, a5); let ai2 = vuzp2q_f64(a4, a5);
                                let ar3 = vuzp1q_f64(a6, a7); let ai3 = vuzp2q_f64(a6, a7);
                                $re = vfmaq_f64($re, ar0, xr0);
                                $re = vfmaq_f64($re, ai0, xi0);
                                $im = vfmaq_f64($im, ar0, xi0);
                                $im = vfmsq_f64($im, ai0, xr0);
                                $re = vfmaq_f64($re, ar1, xr1);
                                $re = vfmaq_f64($re, ai1, xi1);
                                $im = vfmaq_f64($im, ar1, xi1);
                                $im = vfmsq_f64($im, ai1, xr1);
                                $re = vfmaq_f64($re, ar2, xr2);
                                $re = vfmaq_f64($re, ai2, xi2);
                                $im = vfmaq_f64($im, ar2, xi2);
                                $im = vfmsq_f64($im, ai2, xr2);
                                $re = vfmaq_f64($re, ar3, xr3);
                                $re = vfmaq_f64($re, ai3, xi3);
                                $im = vfmaq_f64($im, ar3, xi3);
                                $im = vfmsq_f64($im, ai3, xr3);
                            }};
                        }

                        upd8!(col0, acc_re0, acc_im0);
                        upd8!(col1, acc_re1, acc_im1);
                        upd8!(col2, acc_re2, acc_im2);
                        upd8!(col3, acc_re3, acc_im3);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);

                        macro_rules! upd4 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f64($col.add(2 * (i + 0)));
                                let a1 = vld1q_f64($col.add(2 * (i + 1)));
                                let a2 = vld1q_f64($col.add(2 * (i + 2)));
                                let a3 = vld1q_f64($col.add(2 * (i + 3)));
                                let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                                let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                                $re = vfmaq_f64($re, ar0, xr0);
                                $re = vfmaq_f64($re, ai0, xi0);
                                $im = vfmaq_f64($im, ar0, xi0);
                                $im = vfmsq_f64($im, ai0, xr0);
                                $re = vfmaq_f64($re, ar1, xr1);
                                $re = vfmaq_f64($re, ai1, xi1);
                                $im = vfmaq_f64($im, ar1, xi1);
                                $im = vfmsq_f64($im, ai1, xr1);
                            }};
                        }

                        upd4!(col0, acc_re0, acc_im0);
                        upd4!(col1, acc_re1, acc_im1);
                        upd4!(col2, acc_re2, acc_im2);
                        upd4!(col3, acc_re3, acc_im3);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let xr = vuzp1q_f64(x0, x1);
                        let xi = vuzp2q_f64(x0, x1);

                        macro_rules! upd2 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f64($col.add(2 * (i + 0)));
                                let a1 = vld1q_f64($col.add(2 * (i + 1)));
                                let ar = vuzp1q_f64(a0, a1); let ai = vuzp2q_f64(a0, a1);
                                $re = vfmaq_f64($re, ar, xr);
                                $re = vfmaq_f64($re, ai, xi);
                                $im = vfmaq_f64($im, ar, xi);
                                $im = vfmsq_f64($im, ai, xr);
                            }};
                        }

                        upd2!(col0, acc_re0, acc_im0);
                        upd2!(col1, acc_re1, acc_im1);
                        upd2!(col2, acc_re2, acc_im2);
                        upd2!(col3, acc_re3, acc_im3);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f64(acc_re0); sum_im0 += vaddvq_f64(acc_im0);
                    sum_re1 += vaddvq_f64(acc_re1); sum_im1 += vaddvq_f64(acc_im1);
                    sum_re2 += vaddvq_f64(acc_re2); sum_im2 += vaddvq_f64(acc_im2);
                    sum_re3 += vaddvq_f64(acc_re3); sum_im3 += vaddvq_f64(acc_im3);

                    while i < mr {
                        let xr = *x_ptr.add(2 * (row_idx + i));
                        let xi = *x_ptr.add(2 * (row_idx + i) + 1);

                        let a_r = *col0.add(2 * i); let a_i = *col0.add(2 * i + 1);
                        let b_r = *col1.add(2 * i); let b_i = *col1.add(2 * i + 1);
                        let c_r = *col2.add(2 * i); let c_i = *col2.add(2 * i + 1);
                        let d_r = *col3.add(2 * i); let d_i = *col3.add(2 * i + 1);

                        sum_re0 += a_r * xr + a_i * xi; sum_im0 += a_r * xi - a_i * xr;
                        sum_re1 += b_r * xr + b_i * xi; sum_im1 += b_r * xi - b_i * xr;
                        sum_re2 += c_r * xr + c_i * xi; sum_im2 += c_r * xi - c_i * xr;
                        sum_re3 += d_r * xr + d_i * xi; sum_im3 += d_r * xi - d_i * xr;

                        i += 1;
                    }

                    row_idx += mr;
                }

                let po = out.as_mut_ptr().add(2 * col_idx);
                *po.add(0) += sum_re0; *po.add(1) += sum_im0;
                *po.add(2) += sum_re1; *po.add(3) += sum_im1;
                *po.add(4) += sum_re2; *po.add(5) += sum_im2;
                *po.add(6) += sum_re3; *po.add(7) += sum_im3;

                col_idx += 4;
            }

            // 2 columns
            if col_idx + 2 <= n_cols {
                let pa0 = a_ptr.add(2 * (col_idx * lda));
                let pa1 = a_ptr.add(2 * ((col_idx + 1) * lda));

                let mut sum_re0 = 0.0; let mut sum_im0 = 0.0;
                let mut sum_re1 = 0.0; let mut sum_im1 = 0.0;

                let mut row_idx = 0;
                while row_idx < n_rows {
                    let mr = core::cmp::min(row_block, n_rows - row_idx);

                    let mut acc_re0 = vdupq_n_f64(0.0); let mut acc_im0 = vdupq_n_f64(0.0);
                    let mut acc_re1 = vdupq_n_f64(0.0); let mut acc_im1 = vdupq_n_f64(0.0);

                    let col0 = pa0.add(2 * row_idx);
                    let col1 = pa1.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let x4 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 4)));
                        let x5 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 5)));
                        let x6 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 6)));
                        let x7 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 7)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);
                        let xr2 = vuzp1q_f64(x4, x5); let xi2 = vuzp2q_f64(x4, x5);
                        let xr3 = vuzp1q_f64(x6, x7); let xi3 = vuzp2q_f64(x6, x7);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let a2 = vld1q_f64(col0.add(2 * (i + 2)));
                        let a3 = vld1q_f64(col0.add(2 * (i + 3)));
                        let a4 = vld1q_f64(col0.add(2 * (i + 4)));
                        let a5 = vld1q_f64(col0.add(2 * (i + 5)));
                        let a6 = vld1q_f64(col0.add(2 * (i + 6)));
                        let a7 = vld1q_f64(col0.add(2 * (i + 7)));
                        let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                        let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                        let ar2 = vuzp1q_f64(a4, a5); let ai2 = vuzp2q_f64(a4, a5);
                        let ar3 = vuzp1q_f64(a6, a7); let ai3 = vuzp2q_f64(a6, a7);
                        acc_re0 = vfmaq_f64(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f64(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f64(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f64(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f64(acc_im0, ai1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ar2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ai2, xi2);
                        acc_im0 = vfmaq_f64(acc_im0, ar2, xi2);
                        acc_im0 = vfmsq_f64(acc_im0, ai2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ar3, xr3);
                        acc_re0 = vfmaq_f64(acc_re0, ai3, xi3);
                        acc_im0 = vfmaq_f64(acc_im0, ar3, xi3);
                        acc_im0 = vfmsq_f64(acc_im0, ai3, xr3);

                        let b0 = vld1q_f64(col1.add(2 * (i + 0)));
                        let b1 = vld1q_f64(col1.add(2 * (i + 1)));
                        let b2 = vld1q_f64(col1.add(2 * (i + 2)));
                        let b3 = vld1q_f64(col1.add(2 * (i + 3)));
                        let b4 = vld1q_f64(col1.add(2 * (i + 4)));
                        let b5 = vld1q_f64(col1.add(2 * (i + 5)));
                        let b6 = vld1q_f64(col1.add(2 * (i + 6)));
                        let b7 = vld1q_f64(col1.add(2 * (i + 7)));
                        let br0 = vuzp1q_f64(b0, b1); let bi0 = vuzp2q_f64(b0, b1);
                        let br1 = vuzp1q_f64(b2, b3); let bi1 = vuzp2q_f64(b2, b3);
                        let br2 = vuzp1q_f64(b4, b5); let bi2 = vuzp2q_f64(b4, b5);
                        let br3 = vuzp1q_f64(b6, b7); let bi3 = vuzp2q_f64(b6, b7);
                        acc_re1 = vfmaq_f64(acc_re1, br0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, bi0, xi0);
                        acc_im1 = vfmaq_f64(acc_im1, br0, xi0);
                        acc_im1 = vfmsq_f64(acc_im1, bi0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, br1, xr1);
                        acc_re1 = vfmaq_f64(acc_re1, bi1, xi1);
                        acc_im1 = vfmaq_f64(acc_im1, br1, xi1);
                        acc_im1 = vfmsq_f64(acc_im1, bi1, xr1);
                        acc_re1 = vfmaq_f64(acc_re1, br2, xr2);
                        acc_re1 = vfmaq_f64(acc_re1, bi2, xi2);
                        acc_im1 = vfmaq_f64(acc_im1, br2, xi2);
                        acc_im1 = vfmsq_f64(acc_im1, bi2, xr2);
                        acc_re1 = vfmaq_f64(acc_re1, br3, xr3);
                        acc_re1 = vfmaq_f64(acc_re1, bi3, xi3);
                        acc_im1 = vfmaq_f64(acc_im1, br3, xi3);
                        acc_im1 = vfmsq_f64(acc_im1, bi3, xr3);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let a2 = vld1q_f64(col0.add(2 * (i + 2)));
                        let a3 = vld1q_f64(col0.add(2 * (i + 3)));
                        let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                        let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                        acc_re0 = vfmaq_f64(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f64(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f64(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f64(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f64(acc_im0, ai1, xr1);

                        let b0 = vld1q_f64(col1.add(2 * (i + 0)));
                        let b1 = vld1q_f64(col1.add(2 * (i + 1)));
                        let b2 = vld1q_f64(col1.add(2 * (i + 2)));
                        let b3 = vld1q_f64(col1.add(2 * (i + 3)));
                        let br0 = vuzp1q_f64(b0, b1); let bi0 = vuzp2q_f64(b0, b1);
                        let br1 = vuzp1q_f64(b2, b3); let bi1 = vuzp2q_f64(b2, b3);
                        acc_re1 = vfmaq_f64(acc_re1, br0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, bi0, xi0);
                        acc_im1 = vfmaq_f64(acc_im1, br0, xi0);
                        acc_im1 = vfmsq_f64(acc_im1, bi0, xr0);
                        acc_re1 = vfmaq_f64(acc_re1, br1, xr1);
                        acc_re1 = vfmaq_f64(acc_re1, bi1, xi1);
                        acc_im1 = vfmaq_f64(acc_im1, br1, xi1);
                        acc_im1 = vfmsq_f64(acc_im1, bi1, xr1);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let xr = vuzp1q_f64(x0, x1);
                        let xi = vuzp2q_f64(x0, x1);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let ar = vuzp1q_f64(a0, a1); let ai = vuzp2q_f64(a0, a1);
                        acc_re0 = vfmaq_f64(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f64(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f64(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f64(acc_im0, ai, xr);

                        let b0 = vld1q_f64(col1.add(2 * (i + 0)));
                        let b1 = vld1q_f64(col1.add(2 * (i + 1)));
                        let br = vuzp1q_f64(b0, b1); let bi = vuzp2q_f64(b0, b1);
                        acc_re1 = vfmaq_f64(acc_re1, br, xr);
                        acc_re1 = vfmaq_f64(acc_re1, bi, xi);
                        acc_im1 = vfmaq_f64(acc_im1, br, xi);
                        acc_im1 = vfmsq_f64(acc_im1, bi, xr);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f64(acc_re0); sum_im0 += vaddvq_f64(acc_im0);
                    sum_re1 += vaddvq_f64(acc_re1); sum_im1 += vaddvq_f64(acc_im1);

                    while i < mr {
                        let xr = *x_ptr.add(2 * (row_idx + i));
                        let xi = *x_ptr.add(2 * (row_idx + i) + 1);

                        let a_r = *col0.add(2 * i); let a_i = *col0.add(2 * i + 1);
                        let b_r = *col1.add(2 * i); let b_i = *col1.add(2 * i + 1);

                        sum_re0 += a_r * xr + a_i * xi; sum_im0 += a_r * xi - a_i * xr;
                        sum_re1 += b_r * xr + b_i * xi; sum_im1 += b_r * xi - b_i * xr;

                        i += 1;
                    }

                    row_idx += mr;
                }

                let po = out.as_mut_ptr().add(2 * col_idx);
                *po.add(0) += sum_re0; *po.add(1) += sum_im0;
                *po.add(2) += sum_re1; *po.add(3) += sum_im1;

                col_idx += 2;
            }

            // 1 column
            if col_idx < n_cols {
                let pa0 = a_ptr.add(2 * (col_idx * lda));

                let mut sum_re0 = 0.0; let mut sum_im0 = 0.0;

                let mut row_idx = 0;
                while row_idx < n_rows {
                    let mr = core::cmp::min(row_block, n_rows - row_idx);

                    let mut acc_re0 = vdupq_n_f64(0.0);
                    let mut acc_im0 = vdupq_n_f64(0.0);

                    let col0 = pa0.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let x4 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 4)));
                        let x5 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 5)));
                        let x6 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 6)));
                        let x7 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 7)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);
                        let xr2 = vuzp1q_f64(x4, x5); let xi2 = vuzp2q_f64(x4, x5);
                        let xr3 = vuzp1q_f64(x6, x7); let xi3 = vuzp2q_f64(x6, x7);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let a2 = vld1q_f64(col0.add(2 * (i + 2)));
                        let a3 = vld1q_f64(col0.add(2 * (i + 3)));
                        let a4 = vld1q_f64(col0.add(2 * (i + 4)));
                        let a5 = vld1q_f64(col0.add(2 * (i + 5)));
                        let a6 = vld1q_f64(col0.add(2 * (i + 6)));
                        let a7 = vld1q_f64(col0.add(2 * (i + 7)));
                        let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                        let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);
                        let ar2 = vuzp1q_f64(a4, a5); let ai2 = vuzp2q_f64(a4, a5);
                        let ar3 = vuzp1q_f64(a6, a7); let ai3 = vuzp2q_f64(a6, a7);

                        acc_re0 = vfmaq_f64(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f64(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f64(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f64(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f64(acc_im0, ai1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ar2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ai2, xi2);
                        acc_im0 = vfmaq_f64(acc_im0, ar2, xi2);
                        acc_im0 = vfmsq_f64(acc_im0, ai2, xr2);
                        acc_re0 = vfmaq_f64(acc_re0, ar3, xr3);
                        acc_re0 = vfmaq_f64(acc_re0, ai3, xi3);
                        acc_im0 = vfmaq_f64(acc_im0, ar3, xi3);
                        acc_im0 = vfmsq_f64(acc_im0, ai3, xr3);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let x2 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 2)));
                        let x3 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 3)));
                        let xr0 = vuzp1q_f64(x0, x1); let xi0 = vuzp2q_f64(x0, x1);
                        let xr1 = vuzp1q_f64(x2, x3); let xi1 = vuzp2q_f64(x2, x3);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let a2 = vld1q_f64(col0.add(2 * (i + 2)));
                        let a3 = vld1q_f64(col0.add(2 * (i + 3)));
                        let ar0 = vuzp1q_f64(a0, a1); let ai0 = vuzp2q_f64(a0, a1);
                        let ar1 = vuzp1q_f64(a2, a3); let ai1 = vuzp2q_f64(a2, a3);

                        acc_re0 = vfmaq_f64(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f64(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f64(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f64(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f64(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f64(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f64(acc_im0, ai1, xr1);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 0)));
                        let x1 = vld1q_f64(x_ptr.add(2 * (row_idx + i + 1)));
                        let xr = vuzp1q_f64(x0, x1);
                        let xi = vuzp2q_f64(x0, x1);

                        let a0 = vld1q_f64(col0.add(2 * (i + 0)));
                        let a1 = vld1q_f64(col0.add(2 * (i + 1)));
                        let ar = vuzp1q_f64(a0, a1); let ai = vuzp2q_f64(a0, a1);

                        acc_re0 = vfmaq_f64(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f64(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f64(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f64(acc_im0, ai, xr);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f64(acc_re0);
                    sum_im0 += vaddvq_f64(acc_im0);

                    while i < mr {
                        let xr = *x_ptr.add(2 * (row_idx + i));
                        let xi = *x_ptr.add(2 * (row_idx + i) + 1);

                        let a_r = *col0.add(2 * i);
                        let a_i = *col0.add(2 * i + 1);

                        sum_re0 += a_r * xr + a_i * xi;
                        sum_im0 += a_r * xi - a_i * xr;

                        i += 1;
                    }

                    row_idx += mr;
                }

                let po = out.as_mut_ptr().add(2 * col_idx);
                *po.add(0) += sum_re0; *po.add(1) += sum_im0;

                return;
            }

            return;
        }
    }

    for col_idx in 0..n_cols {
        unsafe {
            let col_ptr = matrix.as_ptr().add(2 * (col_idx * lda));
            let col = core::slice::from_raw_parts(col_ptr, 2 * n_rows);

            let s = zdotc(n_rows, col, 1, x, incx);

            let po = out.as_mut_ptr().add(2 * col_idx);
            *po.add(0) += s[0];
            *po.add(1) += s[1];
        }
    }
}

