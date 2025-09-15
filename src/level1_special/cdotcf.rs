//! Computes fused column dots (conjugated): out := out + conj(A)^T x
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of complex rows (m).
//! - `n_cols` (usize)      : Number of columns (n).
//! - `matrix` (&[f32])     : Column-major A (interleaved) with dims (`lda` x `n_cols`).
//! - `lda`    (usize)      : Leading dimension in complex elements (>= `n_rows`).
//! - `x`      (&[f32])     : Complex vector of length `n_rows` (interleaved) with stride `incx`.
//! - `incx`   (usize)      : Stride for `x` in complex elements.
//! - `out`    (&mut [f32]) : Complex output (interleaved) of length `n_cols`, accumulated in place.
//!
//! # Notes
//! - Fast path when `incx == 1` uses NEON + blocking (NR=8, MC=128).
//! - Otherwise falls back to level-1 `cdotc` per column.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vld1q_f32,
    vdupq_n_f32,
    vfmaq_f32,
    vfmsq_f32,
    vaddvq_f32,
    vuzp1q_f32,
    vuzp2q_f32,
};

use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;
use crate::level1::cdotc::cdotc;

const MC: usize = 128;
const NR: usize = 8;

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn cdotcf(
    n_rows : usize,
    n_cols : usize,
    matrix : &[f32],
    lda    : usize,
    x      : &[f32],
    incx   : usize,
    out    : &mut [f32],
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

                    let mut acc_re0 = vdupq_n_f32(0.0); let mut acc_im0 = vdupq_n_f32(0.0);
                    let mut acc_re1 = vdupq_n_f32(0.0); let mut acc_im1 = vdupq_n_f32(0.0);
                    let mut acc_re2 = vdupq_n_f32(0.0); let mut acc_im2 = vdupq_n_f32(0.0);
                    let mut acc_re3 = vdupq_n_f32(0.0); let mut acc_im3 = vdupq_n_f32(0.0);
                    let mut acc_re4 = vdupq_n_f32(0.0); let mut acc_im4 = vdupq_n_f32(0.0);
                    let mut acc_re5 = vdupq_n_f32(0.0); let mut acc_im5 = vdupq_n_f32(0.0);
                    let mut acc_re6 = vdupq_n_f32(0.0); let mut acc_im6 = vdupq_n_f32(0.0);
                    let mut acc_re7 = vdupq_n_f32(0.0); let mut acc_im7 = vdupq_n_f32(0.0);

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
                        let x0  = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let x2  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 4)));
                        let x3  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 6)));
                        let xr0 = vuzp1q_f32(x0, x1); let xi0 = vuzp2q_f32(x0, x1);
                        let xr1 = vuzp1q_f32(x2, x3); let xi1 = vuzp2q_f32(x2, x3);

                        // col 0
                        let a0  = vld1q_f32(col0.add(2 * i));
                        let a1  = vld1q_f32(col0.add(2 * (i + 2)));
                        let a2  = vld1q_f32(col0.add(2 * (i + 4)));
                        let a3  = vld1q_f32(col0.add(2 * (i + 6)));
                        let ar0 = vuzp1q_f32(a0, a1); let ai0 = vuzp2q_f32(a0, a1);
                        let ar1 = vuzp1q_f32(a2, a3); let ai1 = vuzp2q_f32(a2, a3);
                        acc_re0 = vfmaq_f32(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f32(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f32(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f32(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f32(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f32(acc_im0, ai1, xr1);

                        // col 1
                        let b0  = vld1q_f32(col1.add(2 * i));
                        let b1  = vld1q_f32(col1.add(2 * (i + 2)));
                        let b2  = vld1q_f32(col1.add(2 * (i + 4)));
                        let b3  = vld1q_f32(col1.add(2 * (i + 6)));
                        let br0 = vuzp1q_f32(b0, b1); let bi0 = vuzp2q_f32(b0, b1);
                        let br1 = vuzp1q_f32(b2, b3); let bi1 = vuzp2q_f32(b2, b3);
                        acc_re1 = vfmaq_f32(acc_re1, br0, xr0);
                        acc_re1 = vfmaq_f32(acc_re1, bi0, xi0);
                        acc_im1 = vfmaq_f32(acc_im1, br0, xi0);
                        acc_im1 = vfmsq_f32(acc_im1, bi0, xr0);
                        acc_re1 = vfmaq_f32(acc_re1, br1, xr1);
                        acc_re1 = vfmaq_f32(acc_re1, bi1, xi1);
                        acc_im1 = vfmaq_f32(acc_im1, br1, xi1);
                        acc_im1 = vfmsq_f32(acc_im1, bi1, xr1);

                        // col 2
                        let c0  = vld1q_f32(col2.add(2 * i));
                        let c1  = vld1q_f32(col2.add(2 * (i + 2)));
                        let c2  = vld1q_f32(col2.add(2 * (i + 4)));
                        let c3  = vld1q_f32(col2.add(2 * (i + 6)));
                        let cr0 = vuzp1q_f32(c0, c1); let ci0 = vuzp2q_f32(c0, c1);
                        let cr1 = vuzp1q_f32(c2, c3); let ci1 = vuzp2q_f32(c2, c3);
                        acc_re2 = vfmaq_f32(acc_re2, cr0, xr0);
                        acc_re2 = vfmaq_f32(acc_re2, ci0, xi0);
                        acc_im2 = vfmaq_f32(acc_im2, cr0, xi0);
                        acc_im2 = vfmsq_f32(acc_im2, ci0, xr0);
                        acc_re2 = vfmaq_f32(acc_re2, cr1, xr1);
                        acc_re2 = vfmaq_f32(acc_re2, ci1, xi1);
                        acc_im2 = vfmaq_f32(acc_im2, cr1, xi1);
                        acc_im2 = vfmsq_f32(acc_im2, ci1, xr1);

                        // col 3
                        let d0  = vld1q_f32(col3.add(2 * i));
                        let d1  = vld1q_f32(col3.add(2 * (i + 2)));
                        let d2  = vld1q_f32(col3.add(2 * (i + 4)));
                        let d3  = vld1q_f32(col3.add(2 * (i + 6)));
                        let dr0 = vuzp1q_f32(d0, d1); let di0 = vuzp2q_f32(d0, d1);
                        let dr1 = vuzp1q_f32(d2, d3); let di1 = vuzp2q_f32(d2, d3);
                        acc_re3 = vfmaq_f32(acc_re3, dr0, xr0);
                        acc_re3 = vfmaq_f32(acc_re3, di0, xi0);
                        acc_im3 = vfmaq_f32(acc_im3, dr0, xi0);
                        acc_im3 = vfmsq_f32(acc_im3, di0, xr0);
                        acc_re3 = vfmaq_f32(acc_re3, dr1, xr1);
                        acc_re3 = vfmaq_f32(acc_re3, di1, xi1);
                        acc_im3 = vfmaq_f32(acc_im3, dr1, xi1);
                        acc_im3 = vfmsq_f32(acc_im3, di1, xr1);

                        // col 4
                        let e0  = vld1q_f32(col4.add(2 * i));
                        let e1  = vld1q_f32(col4.add(2 * (i + 2)));
                        let e2  = vld1q_f32(col4.add(2 * (i + 4)));
                        let e3  = vld1q_f32(col4.add(2 * (i + 6)));
                        let er0 = vuzp1q_f32(e0, e1); let ei0 = vuzp2q_f32(e0, e1);
                        let er1 = vuzp1q_f32(e2, e3); let ei1 = vuzp2q_f32(e2, e3);
                        acc_re4 = vfmaq_f32(acc_re4, er0, xr0);
                        acc_re4 = vfmaq_f32(acc_re4, ei0, xi0);
                        acc_im4 = vfmaq_f32(acc_im4, er0, xi0);
                        acc_im4 = vfmsq_f32(acc_im4, ei0, xr0);
                        acc_re4 = vfmaq_f32(acc_re4, er1, xr1);
                        acc_re4 = vfmaq_f32(acc_re4, ei1, xi1);
                        acc_im4 = vfmaq_f32(acc_im4, er1, xi1);
                        acc_im4 = vfmsq_f32(acc_im4, ei1, xr1);

                        // col 5
                        let f0  = vld1q_f32(col5.add(2 * i));
                        let f1  = vld1q_f32(col5.add(2 * (i + 2)));
                        let f2  = vld1q_f32(col5.add(2 * (i + 4)));
                        let f3  = vld1q_f32(col5.add(2 * (i + 6)));
                        let fr0 = vuzp1q_f32(f0, f1); let fi0 = vuzp2q_f32(f0, f1);
                        let fr1 = vuzp1q_f32(f2, f3); let fi1 = vuzp2q_f32(f2, f3);
                        acc_re5 = vfmaq_f32(acc_re5, fr0, xr0);
                        acc_re5 = vfmaq_f32(acc_re5, fi0, xi0);
                        acc_im5 = vfmaq_f32(acc_im5, fr0, xi0);
                        acc_im5 = vfmsq_f32(acc_im5, fi0, xr0);
                        acc_re5 = vfmaq_f32(acc_re5, fr1, xr1);
                        acc_re5 = vfmaq_f32(acc_re5, fi1, xi1);
                        acc_im5 = vfmaq_f32(acc_im5, fr1, xi1);
                        acc_im5 = vfmsq_f32(acc_im5, fi1, xr1);

                        // col 6
                        let g0  = vld1q_f32(col6.add(2 * i));
                        let g1  = vld1q_f32(col6.add(2 * (i + 2)));
                        let g2  = vld1q_f32(col6.add(2 * (i + 4)));
                        let g3  = vld1q_f32(col6.add(2 * (i + 6)));
                        let gr0 = vuzp1q_f32(g0, g1); let gi0 = vuzp2q_f32(g0, g1);
                        let gr1 = vuzp1q_f32(g2, g3); let gi1 = vuzp2q_f32(g2, g3);
                        acc_re6 = vfmaq_f32(acc_re6, gr0, xr0);
                        acc_re6 = vfmaq_f32(acc_re6, gi0, xi0);
                        acc_im6 = vfmaq_f32(acc_im6, gr0, xi0);
                        acc_im6 = vfmsq_f32(acc_im6, gi0, xr0);
                        acc_re6 = vfmaq_f32(acc_re6, gr1, xr1);
                        acc_re6 = vfmaq_f32(acc_re6, gi1, xi1);
                        acc_im6 = vfmaq_f32(acc_im6, gr1, xi1);
                        acc_im6 = vfmsq_f32(acc_im6, gi1, xr1);

                        // col 7
                        let h0  = vld1q_f32(col7.add(2 * i));
                        let h1  = vld1q_f32(col7.add(2 * (i + 2)));
                        let h2  = vld1q_f32(col7.add(2 * (i + 4)));
                        let h3  = vld1q_f32(col7.add(2 * (i + 6)));
                        let hr0 = vuzp1q_f32(h0, h1); let hi0 = vuzp2q_f32(h0, h1);
                        let hr1 = vuzp1q_f32(h2, h3); let hi1 = vuzp2q_f32(h2, h3);
                        acc_re7 = vfmaq_f32(acc_re7, hr0, xr0);
                        acc_re7 = vfmaq_f32(acc_re7, hi0, xi0);
                        acc_im7 = vfmaq_f32(acc_im7, hr0, xi0);
                        acc_im7 = vfmsq_f32(acc_im7, hi0, xr0);
                        acc_re7 = vfmaq_f32(acc_re7, hr1, xr1);
                        acc_re7 = vfmaq_f32(acc_re7, hi1, xi1);
                        acc_im7 = vfmaq_f32(acc_im7, hr1, xi1);
                        acc_im7 = vfmsq_f32(acc_im7, hi1, xr1);

                        i += 8;
                    }

                    // 4 complex
                    while i + 4 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1 = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let xr = vuzp1q_f32(x0, x1); let xi = vuzp2q_f32(x0, x1);

                        macro_rules! upd4 {
                            ($col:ident, $ar:ident, $ai:ident, $re:ident, $im:ident) => {{
                                let a0  = vld1q_f32($col.add(2 * i));
                                let a1  = vld1q_f32($col.add(2 * (i + 2)));
                                let $ar = vuzp1q_f32(a0, a1);
                                let $ai = vuzp2q_f32(a0, a1);
                                $re = vfmaq_f32($re, $ar, xr);
                                $re = vfmaq_f32($re, $ai, xi);
                                $im = vfmaq_f32($im, $ar, xi);
                                $im = vfmsq_f32($im, $ai, xr);
                            }};
                        }

                        upd4!(col0, ar, ai, acc_re0, acc_im0);
                        upd4!(col1, br, bi, acc_re1, acc_im1);
                        upd4!(col2, cr, ci, acc_re2, acc_im2);
                        upd4!(col3, dr, di, acc_re3, acc_im3);
                        upd4!(col4, er, ei, acc_re4, acc_im4);
                        upd4!(col5, fr, fi, acc_re5, acc_im5);
                        upd4!(col6, gr, gi, acc_re6, acc_im6);
                        upd4!(col7, hr, hi, acc_re7, acc_im7);

                        i += 4;
                    }

                    // 2 complex
                    while i + 2 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let xr = vuzp1q_f32(x0, x0);
                        let xi = vuzp2q_f32(x0, x0);

                        macro_rules! upd2 {
                            ($col:ident, $ar:ident, $ai:ident, $re:ident, $im:ident) => {{
                                let a0  = vld1q_f32($col.add(2 * i));
                                let $ar = vuzp1q_f32(a0, a0);
                                let $ai = vuzp2q_f32(a0, a0);
                                $re = vfmaq_f32($re, $ar, xr);
                                $re = vfmaq_f32($re, $ai, xi);
                                $im = vfmaq_f32($im, $ar, xi);
                                $im = vfmsq_f32($im, $ai, xr);
                            }};
                        }

                        upd2!(col0, ar, ai, acc_re0, acc_im0);
                        upd2!(col1, br, bi, acc_re1, acc_im1);
                        upd2!(col2, cr, ci, acc_re2, acc_im2);
                        upd2!(col3, dr, di, acc_re3, acc_im3);
                        upd2!(col4, er, ei, acc_re4, acc_im4);
                        upd2!(col5, fr, fi, acc_re5, acc_im5);
                        upd2!(col6, gr, gi, acc_re6, acc_im6);
                        upd2!(col7, hr, hi, acc_re7, acc_im7);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f32(acc_re0); sum_im0 += vaddvq_f32(acc_im0);
                    sum_re1 += vaddvq_f32(acc_re1); sum_im1 += vaddvq_f32(acc_im1);
                    sum_re2 += vaddvq_f32(acc_re2); sum_im2 += vaddvq_f32(acc_im2);
                    sum_re3 += vaddvq_f32(acc_re3); sum_im3 += vaddvq_f32(acc_im3);
                    sum_re4 += vaddvq_f32(acc_re4); sum_im4 += vaddvq_f32(acc_im4);
                    sum_re5 += vaddvq_f32(acc_re5); sum_im5 += vaddvq_f32(acc_im5);
                    sum_re6 += vaddvq_f32(acc_re6); sum_im6 += vaddvq_f32(acc_im6);
                    sum_re7 += vaddvq_f32(acc_re7); sum_im7 += vaddvq_f32(acc_im7);

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

                    let mut acc_re0 = vdupq_n_f32(0.0); let mut acc_im0 = vdupq_n_f32(0.0);
                    let mut acc_re1 = vdupq_n_f32(0.0); let mut acc_im1 = vdupq_n_f32(0.0);
                    let mut acc_re2 = vdupq_n_f32(0.0); let mut acc_im2 = vdupq_n_f32(0.0);
                    let mut acc_re3 = vdupq_n_f32(0.0); let mut acc_im3 = vdupq_n_f32(0.0);

                    let col0 = pa0.add(2 * row_idx);
                    let col1 = pa1.add(2 * row_idx);
                    let col2 = pa2.add(2 * row_idx);
                    let col3 = pa3.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0  = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let x2  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 4)));
                        let x3  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 6)));
                        let xr0 = vuzp1q_f32(x0, x1); let xi0 = vuzp2q_f32(x0, x1);
                        let xr1 = vuzp1q_f32(x2, x3); let xi1 = vuzp2q_f32(x2, x3);

                        macro_rules! upd8 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0  = vld1q_f32($col.add(2 * i));
                                let a1  = vld1q_f32($col.add(2 * (i + 2)));
                                let a2  = vld1q_f32($col.add(2 * (i + 4)));
                                let a3  = vld1q_f32($col.add(2 * (i + 6)));
                                let ar0 = vuzp1q_f32(a0, a1); let ai0 = vuzp2q_f32(a0, a1);
                                let ar1 = vuzp1q_f32(a2, a3); let ai1 = vuzp2q_f32(a2, a3);
                                $re = vfmaq_f32($re, ar0, xr0);
                                $re = vfmaq_f32($re, ai0, xi0);
                                $im = vfmaq_f32($im, ar0, xi0);
                                $im = vfmsq_f32($im, ai0, xr0);
                                $re = vfmaq_f32($re, ar1, xr1);
                                $re = vfmaq_f32($re, ai1, xi1);
                                $im = vfmaq_f32($im, ar1, xi1);
                                $im = vfmsq_f32($im, ai1, xr1);
                            }};
                        }

                        upd8!(col0, acc_re0, acc_im0);
                        upd8!(col1, acc_re1, acc_im1);
                        upd8!(col2, acc_re2, acc_im2);
                        upd8!(col3, acc_re3, acc_im3);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1 = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let xr = vuzp1q_f32(x0, x1); let xi = vuzp2q_f32(x0, x1);

                        macro_rules! upd4 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f32($col.add(2 * i));
                                let a1 = vld1q_f32($col.add(2 * (i + 2)));
                                let ar = vuzp1q_f32(a0, a1); let ai = vuzp2q_f32(a0, a1);
                                $re = vfmaq_f32($re, ar, xr);
                                $re = vfmaq_f32($re, ai, xi);
                                $im = vfmaq_f32($im, ar, xi);
                                $im = vfmsq_f32($im, ai, xr);
                            }};
                        }

                        upd4!(col0, acc_re0, acc_im0);
                        upd4!(col1, acc_re1, acc_im1);
                        upd4!(col2, acc_re2, acc_im2);
                        upd4!(col3, acc_re3, acc_im3);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let xr = vuzp1q_f32(x0, x0);
                        let xi = vuzp2q_f32(x0, x0);

                        macro_rules! upd2 {
                            ($col:ident, $re:ident, $im:ident) => {{
                                let a0 = vld1q_f32($col.add(2 * i));
                                let ar = vuzp1q_f32(a0, a0); let ai = vuzp2q_f32(a0, a0);
                                $re = vfmaq_f32($re, ar, xr);
                                $re = vfmaq_f32($re, ai, xi);
                                $im = vfmaq_f32($im, ar, xi);
                                $im = vfmsq_f32($im, ai, xr);
                            }};
                        }

                        upd2!(col0, acc_re0, acc_im0);
                        upd2!(col1, acc_re1, acc_im1);
                        upd2!(col2, acc_re2, acc_im2);
                        upd2!(col3, acc_re3, acc_im3);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f32(acc_re0); sum_im0 += vaddvq_f32(acc_im0);
                    sum_re1 += vaddvq_f32(acc_re1); sum_im1 += vaddvq_f32(acc_im1);
                    sum_re2 += vaddvq_f32(acc_re2); sum_im2 += vaddvq_f32(acc_im2);
                    sum_re3 += vaddvq_f32(acc_re3); sum_im3 += vaddvq_f32(acc_im3);

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

                    let mut acc_re0 = vdupq_n_f32(0.0); let mut acc_im0 = vdupq_n_f32(0.0);
                    let mut acc_re1 = vdupq_n_f32(0.0); let mut acc_im1 = vdupq_n_f32(0.0);

                    let col0 = pa0.add(2 * row_idx);
                    let col1 = pa1.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0  = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let x2  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 4)));
                        let x3  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 6)));
                        let xr0 = vuzp1q_f32(x0, x1); let xi0 = vuzp2q_f32(x0, x1);
                        let xr1 = vuzp1q_f32(x2, x3); let xi1 = vuzp2q_f32(x2, x3);

                        let a0  = vld1q_f32(col0.add(2 * i));
                        let a1  = vld1q_f32(col0.add(2 * (i + 2)));
                        let a2  = vld1q_f32(col0.add(2 * (i + 4)));
                        let a3  = vld1q_f32(col0.add(2 * (i + 6)));
                        let ar0 = vuzp1q_f32(a0, a1); let ai0 = vuzp2q_f32(a0, a1);
                        let ar1 = vuzp1q_f32(a2, a3); let ai1 = vuzp2q_f32(a2, a3);
                        acc_re0 = vfmaq_f32(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f32(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f32(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f32(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f32(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f32(acc_im0, ai1, xr1);

                        let b0  = vld1q_f32(col1.add(2 * i));
                        let b1  = vld1q_f32(col1.add(2 * (i + 2)));
                        let b2  = vld1q_f32(col1.add(2 * (i + 4)));
                        let b3  = vld1q_f32(col1.add(2 * (i + 6)));
                        let br0 = vuzp1q_f32(b0, b1); let bi0 = vuzp2q_f32(b0, b1);
                        let br1 = vuzp1q_f32(b2, b3); let bi1 = vuzp2q_f32(b2, b3);
                        acc_re1 = vfmaq_f32(acc_re1, br0, xr0);
                        acc_re1 = vfmaq_f32(acc_re1, bi0, xi0);
                        acc_im1 = vfmaq_f32(acc_im1, br0, xi0);
                        acc_im1 = vfmsq_f32(acc_im1, bi0, xr0);
                        acc_re1 = vfmaq_f32(acc_re1, br1, xr1);
                        acc_re1 = vfmaq_f32(acc_re1, bi1, xi1);
                        acc_im1 = vfmaq_f32(acc_im1, br1, xi1);
                        acc_im1 = vfmsq_f32(acc_im1, bi1, xr1);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1 = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let xr = vuzp1q_f32(x0, x1); let xi = vuzp2q_f32(x0, x1);

                        let a0  = vld1q_f32(col0.add(2 * i));
                        let a1  = vld1q_f32(col0.add(2 * (i + 2)));
                        let ar  = vuzp1q_f32(a0, a1); let ai = vuzp2q_f32(a0, a1);
                        acc_re0 = vfmaq_f32(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f32(acc_im0, ai, xr);

                        let b0  = vld1q_f32(col1.add(2 * i));
                        let b1  = vld1q_f32(col1.add(2 * (i + 2)));
                        let br  = vuzp1q_f32(b0, b1); let bi = vuzp2q_f32(b0, b1);
                        acc_re1 = vfmaq_f32(acc_re1, br, xr);
                        acc_re1 = vfmaq_f32(acc_re1, bi, xi);
                        acc_im1 = vfmaq_f32(acc_im1, br, xi);
                        acc_im1 = vfmsq_f32(acc_im1, bi, xr);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let xr = vuzp1q_f32(x0, x0);
                        let xi = vuzp2q_f32(x0, x0);

                        let a0  = vld1q_f32(col0.add(2 * i));
                        let ar  = vuzp1q_f32(a0, a0); let ai = vuzp2q_f32(a0, a0);
                        acc_re0 = vfmaq_f32(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f32(acc_im0, ai, xr);

                        let b0  = vld1q_f32(col1.add(2 * i));
                        let br  = vuzp1q_f32(b0, b0); let bi = vuzp2q_f32(b0, b0);
                        acc_re1 = vfmaq_f32(acc_re1, br, xr);
                        acc_re1 = vfmaq_f32(acc_re1, bi, xi);
                        acc_im1 = vfmaq_f32(acc_im1, br, xi);
                        acc_im1 = vfmsq_f32(acc_im1, bi, xr);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f32(acc_re0); sum_im0 += vaddvq_f32(acc_im0);
                    sum_re1 += vaddvq_f32(acc_re1); sum_im1 += vaddvq_f32(acc_im1);

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

                    let mut acc_re0 = vdupq_n_f32(0.0);
                    let mut acc_im0 = vdupq_n_f32(0.0);

                    let col0 = pa0.add(2 * row_idx);

                    let mut i = 0;

                    while i + 8 <= mr {
                        let x0  = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let x2  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 4)));
                        let x3  = vld1q_f32(x_ptr.add(2 * (row_idx + i + 6)));
                        let xr0 = vuzp1q_f32(x0, x1); let xi0 = vuzp2q_f32(x0, x1);
                        let xr1 = vuzp1q_f32(x2, x3); let xi1 = vuzp2q_f32(x2, x3);

                        let a0  = vld1q_f32(col0.add(2 * i));
                        let a1  = vld1q_f32(col0.add(2 * (i + 2)));
                        let a2  = vld1q_f32(col0.add(2 * (i + 4)));
                        let a3  = vld1q_f32(col0.add(2 * (i + 6)));
                        let ar0 = vuzp1q_f32(a0, a1); let ai0 = vuzp2q_f32(a0, a1);
                        let ar1 = vuzp1q_f32(a2, a3); let ai1 = vuzp2q_f32(a2, a3);

                        acc_re0 = vfmaq_f32(acc_re0, ar0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ai0, xi0);
                        acc_im0 = vfmaq_f32(acc_im0, ar0, xi0);
                        acc_im0 = vfmsq_f32(acc_im0, ai0, xr0);
                        acc_re0 = vfmaq_f32(acc_re0, ar1, xr1);
                        acc_re0 = vfmaq_f32(acc_re0, ai1, xi1);
                        acc_im0 = vfmaq_f32(acc_im0, ar1, xi1);
                        acc_im0 = vfmsq_f32(acc_im0, ai1, xr1);

                        i += 8;
                    }

                    while i + 4 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let x1 = vld1q_f32(x_ptr.add(2 * (row_idx + i + 2)));
                        let xr = vuzp1q_f32(x0, x1);
                        let xi = vuzp2q_f32(x0, x1);

                        let a0 = vld1q_f32(col0.add(2 * i));
                        let a1 = vld1q_f32(col0.add(2 * (i + 2)));
                        let ar = vuzp1q_f32(a0, a1); let ai = vuzp2q_f32(a0, a1);

                        acc_re0 = vfmaq_f32(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f32(acc_im0, ai, xr);

                        i += 4;
                    }

                    while i + 2 <= mr {
                        let x0 = vld1q_f32(x_ptr.add(2 * (row_idx + i)));
                        let xr = vuzp1q_f32(x0, x0);
                        let xi = vuzp2q_f32(x0, x0);

                        let a0 = vld1q_f32(col0.add(2 * i));
                        let ar = vuzp1q_f32(a0, a0); let ai = vuzp2q_f32(a0, a0);

                        acc_re0 = vfmaq_f32(acc_re0, ar, xr);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi);
                        acc_im0 = vfmsq_f32(acc_im0, ai, xr);

                        i += 2;
                    }

                    sum_re0 += vaddvq_f32(acc_re0);
                    sum_im0 += vaddvq_f32(acc_im0);

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

    // non unit stride
    for col_idx in 0..n_cols {
        unsafe {
            let col_ptr = matrix.as_ptr().add(2 * (col_idx * lda));
            let col = core::slice::from_raw_parts(col_ptr, 2 * n_rows);

            let s = cdotc(n_rows, col, 1, x, incx);

            let po = out.as_mut_ptr().add(2 * col_idx);
            *po.add(0) += s[0];
            *po.add(1) += s[1];
        }
    }
}

