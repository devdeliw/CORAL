use core::arch::aarch64::{vld1q_f64, vdupq_n_f64, vfmaq_f64, vst1q_f64};
use crate::level1::assert_length_helpers::required_len_ok;
use crate::level1::daxpy::daxpy;

#[inline(always)]
pub fn daxpyf(
    m     : usize,
    n     : usize,
    x     : &[f64],
    incx  : isize,
    a     : &[f64],
    lda   : usize,
    y     : &mut [f64],
    incy  : isize,
) {
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), m, incy), "y too short for m/incy");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m");
        debug_assert!(
            a.len() >= (n - 1).saturating_mul(lda) + m,
            "A too small for m, n, lda; need at least (n - 1)*lda + m"
        );
    }

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            let mut j = 0usize;

            while j + 8 <= n {
                let s0 = vdupq_n_f64(*x.get_unchecked(j + 0));
                let s1 = vdupq_n_f64(*x.get_unchecked(j + 1));
                let s2 = vdupq_n_f64(*x.get_unchecked(j + 2));
                let s3 = vdupq_n_f64(*x.get_unchecked(j + 3));
                let s4 = vdupq_n_f64(*x.get_unchecked(j + 4));
                let s5 = vdupq_n_f64(*x.get_unchecked(j + 5));
                let s6 = vdupq_n_f64(*x.get_unchecked(j + 6));
                let s7 = vdupq_n_f64(*x.get_unchecked(j + 7));

                let pa0 = a.as_ptr().add((j + 0) * lda);
                let pa1 = a.as_ptr().add((j + 1) * lda);
                let pa2 = a.as_ptr().add((j + 2) * lda);
                let pa3 = a.as_ptr().add((j + 3) * lda);
                let pa4 = a.as_ptr().add((j + 4) * lda);
                let pa5 = a.as_ptr().add((j + 5) * lda);
                let pa6 = a.as_ptr().add((j + 6) * lda);
                let pa7 = a.as_ptr().add((j + 7) * lda);

                let mut i = 0usize;

                while i + 16 <= m {
                    let yb = y.as_mut_ptr().add(i);

                    let mut y0 = vld1q_f64(yb.add( 0));
                    let mut y1 = vld1q_f64(yb.add( 2));
                    let mut y2 = vld1q_f64(yb.add( 4));
                    let mut y3 = vld1q_f64(yb.add( 6));
                    let mut y4 = vld1q_f64(yb.add( 8));
                    let mut y5 = vld1q_f64(yb.add(10));
                    let mut y6 = vld1q_f64(yb.add(12));
                    let mut y7 = vld1q_f64(yb.add(14));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);
                    let mut a2p = pa2.add(i);
                    let mut a3p = pa3.add(i);
                    let mut a4p = pa4.add(i);
                    let mut a5p = pa5.add(i);
                    let mut a6p = pa6.add(i);
                    let mut a7p = pa7.add(i);

                    // col 0
                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y4 = vfmaq_f64(y4, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y5 = vfmaq_f64(y5, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y6 = vfmaq_f64(y6, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y7 = vfmaq_f64(y7, s0, vld1q_f64(a0p));

                    // col 1
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y4 = vfmaq_f64(y4, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y5 = vfmaq_f64(y5, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y6 = vfmaq_f64(y6, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y7 = vfmaq_f64(y7, s1, vld1q_f64(a1p));

                    // col 2
                    y0 = vfmaq_f64(y0, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y2 = vfmaq_f64(y2, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y3 = vfmaq_f64(y3, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y4 = vfmaq_f64(y4, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y5 = vfmaq_f64(y5, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y6 = vfmaq_f64(y6, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y7 = vfmaq_f64(y7, s2, vld1q_f64(a2p));

                    // col 3
                    y0 = vfmaq_f64(y0, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y2 = vfmaq_f64(y2, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y3 = vfmaq_f64(y3, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y4 = vfmaq_f64(y4, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y5 = vfmaq_f64(y5, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y6 = vfmaq_f64(y6, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y7 = vfmaq_f64(y7, s3, vld1q_f64(a3p));

                    // col 4
                    y0 = vfmaq_f64(y0, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y1 = vfmaq_f64(y1, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y2 = vfmaq_f64(y2, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y3 = vfmaq_f64(y3, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y4 = vfmaq_f64(y4, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y5 = vfmaq_f64(y5, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y6 = vfmaq_f64(y6, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y7 = vfmaq_f64(y7, s4, vld1q_f64(a4p));

                    // col 5
                    y0 = vfmaq_f64(y0, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y1 = vfmaq_f64(y1, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y2 = vfmaq_f64(y2, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y3 = vfmaq_f64(y3, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y4 = vfmaq_f64(y4, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y5 = vfmaq_f64(y5, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y6 = vfmaq_f64(y6, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y7 = vfmaq_f64(y7, s5, vld1q_f64(a5p));

                    // col 6
                    y0 = vfmaq_f64(y0, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y1 = vfmaq_f64(y1, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y2 = vfmaq_f64(y2, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y3 = vfmaq_f64(y3, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y4 = vfmaq_f64(y4, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y5 = vfmaq_f64(y5, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y6 = vfmaq_f64(y6, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y7 = vfmaq_f64(y7, s6, vld1q_f64(a6p));

                    // col 7
                    y0 = vfmaq_f64(y0, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y1 = vfmaq_f64(y1, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y2 = vfmaq_f64(y2, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y3 = vfmaq_f64(y3, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y4 = vfmaq_f64(y4, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y5 = vfmaq_f64(y5, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y6 = vfmaq_f64(y6, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y7 = vfmaq_f64(y7, s7, vld1q_f64(a7p));

                    vst1q_f64(yb.add( 0), y0);
                    vst1q_f64(yb.add( 2), y1);
                    vst1q_f64(yb.add( 4), y2);
                    vst1q_f64(yb.add( 6), y3);
                    vst1q_f64(yb.add( 8), y4);
                    vst1q_f64(yb.add(10), y5);
                    vst1q_f64(yb.add(12), y6);
                    vst1q_f64(yb.add(14), y7);

                    i += 16;
                }

                while i + 8 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));
                    let mut y2 = vld1q_f64(yb.add(4));
                    let mut y3 = vld1q_f64(yb.add(6));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);
                    let mut a2p = pa2.add(i);
                    let mut a3p = pa3.add(i);
                    let mut a4p = pa4.add(i);
                    let mut a5p = pa5.add(i);
                    let mut a6p = pa6.add(i);
                    let mut a7p = pa7.add(i);

                    // col 0
                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p));

                    // col 1
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p));

                    // col 2
                    y0 = vfmaq_f64(y0, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y2 = vfmaq_f64(y2, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y3 = vfmaq_f64(y3, s2, vld1q_f64(a2p));

                    // col 3
                    y0 = vfmaq_f64(y0, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y2 = vfmaq_f64(y2, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y3 = vfmaq_f64(y3, s3, vld1q_f64(a3p));

                    // col 4
                    y0 = vfmaq_f64(y0, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y1 = vfmaq_f64(y1, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y2 = vfmaq_f64(y2, s4, vld1q_f64(a4p)); a4p = a4p.add(2);
                    y3 = vfmaq_f64(y3, s4, vld1q_f64(a4p));

                    // col 5
                    y0 = vfmaq_f64(y0, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y1 = vfmaq_f64(y1, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y2 = vfmaq_f64(y2, s5, vld1q_f64(a5p)); a5p = a5p.add(2);
                    y3 = vfmaq_f64(y3, s5, vld1q_f64(a5p));

                    // col 6
                    y0 = vfmaq_f64(y0, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y1 = vfmaq_f64(y1, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y2 = vfmaq_f64(y2, s6, vld1q_f64(a6p)); a6p = a6p.add(2);
                    y3 = vfmaq_f64(y3, s6, vld1q_f64(a6p));

                    // col 7
                    y0 = vfmaq_f64(y0, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y1 = vfmaq_f64(y1, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y2 = vfmaq_f64(y2, s7, vld1q_f64(a7p)); a7p = a7p.add(2);
                    y3 = vfmaq_f64(y3, s7, vld1q_f64(a7p));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);
                    vst1q_f64(yb.add(4), y2);
                    vst1q_f64(yb.add(6), y3);

                    i += 8;
                }

                while i + 4 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));

                    let a0p = pa0.add(i);
                    let a1p = pa1.add(i);
                    let a2p = pa2.add(i);
                    let a3p = pa3.add(i);
                    let a4p = pa4.add(i);
                    let a5p = pa5.add(i);
                    let a6p = pa6.add(i);
                    let a7p = pa7.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p));
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p.add(2)));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p));
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p.add(2)));

                    y0 = vfmaq_f64(y0, s2, vld1q_f64(a2p));
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(a2p.add(2)));

                    y0 = vfmaq_f64(y0, s3, vld1q_f64(a3p));
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(a3p.add(2)));

                    y0 = vfmaq_f64(y0, s4, vld1q_f64(a4p));
                    y1 = vfmaq_f64(y1, s4, vld1q_f64(a4p.add(2)));

                    y0 = vfmaq_f64(y0, s5, vld1q_f64(a5p));
                    y1 = vfmaq_f64(y1, s5, vld1q_f64(a5p.add(2)));

                    y0 = vfmaq_f64(y0, s6, vld1q_f64(a6p));
                    y1 = vfmaq_f64(y1, s6, vld1q_f64(a6p.add(2)));

                    y0 = vfmaq_f64(y0, s7, vld1q_f64(a7p));
                    y1 = vfmaq_f64(y1, s7, vld1q_f64(a7p.add(2)));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);

                    i += 4;
                }

                while i + 2 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(pa0.add(i)));
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(pa1.add(i)));
                    y0 = vfmaq_f64(y0, s2, vld1q_f64(pa2.add(i)));
                    y0 = vfmaq_f64(y0, s3, vld1q_f64(pa3.add(i)));
                    y0 = vfmaq_f64(y0, s4, vld1q_f64(pa4.add(i)));
                    y0 = vfmaq_f64(y0, s5, vld1q_f64(pa5.add(i)));
                    y0 = vfmaq_f64(y0, s6, vld1q_f64(pa6.add(i)));
                    y0 = vfmaq_f64(y0, s7, vld1q_f64(pa7.add(i)));

                    vst1q_f64(yb, y0);

                    i += 2;
                }

                // tail 
                while i < m {
                    let mut acc = *y.get_unchecked(i);
                    acc += *x.get_unchecked(j + 0) * *pa0.add(i);
                    acc += *x.get_unchecked(j + 1) * *pa1.add(i);
                    acc += *x.get_unchecked(j + 2) * *pa2.add(i);
                    acc += *x.get_unchecked(j + 3) * *pa3.add(i);
                    acc += *x.get_unchecked(j + 4) * *pa4.add(i);
                    acc += *x.get_unchecked(j + 5) * *pa5.add(i);
                    acc += *x.get_unchecked(j + 6) * *pa6.add(i);
                    acc += *x.get_unchecked(j + 7) * *pa7.add(i);
                    *y.get_unchecked_mut(i) = acc;
                    i += 1;
                }

                j += 8;
            }

            if j + 4 <= n {
                let s0 = vdupq_n_f64(*x.get_unchecked(j + 0));
                let s1 = vdupq_n_f64(*x.get_unchecked(j + 1));
                let s2 = vdupq_n_f64(*x.get_unchecked(j + 2));
                let s3 = vdupq_n_f64(*x.get_unchecked(j + 3));

                let pa0 = a.as_ptr().add((j + 0) * lda);
                let pa1 = a.as_ptr().add((j + 1) * lda);
                let pa2 = a.as_ptr().add((j + 2) * lda);
                let pa3 = a.as_ptr().add((j + 3) * lda);

                let mut i = 0usize;

                while i + 16 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add( 0));
                    let mut y1 = vld1q_f64(yb.add( 2));
                    let mut y2 = vld1q_f64(yb.add( 4));
                    let mut y3 = vld1q_f64(yb.add( 6));
                    let mut y4 = vld1q_f64(yb.add( 8));
                    let mut y5 = vld1q_f64(yb.add(10));
                    let mut y6 = vld1q_f64(yb.add(12));
                    let mut y7 = vld1q_f64(yb.add(14));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);
                    let mut a2p = pa2.add(i);
                    let mut a3p = pa3.add(i);

                    // col 0
                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y4 = vfmaq_f64(y4, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y5 = vfmaq_f64(y5, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y6 = vfmaq_f64(y6, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y7 = vfmaq_f64(y7, s0, vld1q_f64(a0p));

                    // col 1
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y4 = vfmaq_f64(y4, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y5 = vfmaq_f64(y5, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y6 = vfmaq_f64(y6, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y7 = vfmaq_f64(y7, s1, vld1q_f64(a1p));

                    // col 2
                    y0 = vfmaq_f64(y0, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y2 = vfmaq_f64(y2, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y3 = vfmaq_f64(y3, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y4 = vfmaq_f64(y4, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y5 = vfmaq_f64(y5, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y6 = vfmaq_f64(y6, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y7 = vfmaq_f64(y7, s2, vld1q_f64(a2p));

                    // col 3
                    y0 = vfmaq_f64(y0, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y2 = vfmaq_f64(y2, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y3 = vfmaq_f64(y3, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y4 = vfmaq_f64(y4, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y5 = vfmaq_f64(y5, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y6 = vfmaq_f64(y6, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y7 = vfmaq_f64(y7, s3, vld1q_f64(a3p));

                    vst1q_f64(yb.add( 0), y0);
                    vst1q_f64(yb.add( 2), y1);
                    vst1q_f64(yb.add( 4), y2);
                    vst1q_f64(yb.add( 6), y3);
                    vst1q_f64(yb.add( 8), y4);
                    vst1q_f64(yb.add(10), y5);
                    vst1q_f64(yb.add(12), y6);
                    vst1q_f64(yb.add(14), y7);

                    i += 16;
                }

                while i + 8 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));
                    let mut y2 = vld1q_f64(yb.add(4));
                    let mut y3 = vld1q_f64(yb.add(6));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);
                    let mut a2p = pa2.add(i);
                    let mut a3p = pa3.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p));

                    y0 = vfmaq_f64(y0, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y2 = vfmaq_f64(y2, s2, vld1q_f64(a2p)); a2p = a2p.add(2);
                    y3 = vfmaq_f64(y3, s2, vld1q_f64(a2p));

                    y0 = vfmaq_f64(y0, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y2 = vfmaq_f64(y2, s3, vld1q_f64(a3p)); a3p = a3p.add(2);
                    y3 = vfmaq_f64(y3, s3, vld1q_f64(a3p));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);
                    vst1q_f64(yb.add(4), y2);
                    vst1q_f64(yb.add(6), y3);

                    i += 8;
                }

                while i + 4 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(pa0.add(i)));
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(pa0.add(i + 2)));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(pa1.add(i)));
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(pa1.add(i + 2)));

                    y0 = vfmaq_f64(y0, s2, vld1q_f64(pa2.add(i)));
                    y1 = vfmaq_f64(y1, s2, vld1q_f64(pa2.add(i + 2)));

                    y0 = vfmaq_f64(y0, s3, vld1q_f64(pa3.add(i)));
                    y1 = vfmaq_f64(y1, s3, vld1q_f64(pa3.add(i + 2)));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);

                    i += 4;
                }

                while i + 2 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(pa0.add(i)));
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(pa1.add(i)));
                    y0 = vfmaq_f64(y0, s2, vld1q_f64(pa2.add(i)));
                    y0 = vfmaq_f64(y0, s3, vld1q_f64(pa3.add(i)));

                    vst1q_f64(yb, y0);

                    i += 2;
                }

                while i < m {
                    let mut acc = *y.get_unchecked(i);
                    acc += *x.get_unchecked(j + 0) * *pa0.add(i);
                    acc += *x.get_unchecked(j + 1) * *pa1.add(i);
                    acc += *x.get_unchecked(j + 2) * *pa2.add(i);
                    acc += *x.get_unchecked(j + 3) * *pa3.add(i);
                    *y.get_unchecked_mut(i) = acc;
                    i += 1;
                }

                j += 4;
            }

            if j + 2 <= n {
                let s0 = vdupq_n_f64(*x.get_unchecked(j + 0));
                let s1 = vdupq_n_f64(*x.get_unchecked(j + 1));

                let pa0 = a.as_ptr().add((j + 0) * lda);
                let pa1 = a.as_ptr().add((j + 1) * lda);

                let mut i = 0usize;

                while i + 16 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add( 0));
                    let mut y1 = vld1q_f64(yb.add( 2));
                    let mut y2 = vld1q_f64(yb.add( 4));
                    let mut y3 = vld1q_f64(yb.add( 6));
                    let mut y4 = vld1q_f64(yb.add( 8));
                    let mut y5 = vld1q_f64(yb.add(10));
                    let mut y6 = vld1q_f64(yb.add(12));
                    let mut y7 = vld1q_f64(yb.add(14));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y4 = vfmaq_f64(y4, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y5 = vfmaq_f64(y5, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y6 = vfmaq_f64(y6, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y7 = vfmaq_f64(y7, s0, vld1q_f64(a0p));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y4 = vfmaq_f64(y4, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y5 = vfmaq_f64(y5, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y6 = vfmaq_f64(y6, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y7 = vfmaq_f64(y7, s1, vld1q_f64(a1p));

                    vst1q_f64(yb.add( 0), y0);
                    vst1q_f64(yb.add( 2), y1);
                    vst1q_f64(yb.add( 4), y2);
                    vst1q_f64(yb.add( 6), y3);
                    vst1q_f64(yb.add( 8), y4);
                    vst1q_f64(yb.add(10), y5);
                    vst1q_f64(yb.add(12), y6);
                    vst1q_f64(yb.add(14), y7);

                    i += 16;
                }

                while i + 8 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));
                    let mut y2 = vld1q_f64(yb.add(4));
                    let mut y3 = vld1q_f64(yb.add(6));

                    let mut a0p = pa0.add(i);
                    let mut a1p = pa1.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y2 = vfmaq_f64(y2, s1, vld1q_f64(a1p)); a1p = a1p.add(2);
                    y3 = vfmaq_f64(y3, s1, vld1q_f64(a1p));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);
                    vst1q_f64(yb.add(4), y2);
                    vst1q_f64(yb.add(6), y3);

                    i += 8;
                }

                while i + 4 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(pa0.add(i)));
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(pa0.add(i + 2)));

                    y0 = vfmaq_f64(y0, s1, vld1q_f64(pa1.add(i)));
                    y1 = vfmaq_f64(y1, s1, vld1q_f64(pa1.add(i + 2)));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);

                    i += 4;
                }

                while i + 2 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(pa0.add(i)));
                    y0 = vfmaq_f64(y0, s1, vld1q_f64(pa1.add(i)));

                    vst1q_f64(yb, y0);

                    i += 2;
                }

                while i < m {
                    let mut acc = *y.get_unchecked(i);
                    acc += *x.get_unchecked(j + 0) * *pa0.add(i);
                    acc += *x.get_unchecked(j + 1) * *pa1.add(i);
                    *y.get_unchecked_mut(i) = acc;
                    i += 1;
                }

                j += 2;
            }

            if j < n {
                let s0 = vdupq_n_f64(*x.get_unchecked(j));
                let pa0 = a.as_ptr().add(j * lda);

                let mut i = 0usize;

                while i + 16 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add( 0));
                    let mut y1 = vld1q_f64(yb.add( 2));
                    let mut y2 = vld1q_f64(yb.add( 4));
                    let mut y3 = vld1q_f64(yb.add( 6));
                    let mut y4 = vld1q_f64(yb.add( 8));
                    let mut y5 = vld1q_f64(yb.add(10));
                    let mut y6 = vld1q_f64(yb.add(12));
                    let mut y7 = vld1q_f64(yb.add(14));

                    let mut a0p = pa0.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y4 = vfmaq_f64(y4, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y5 = vfmaq_f64(y5, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y6 = vfmaq_f64(y6, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y7 = vfmaq_f64(y7, s0, vld1q_f64(a0p));

                    vst1q_f64(yb.add( 0), y0);
                    vst1q_f64(yb.add( 2), y1);
                    vst1q_f64(yb.add( 4), y2);
                    vst1q_f64(yb.add( 6), y3);
                    vst1q_f64(yb.add( 8), y4);
                    vst1q_f64(yb.add(10), y5);
                    vst1q_f64(yb.add(12), y6);
                    vst1q_f64(yb.add(14), y7);

                    i += 16;
                }

                while i + 8 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let mut y0 = vld1q_f64(yb.add(0));
                    let mut y1 = vld1q_f64(yb.add(2));
                    let mut y2 = vld1q_f64(yb.add(4));
                    let mut y3 = vld1q_f64(yb.add(6));

                    let mut a0p = pa0.add(i);

                    y0 = vfmaq_f64(y0, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y1 = vfmaq_f64(y1, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y2 = vfmaq_f64(y2, s0, vld1q_f64(a0p)); a0p = a0p.add(2);
                    y3 = vfmaq_f64(y3, s0, vld1q_f64(a0p));

                    vst1q_f64(yb.add(0), y0);
                    vst1q_f64(yb.add(2), y1);
                    vst1q_f64(yb.add(4), y2);
                    vst1q_f64(yb.add(6), y3);

                    i += 8;
                }

                while i + 2 <= m {
                    let yb = y.as_mut_ptr().add(i);
                    let y0 = vfmaq_f64(vld1q_f64(yb), s0, vld1q_f64(pa0.add(i)));
                    vst1q_f64(yb, y0);
                    i += 2;
                }

                while i < m {
                    *y.get_unchecked_mut(i) = *y.get_unchecked(i) + *x.get_unchecked(j) * *pa0.add(i);
                    i += 1;
                }
            }

            return;
        }

        let stepx  = if incx > 0 { incx as usize } else { (-incx) as usize };
        let mut px = x.as_ptr().wrapping_add(if incx >= 0 { 0 } else { (n - 1) * stepx });

        for _jj in 0..n {
            let scaled = *px;
            if scaled != 0.0 {
                // col major contiguous 
                let col_ptr = a.as_ptr().add(_jj * lda);
                let col = core::slice::from_raw_parts(col_ptr, m);
                daxpy(m, scaled, col, 1, y, incy);
            }

            px = px.offset(incx);
        }
    }
}

