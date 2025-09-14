use core::arch::aarch64::*;
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level1::zaxpy::zaxpy;

#[inline(always)]
pub fn zaxpyf(
    m: usize,
    n: usize,
    x: &[f64],
    incx: isize,
    a: &[f64],
    lda: usize,
    y: &mut [f64],
    incy: isize,
) {
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), m, incy), "y too short for m/incy");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m (in complexes)");
        let need = 2 * ((n - 1).saturating_mul(lda) + m);
        debug_assert!(a.len() >= need, "A too small for m,n,lda");
    }

    unsafe {
        if incx == 1 && incy == 1 {
            let m2 = m << 1;
            let lda2 = lda << 1;

            let xptr = x.as_ptr();
            let yptr = y.as_mut_ptr();
            let aptr = a.as_ptr();

            let mut j = 0usize;

            while j + 8 <= n {
                let xr0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 0));
                let xi0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 1));
                let xr1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 0));
                let xi1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 1));
                let xr2 = vdupq_n_f64(*xptr.add(2 * (j + 2) + 0));
                let xi2 = vdupq_n_f64(*xptr.add(2 * (j + 2) + 1));
                let xr3 = vdupq_n_f64(*xptr.add(2 * (j + 3) + 0));
                let xi3 = vdupq_n_f64(*xptr.add(2 * (j + 3) + 1));
                let xr4 = vdupq_n_f64(*xptr.add(2 * (j + 4) + 0));
                let xi4 = vdupq_n_f64(*xptr.add(2 * (j + 4) + 1));
                let xr5 = vdupq_n_f64(*xptr.add(2 * (j + 5) + 0));
                let xi5 = vdupq_n_f64(*xptr.add(2 * (j + 5) + 1));
                let xr6 = vdupq_n_f64(*xptr.add(2 * (j + 6) + 0));
                let xi6 = vdupq_n_f64(*xptr.add(2 * (j + 6) + 1));
                let xr7 = vdupq_n_f64(*xptr.add(2 * (j + 7) + 0));
                let xi7 = vdupq_n_f64(*xptr.add(2 * (j + 7) + 1));

                let pa0 = aptr.add(lda2 * (j + 0));
                let pa1 = aptr.add(lda2 * (j + 1));
                let pa2 = aptr.add(lda2 * (j + 2));
                let pa3 = aptr.add(lda2 * (j + 3));
                let pa4 = aptr.add(lda2 * (j + 4));
                let pa5 = aptr.add(lda2 * (j + 5));
                let pa6 = aptr.add(lda2 * (j + 6));
                let pa7 = aptr.add(lda2 * (j + 7));

                let mut i2 = 0usize;

                while i2 + 8 <= m2 {
                    let yp0 = yptr.add(i2);
                    let mut y0a: float64x2x2_t = vld2q_f64(yp0);
                    let yp0b = yptr.add(i2 + 4);
                    let mut y0b: float64x2x2_t = vld2q_f64(yp0b);

                    let a0a: float64x2x2_t = vld2q_f64(pa0.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr0, a0a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi0, a0a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr0, a0a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi0, a0a.0);
                    let a0b: float64x2x2_t = vld2q_f64(pa0.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr0, a0b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi0, a0b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr0, a0b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi0, a0b.0);

                    let a1a: float64x2x2_t = vld2q_f64(pa1.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr1, a1a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi1, a1a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr1, a1a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi1, a1a.0);
                    let a1b: float64x2x2_t = vld2q_f64(pa1.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr1, a1b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi1, a1b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr1, a1b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi1, a1b.0);

                    let a2a: float64x2x2_t = vld2q_f64(pa2.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr2, a2a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi2, a2a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr2, a2a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi2, a2a.0);
                    let a2b: float64x2x2_t = vld2q_f64(pa2.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr2, a2b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi2, a2b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr2, a2b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi2, a2b.0);

                    let a3a: float64x2x2_t = vld2q_f64(pa3.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr3, a3a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi3, a3a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr3, a3a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi3, a3a.0);
                    let a3b: float64x2x2_t = vld2q_f64(pa3.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr3, a3b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi3, a3b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr3, a3b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi3, a3b.0);

                    let a4a: float64x2x2_t = vld2q_f64(pa4.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr4, a4a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi4, a4a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr4, a4a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi4, a4a.0);
                    let a4b: float64x2x2_t = vld2q_f64(pa4.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr4, a4b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi4, a4b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr4, a4b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi4, a4b.0);

                    let a5a: float64x2x2_t = vld2q_f64(pa5.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr5, a5a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi5, a5a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr5, a5a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi5, a5a.0);
                    let a5b: float64x2x2_t = vld2q_f64(pa5.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr5, a5b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi5, a5b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr5, a5b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi5, a5b.0);

                    let a6a: float64x2x2_t = vld2q_f64(pa6.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr6, a6a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi6, a6a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr6, a6a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi6, a6a.0);
                    let a6b: float64x2x2_t = vld2q_f64(pa6.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr6, a6b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi6, a6b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr6, a6b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi6, a6b.0);

                    let a7a: float64x2x2_t = vld2q_f64(pa7.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr7, a7a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi7, a7a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr7, a7a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi7, a7a.0);
                    let a7b: float64x2x2_t = vld2q_f64(pa7.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr7, a7b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi7, a7b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr7, a7b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi7, a7b.0);

                    vst2q_f64(yp0, y0a);
                    vst2q_f64(yp0b, y0b);

                    i2 += 8;
                }

                while i2 + 4 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float64x2x2_t = vld2q_f64(yp);

                    let a0: float64x2x2_t = vld2q_f64(pa0.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                    yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                    let a1: float64x2x2_t = vld2q_f64(pa1.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                    yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                    let a2: float64x2x2_t = vld2q_f64(pa2.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr2, a2.0);
                    yv.0 = vfmsq_f64(yv.0, xi2, a2.1);
                    yv.1 = vfmaq_f64(yv.1, xr2, a2.1);
                    yv.1 = vfmaq_f64(yv.1, xi2, a2.0);

                    let a3: float64x2x2_t = vld2q_f64(pa3.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr3, a3.0);
                    yv.0 = vfmsq_f64(yv.0, xi3, a3.1);
                    yv.1 = vfmaq_f64(yv.1, xr3, a3.1);
                    yv.1 = vfmaq_f64(yv.1, xi3, a3.0);

                    let a4: float64x2x2_t = vld2q_f64(pa4.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr4, a4.0);
                    yv.0 = vfmsq_f64(yv.0, xi4, a4.1);
                    yv.1 = vfmaq_f64(yv.1, xr4, a4.1);
                    yv.1 = vfmaq_f64(yv.1, xi4, a4.0);

                    let a5: float64x2x2_t = vld2q_f64(pa5.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr5, a5.0);
                    yv.0 = vfmsq_f64(yv.0, xi5, a5.1);
                    yv.1 = vfmaq_f64(yv.1, xr5, a5.1);
                    yv.1 = vfmaq_f64(yv.1, xi5, a5.0);

                    let a6: float64x2x2_t = vld2q_f64(pa6.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr6, a6.0);
                    yv.0 = vfmsq_f64(yv.0, xi6, a6.1);
                    yv.1 = vfmaq_f64(yv.1, xr6, a6.1);
                    yv.1 = vfmaq_f64(yv.1, xi6, a6.0);

                    let a7: float64x2x2_t = vld2q_f64(pa7.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr7, a7.0);
                    yv.0 = vfmsq_f64(yv.0, xi7, a7.1);
                    yv.1 = vfmaq_f64(yv.1, xr7, a7.1);
                    yv.1 = vfmaq_f64(yv.1, xi7, a7.0);

                    vst2q_f64(yp, yv);

                    i2 += 4;
                }

                while i2 < m2 {
                    let ar0 = *pa0.add(i2 + 0);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2 + 0);
                    let ai1 = *pa1.add(i2 + 1);
                    let ar2 = *pa2.add(i2 + 0);
                    let ai2 = *pa2.add(i2 + 1);
                    let ar3 = *pa3.add(i2 + 0);
                    let ai3 = *pa3.add(i2 + 1);
                    let ar4 = *pa4.add(i2 + 0);
                    let ai4 = *pa4.add(i2 + 1);
                    let ar5 = *pa5.add(i2 + 0);
                    let ai5 = *pa5.add(i2 + 1);
                    let ar6 = *pa6.add(i2 + 0);
                    let ai6 = *pa6.add(i2 + 1);
                    let ar7 = *pa7.add(i2 + 0);
                    let ai7 = *pa7.add(i2 + 1);

                    let xr0s = *xptr.add(2 * (j + 0) + 0);
                    let xi0s = *xptr.add(2 * (j + 0) + 1);
                    let xr1s = *xptr.add(2 * (j + 1) + 0);
                    let xi1s = *xptr.add(2 * (j + 1) + 1);
                    let xr2s = *xptr.add(2 * (j + 2) + 0);
                    let xi2s = *xptr.add(2 * (j + 2) + 1);
                    let xr3s = *xptr.add(2 * (j + 3) + 0);
                    let xi3s = *xptr.add(2 * (j + 3) + 1);
                    let xr4s = *xptr.add(2 * (j + 4) + 0);
                    let xi4s = *xptr.add(2 * (j + 4) + 1);
                    let xr5s = *xptr.add(2 * (j + 5) + 0);
                    let xi5s = *xptr.add(2 * (j + 5) + 1);
                    let xr6s = *xptr.add(2 * (j + 6) + 0);
                    let xi6s = *xptr.add(2 * (j + 6) + 1);
                    let xr7s = *xptr.add(2 * (j + 7) + 0);
                    let xi7s = *xptr.add(2 * (j + 7) + 1);

                    let yrp = yptr.add(i2 + 0);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    yr += xr0s * ar0 - xi0s * ai0;
                    yi += xr0s * ai0 + xi0s * ar0;
                    yr += xr1s * ar1 - xi1s * ai1;
                    yi += xr1s * ai1 + xi1s * ar1;
                    yr += xr2s * ar2 - xi2s * ai2;
                    yi += xr2s * ai2 + xi2s * ar2;
                    yr += xr3s * ar3 - xi3s * ai3;
                    yi += xr3s * ai3 + xi3s * ar3;
                    yr += xr4s * ar4 - xi4s * ai4;
                    yi += xr4s * ai4 + xi4s * ar4;
                    yr += xr5s * ar5 - xi5s * ai5;
                    yi += xr5s * ai5 + xi5s * ar5;
                    yr += xr6s * ar6 - xi6s * ai6;
                    yi += xr6s * ai6 + xi6s * ar6;
                    yr += xr7s * ar7 - xi7s * ai7;
                    yi += xr7s * ai7 + xi7s * ar7;

                    *yrp = yr;
                    *yip = yi;

                    i2 += 2;
                }

                j += 8;
            }

            while j + 4 <= n {
                let xr0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 0));
                let xi0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 1));
                let xr1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 0));
                let xi1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 1));
                let xr2 = vdupq_n_f64(*xptr.add(2 * (j + 2) + 0));
                let xi2 = vdupq_n_f64(*xptr.add(2 * (j + 2) + 1));
                let xr3 = vdupq_n_f64(*xptr.add(2 * (j + 3) + 0));
                let xi3 = vdupq_n_f64(*xptr.add(2 * (j + 3) + 1));

                let pa0 = aptr.add(lda2 * (j + 0));
                let pa1 = aptr.add(lda2 * (j + 1));
                let pa2 = aptr.add(lda2 * (j + 2));
                let pa3 = aptr.add(lda2 * (j + 3));

                let mut i2 = 0usize;

                while i2 + 8 <= m2 {
                    let yp0 = yptr.add(i2);
                    let mut y0a: float64x2x2_t = vld2q_f64(yp0);
                    let yp0b = yptr.add(i2 + 4);
                    let mut y0b: float64x2x2_t = vld2q_f64(yp0b);

                    let a0a: float64x2x2_t = vld2q_f64(pa0.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr0, a0a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi0, a0a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr0, a0a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi0, a0a.0);
                    let a0b: float64x2x2_t = vld2q_f64(pa0.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr0, a0b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi0, a0b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr0, a0b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi0, a0b.0);

                    let a1a: float64x2x2_t = vld2q_f64(pa1.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr1, a1a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi1, a1a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr1, a1a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi1, a1a.0);
                    let a1b: float64x2x2_t = vld2q_f64(pa1.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr1, a1b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi1, a1b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr1, a1b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi1, a1b.0);

                    let a2a: float64x2x2_t = vld2q_f64(pa2.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr2, a2a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi2, a2a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr2, a2a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi2, a2a.0);
                    let a2b: float64x2x2_t = vld2q_f64(pa2.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr2, a2b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi2, a2b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr2, a2b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi2, a2b.0);

                    let a3a: float64x2x2_t = vld2q_f64(pa3.add(i2));
                    y0a.0 = vfmaq_f64(y0a.0, xr3, a3a.0);
                    y0a.0 = vfmsq_f64(y0a.0, xi3, a3a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xr3, a3a.1);
                    y0a.1 = vfmaq_f64(y0a.1, xi3, a3a.0);
                    let a3b: float64x2x2_t = vld2q_f64(pa3.add(i2 + 4));
                    y0b.0 = vfmaq_f64(y0b.0, xr3, a3b.0);
                    y0b.0 = vfmsq_f64(y0b.0, xi3, a3b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xr3, a3b.1);
                    y0b.1 = vfmaq_f64(y0b.1, xi3, a3b.0);

                    vst2q_f64(yp0, y0a);
                    vst2q_f64(yp0b, y0b);

                    i2 += 8;
                }

                while i2 + 4 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float64x2x2_t = vld2q_f64(yp);

                    let a0: float64x2x2_t = vld2q_f64(pa0.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                    yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                    let a1: float64x2x2_t = vld2q_f64(pa1.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                    yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                    let a2: float64x2x2_t = vld2q_f64(pa2.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr2, a2.0);
                    yv.0 = vfmsq_f64(yv.0, xi2, a2.1);
                    yv.1 = vfmaq_f64(yv.1, xr2, a2.1);
                    yv.1 = vfmaq_f64(yv.1, xi2, a2.0);

                    let a3: float64x2x2_t = vld2q_f64(pa3.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr3, a3.0);
                    yv.0 = vfmsq_f64(yv.0, xi3, a3.1);
                    yv.1 = vfmaq_f64(yv.1, xr3, a3.1);
                    yv.1 = vfmaq_f64(yv.1, xi3, a3.0);

                    vst2q_f64(yp, yv);

                    i2 += 4;
                }

                while i2 < m2 {
                    let ar0 = *pa0.add(i2 + 0);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2 + 0);
                    let ai1 = *pa1.add(i2 + 1);
                    let ar2 = *pa2.add(i2 + 0);
                    let ai2 = *pa2.add(i2 + 1);
                    let ar3 = *pa3.add(i2 + 0);
                    let ai3 = *pa3.add(i2 + 1);

                    let xr0s = *xptr.add(2 * (j + 0) + 0);
                    let xi0s = *xptr.add(2 * (j + 0) + 1);
                    let xr1s = *xptr.add(2 * (j + 1) + 0);
                    let xi1s = *xptr.add(2 * (j + 1) + 1);
                    let xr2s = *xptr.add(2 * (j + 2) + 0);
                    let xi2s = *xptr.add(2 * (j + 2) + 1);
                    let xr3s = *xptr.add(2 * (j + 3) + 0);
                    let xi3s = *xptr.add(2 * (j + 3) + 1);

                    let yrp = yptr.add(i2 + 0);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    yr += xr0s * ar0 - xi0s * ai0;
                    yi += xr0s * ai0 + xi0s * ar0;
                    yr += xr1s * ar1 - xi1s * ai1;
                    yi += xr1s * ai1 + xi1s * ar1;
                    yr += xr2s * ar2 - xi2s * ai2;
                    yi += xr2s * ai2 + xi2s * ar2;
                    yr += xr3s * ar3 - xi3s * ai3;
                    yi += xr3s * ai3 + xi3s * ar3;

                    *yrp = yr;
                    *yip = yi;

                    i2 += 2;
                }

                j += 4;
            }

            while j + 2 <= n {
                let xr0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 0));
                let xi0 = vdupq_n_f64(*xptr.add(2 * (j + 0) + 1));
                let xr1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 0));
                let xi1 = vdupq_n_f64(*xptr.add(2 * (j + 1) + 1));

                let pa0 = aptr.add(lda2 * (j + 0));
                let pa1 = aptr.add(lda2 * (j + 1));

                let mut i2 = 0usize;

                while i2 + 4 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float64x2x2_t = vld2q_f64(yp);

                    let a0: float64x2x2_t = vld2q_f64(pa0.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr0, a0.0);
                    yv.0 = vfmsq_f64(yv.0, xi0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xr0, a0.1);
                    yv.1 = vfmaq_f64(yv.1, xi0, a0.0);

                    let a1: float64x2x2_t = vld2q_f64(pa1.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr1, a1.0);
                    yv.0 = vfmsq_f64(yv.0, xi1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xr1, a1.1);
                    yv.1 = vfmaq_f64(yv.1, xi1, a1.0);

                    vst2q_f64(yp, yv);

                    i2 += 4;
                }

                while i2 < m2 {
                    let ar0 = *pa0.add(i2 + 0);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2 + 0);
                    let ai1 = *pa1.add(i2 + 1);

                    let xr0s = *xptr.add(2 * (j + 0) + 0);
                    let xi0s = *xptr.add(2 * (j + 0) + 1);
                    let xr1s = *xptr.add(2 * (j + 1) + 0);
                    let xi1s = *xptr.add(2 * (j + 1) + 1);

                    let yrp = yptr.add(i2 + 0);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    yr += xr0s * ar0 - xi0s * ai0;
                    yi += xr0s * ai0 + xi0s * ar0;
                    yr += xr1s * ar1 - xi1s * ai1;
                    yi += xr1s * ai1 + xi1s * ar1;

                    *yrp = yr;
                    *yip = yi;

                    i2 += 2;
                }

                j += 2;
            }

            while j < n {
                let xr = vdupq_n_f64(*xptr.add(2 * j + 0));
                let xi = vdupq_n_f64(*xptr.add(2 * j + 1));

                let pac = aptr.add(lda2 * j);

                let mut i2 = 0usize;

                while i2 + 4 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float64x2x2_t = vld2q_f64(yp);

                    let av: float64x2x2_t = vld2q_f64(pac.add(i2));
                    yv.0 = vfmaq_f64(yv.0, xr, av.0);
                    yv.0 = vfmsq_f64(yv.0, xi, av.1);
                    yv.1 = vfmaq_f64(yv.1, xr, av.1);
                    yv.1 = vfmaq_f64(yv.1, xi, av.0);

                    vst2q_f64(yp, yv);

                    i2 += 4;
                }

                while i2 < m2 {
                    let ar = *pac.add(i2 + 0);
                    let ai = *pac.add(i2 + 1);

                    let yrp = yptr.add(i2 + 0);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    let xrs = *xptr.add(2 * j + 0);
                    let xis = *xptr.add(2 * j + 1);

                    yr += xrs * ar - xis * ai;
                    yi += xrs * ai + xis * ar;

                    *yrp = yr;
                    *yip = yi;

                    i2 += 2;
                }

                j += 1;
            }

            return;
        }

        let stepx = if incx > 0 { incx as usize } else { (-incx) as usize };
        let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx };

        for j in 0..n {
            let xr = *x.as_ptr().add(2 * ix + 0);
            let xi = *x.as_ptr().add(2 * ix + 1);
            if xr != 0.0 || xi != 0.0 {
                let col_ptr = a.as_ptr().add(2 * (j * lda));
                let col = core::slice::from_raw_parts(col_ptr, 2 * m);
                zaxpy(m, [xr, xi], col, 1, y, incy);
            }
            ix = if incx >= 0 { ix + stepx } else { ix.wrapping_sub(stepx) };
        }
    }
}

