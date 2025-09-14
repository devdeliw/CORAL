use core::arch::aarch64::*;
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level1::caxpy::caxpy;

#[inline(always)]
pub fn caxpyf(
    m    : usize,
    n    : usize,
    x    : &[f32],
    incx : isize,
    a    : &[f32],
    lda  : usize,
    y    : &mut [f32],
    incy : isize,
) {
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), m, incy), "y too short for m/incy");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m (in complexes)");
        let need = 2*((n - 1).saturating_mul(lda) + m);
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
                let xr0 = vdupq_n_f32(*xptr.add(2*(j+0)+0));
                let xi0 = vdupq_n_f32(*xptr.add(2*(j+0)+1));
                let xr1 = vdupq_n_f32(*xptr.add(2*(j+1)+0));
                let xi1 = vdupq_n_f32(*xptr.add(2*(j+1)+1));
                let xr2 = vdupq_n_f32(*xptr.add(2*(j+2)+0));
                let xi2 = vdupq_n_f32(*xptr.add(2*(j+2)+1));
                let xr3 = vdupq_n_f32(*xptr.add(2*(j+3)+0));
                let xi3 = vdupq_n_f32(*xptr.add(2*(j+3)+1));
                let xr4 = vdupq_n_f32(*xptr.add(2*(j+4)+0));
                let xi4 = vdupq_n_f32(*xptr.add(2*(j+4)+1));
                let xr5 = vdupq_n_f32(*xptr.add(2*(j+5)+0));
                let xi5 = vdupq_n_f32(*xptr.add(2*(j+5)+1));
                let xr6 = vdupq_n_f32(*xptr.add(2*(j+6)+0));
                let xi6 = vdupq_n_f32(*xptr.add(2*(j+6)+1));
                let xr7 = vdupq_n_f32(*xptr.add(2*(j+7)+0));
                let xi7 = vdupq_n_f32(*xptr.add(2*(j+7)+1));

                let pa0 = aptr.add(lda2*(j+0));
                let pa1 = aptr.add(lda2*(j+1));
                let pa2 = aptr.add(lda2*(j+2));
                let pa3 = aptr.add(lda2*(j+3));
                let pa4 = aptr.add(lda2*(j+4));
                let pa5 = aptr.add(lda2*(j+5));
                let pa6 = aptr.add(lda2*(j+6));
                let pa7 = aptr.add(lda2*(j+7));

                let mut i2 = 0usize;

                while i2 + 16 <= m2 {
                    let yp0 = yptr.add(i2);
                    let mut y0: float32x4x2_t = vld2q_f32(yp0);

                    let a0 = vld2q_f32(pa0.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr0, a0.0);
                    y0.0 = vfmsq_f32(y0.0, xi0, a0.1);
                    y0.1 = vfmaq_f32(y0.1, xr0, a0.1);
                    y0.1 = vfmaq_f32(y0.1, xi0, a0.0);

                    let a1 = vld2q_f32(pa1.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr1, a1.0);
                    y0.0 = vfmsq_f32(y0.0, xi1, a1.1);
                    y0.1 = vfmaq_f32(y0.1, xr1, a1.1);
                    y0.1 = vfmaq_f32(y0.1, xi1, a1.0);

                    let a2 = vld2q_f32(pa2.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr2, a2.0);
                    y0.0 = vfmsq_f32(y0.0, xi2, a2.1);
                    y0.1 = vfmaq_f32(y0.1, xr2, a2.1);
                    y0.1 = vfmaq_f32(y0.1, xi2, a2.0);

                    let a3 = vld2q_f32(pa3.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr3, a3.0);
                    y0.0 = vfmsq_f32(y0.0, xi3, a3.1);
                    y0.1 = vfmaq_f32(y0.1, xr3, a3.1);
                    y0.1 = vfmaq_f32(y0.1, xi3, a3.0);

                    let a4 = vld2q_f32(pa4.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr4, a4.0);
                    y0.0 = vfmsq_f32(y0.0, xi4, a4.1);
                    y0.1 = vfmaq_f32(y0.1, xr4, a4.1);
                    y0.1 = vfmaq_f32(y0.1, xi4, a4.0);

                    let a5 = vld2q_f32(pa5.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr5, a5.0);
                    y0.0 = vfmsq_f32(y0.0, xi5, a5.1);
                    y0.1 = vfmaq_f32(y0.1, xr5, a5.1);
                    y0.1 = vfmaq_f32(y0.1, xi5, a5.0);

                    let a6 = vld2q_f32(pa6.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr6, a6.0);
                    y0.0 = vfmsq_f32(y0.0, xi6, a6.1);
                    y0.1 = vfmaq_f32(y0.1, xr6, a6.1);
                    y0.1 = vfmaq_f32(y0.1, xi6, a6.0);

                    let a7 = vld2q_f32(pa7.add(i2));
                    y0.0 = vfmaq_f32(y0.0, xr7, a7.0);
                    y0.0 = vfmsq_f32(y0.0, xi7, a7.1);
                    y0.1 = vfmaq_f32(y0.1, xr7, a7.1);
                    y0.1 = vfmaq_f32(y0.1, xi7, a7.0);

                    vst2q_f32(yp0, y0);

                    let i2b = i2 + 8;
                    let yp1 = yptr.add(i2b);
                    let mut y1: float32x4x2_t = vld2q_f32(yp1);

                    let b0 = vld2q_f32(pa0.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr0, b0.0);
                    y1.0 = vfmsq_f32(y1.0, xi0, b0.1);
                    y1.1 = vfmaq_f32(y1.1, xr0, b0.1);
                    y1.1 = vfmaq_f32(y1.1, xi0, b0.0);

                    let b1 = vld2q_f32(pa1.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr1, b1.0);
                    y1.0 = vfmsq_f32(y1.0, xi1, b1.1);
                    y1.1 = vfmaq_f32(y1.1, xr1, b1.1);
                    y1.1 = vfmaq_f32(y1.1, xi1, b1.0);

                    let b2 = vld2q_f32(pa2.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr2, b2.0);
                    y1.0 = vfmsq_f32(y1.0, xi2, b2.1);
                    y1.1 = vfmaq_f32(y1.1, xr2, b2.1);
                    y1.1 = vfmaq_f32(y1.1, xi2, b2.0);

                    let b3 = vld2q_f32(pa3.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr3, b3.0);
                    y1.0 = vfmsq_f32(y1.0, xi3, b3.1);
                    y1.1 = vfmaq_f32(y1.1, xr3, b3.1);
                    y1.1 = vfmaq_f32(y1.1, xi3, b3.0);

                    let b4 = vld2q_f32(pa4.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr4, b4.0);
                    y1.0 = vfmsq_f32(y1.0, xi4, b4.1);
                    y1.1 = vfmaq_f32(y1.1, xr4, b4.1);
                    y1.1 = vfmaq_f32(y1.1, xi4, b4.0);

                    let b5 = vld2q_f32(pa5.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr5, b5.0);
                    y1.0 = vfmsq_f32(y1.0, xi5, b5.1);
                    y1.1 = vfmaq_f32(y1.1, xr5, b5.1);
                    y1.1 = vfmaq_f32(y1.1, xi5, b5.0);

                    let b6 = vld2q_f32(pa6.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr6, b6.0);
                    y1.0 = vfmsq_f32(y1.0, xi6, b6.1);
                    y1.1 = vfmaq_f32(y1.1, xr6, b6.1);
                    y1.1 = vfmaq_f32(y1.1, xi6, b6.0);

                    let b7 = vld2q_f32(pa7.add(i2b));
                    y1.0 = vfmaq_f32(y1.0, xr7, b7.0);
                    y1.0 = vfmsq_f32(y1.0, xi7, b7.1);
                    y1.1 = vfmaq_f32(y1.1, xr7, b7.1);
                    y1.1 = vfmaq_f32(y1.1, xi7, b7.0);

                    vst2q_f32(yp1, y1);

                    i2 += 16;
                }

                while i2 + 8 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float32x4x2_t = vld2q_f32(yp);

                    let av0 = vld2q_f32(pa0.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr0, av0.0);
                    yv.0 = vfmsq_f32(yv.0, xi0, av0.1);
                    yv.1 = vfmaq_f32(yv.1, xr0, av0.1);
                    yv.1 = vfmaq_f32(yv.1, xi0, av0.0);

                    let av1 = vld2q_f32(pa1.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr1, av1.0);
                    yv.0 = vfmsq_f32(yv.0, xi1, av1.1);
                    yv.1 = vfmaq_f32(yv.1, xr1, av1.1);
                    yv.1 = vfmaq_f32(yv.1, xi1, av1.0);

                    let av2 = vld2q_f32(pa2.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr2, av2.0);
                    yv.0 = vfmsq_f32(yv.0, xi2, av2.1);
                    yv.1 = vfmaq_f32(yv.1, xr2, av2.1);
                    yv.1 = vfmaq_f32(yv.1, xi2, av2.0);

                    let av3 = vld2q_f32(pa3.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr3, av3.0);
                    yv.0 = vfmsq_f32(yv.0, xi3, av3.1);
                    yv.1 = vfmaq_f32(yv.1, xr3, av3.1);
                    yv.1 = vfmaq_f32(yv.1, xi3, av3.0);

                    let av4 = vld2q_f32(pa4.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr4, av4.0);
                    yv.0 = vfmsq_f32(yv.0, xi4, av4.1);
                    yv.1 = vfmaq_f32(yv.1, xr4, av4.1);
                    yv.1 = vfmaq_f32(yv.1, xi4, av4.0);

                    let av5 = vld2q_f32(pa5.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr5, av5.0);
                    yv.0 = vfmsq_f32(yv.0, xi5, av5.1);
                    yv.1 = vfmaq_f32(yv.1, xr5, av5.1);
                    yv.1 = vfmaq_f32(yv.1, xi5, av5.0);

                    let av6 = vld2q_f32(pa6.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr6, av6.0);
                    yv.0 = vfmsq_f32(yv.0, xi6, av6.1);
                    yv.1 = vfmaq_f32(yv.1, xr6, av6.1);
                    yv.1 = vfmaq_f32(yv.1, xi6, av6.0);

                    let av7 = vld2q_f32(pa7.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr7, av7.0);
                    yv.0 = vfmsq_f32(yv.0, xi7, av7.1);
                    yv.1 = vfmaq_f32(yv.1, xr7, av7.1);
                    yv.1 = vfmaq_f32(yv.1, xi7, av7.0);

                    vst2q_f32(yp, yv);

                    i2 += 8;
                }

                while i2 < m2 {
                    let ar0 = *pa0.add(i2);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2);
                    let ai1 = *pa1.add(i2 + 1);
                    let ar2 = *pa2.add(i2);
                    let ai2 = *pa2.add(i2 + 1);
                    let ar3 = *pa3.add(i2);
                    let ai3 = *pa3.add(i2 + 1);
                    let ar4 = *pa4.add(i2);
                    let ai4 = *pa4.add(i2 + 1);
                    let ar5 = *pa5.add(i2);
                    let ai5 = *pa5.add(i2 + 1);
                    let ar6 = *pa6.add(i2);
                    let ai6 = *pa6.add(i2 + 1);
                    let ar7 = *pa7.add(i2);
                    let ai7 = *pa7.add(i2 + 1);

                    let xr0s = *xptr.add(2*(j+0)+0);
                    let xi0s = *xptr.add(2*(j+0)+1);
                    let xr1s = *xptr.add(2*(j+1)+0);
                    let xi1s = *xptr.add(2*(j+1)+1);
                    let xr2s = *xptr.add(2*(j+2)+0);
                    let xi2s = *xptr.add(2*(j+2)+1);
                    let xr3s = *xptr.add(2*(j+3)+0);
                    let xi3s = *xptr.add(2*(j+3)+1);
                    let xr4s = *xptr.add(2*(j+4)+0);
                    let xi4s = *xptr.add(2*(j+4)+1);
                    let xr5s = *xptr.add(2*(j+5)+0);
                    let xi5s = *xptr.add(2*(j+5)+1);
                    let xr6s = *xptr.add(2*(j+6)+0);
                    let xi6s = *xptr.add(2*(j+6)+1);
                    let xr7s = *xptr.add(2*(j+7)+0);
                    let xi7s = *xptr.add(2*(j+7)+1);

                    let yrp = yptr.add(i2);
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
                let xr0 = vdupq_n_f32(*xptr.add(2*(j+0)+0));
                let xi0 = vdupq_n_f32(*xptr.add(2*(j+0)+1));
                let xr1 = vdupq_n_f32(*xptr.add(2*(j+1)+0));
                let xi1 = vdupq_n_f32(*xptr.add(2*(j+1)+1));
                let xr2 = vdupq_n_f32(*xptr.add(2*(j+2)+0));
                let xi2 = vdupq_n_f32(*xptr.add(2*(j+2)+1));
                let xr3 = vdupq_n_f32(*xptr.add(2*(j+3)+0));
                let xi3 = vdupq_n_f32(*xptr.add(2*(j+3)+1));

                let pa0 = aptr.add(lda2*(j+0));
                let pa1 = aptr.add(lda2*(j+1));
                let pa2 = aptr.add(lda2*(j+2));
                let pa3 = aptr.add(lda2*(j+3));

                let mut i2 = 0usize;

                while i2 + 8 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float32x4x2_t = vld2q_f32(yp);

                    let a0 = vld2q_f32(pa0.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr0, a0.0);
                    yv.0 = vfmsq_f32(yv.0, xi0, a0.1);
                    yv.1 = vfmaq_f32(yv.1, xr0, a0.1);
                    yv.1 = vfmaq_f32(yv.1, xi0, a0.0);

                    let a1 = vld2q_f32(pa1.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr1, a1.0);
                    yv.0 = vfmsq_f32(yv.0, xi1, a1.1);
                    yv.1 = vfmaq_f32(yv.1, xr1, a1.1);
                    yv.1 = vfmaq_f32(yv.1, xi1, a1.0);

                    let a2 = vld2q_f32(pa2.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr2, a2.0);
                    yv.0 = vfmsq_f32(yv.0, xi2, a2.1);
                    yv.1 = vfmaq_f32(yv.1, xr2, a2.1);
                    yv.1 = vfmaq_f32(yv.1, xi2, a2.0);

                    let a3 = vld2q_f32(pa3.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr3, a3.0);
                    yv.0 = vfmsq_f32(yv.0, xi3, a3.1);
                    yv.1 = vfmaq_f32(yv.1, xr3, a3.1);
                    yv.1 = vfmaq_f32(yv.1, xi3, a3.0);

                    vst2q_f32(yp, yv);

                    i2 += 8;
                }

                while i2 < m2 {
                    let yrp = yptr.add(i2);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    let ar0 = *pa0.add(i2);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2);
                    let ai1 = *pa1.add(i2 + 1);
                    let ar2 = *pa2.add(i2);
                    let ai2 = *pa2.add(i2 + 1);
                    let ar3 = *pa3.add(i2);
                    let ai3 = *pa3.add(i2 + 1);

                    let xr0s = *xptr.add(2*(j+0)+0);
                    let xi0s = *xptr.add(2*(j+0)+1);
                    let xr1s = *xptr.add(2*(j+1)+0);
                    let xi1s = *xptr.add(2*(j+1)+1);
                    let xr2s = *xptr.add(2*(j+2)+0);
                    let xi2s = *xptr.add(2*(j+2)+1);
                    let xr3s = *xptr.add(2*(j+3)+0);
                    let xi3s = *xptr.add(2*(j+3)+1);

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
                let xr0 = vdupq_n_f32(*xptr.add(2*(j+0)+0));
                let xi0 = vdupq_n_f32(*xptr.add(2*(j+0)+1));
                let xr1 = vdupq_n_f32(*xptr.add(2*(j+1)+0));
                let xi1 = vdupq_n_f32(*xptr.add(2*(j+1)+1));

                let pa0 = aptr.add(lda2*(j+0));
                let pa1 = aptr.add(lda2*(j+1));

                let mut i2 = 0usize;

                while i2 + 8 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float32x4x2_t = vld2q_f32(yp);

                    let a0 = vld2q_f32(pa0.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr0, a0.0);
                    yv.0 = vfmsq_f32(yv.0, xi0, a0.1);
                    yv.1 = vfmaq_f32(yv.1, xr0, a0.1);
                    yv.1 = vfmaq_f32(yv.1, xi0, a0.0);

                    let a1 = vld2q_f32(pa1.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr1, a1.0);
                    yv.0 = vfmsq_f32(yv.0, xi1, a1.1);
                    yv.1 = vfmaq_f32(yv.1, xr1, a1.1);
                    yv.1 = vfmaq_f32(yv.1, xi1, a1.0);

                    vst2q_f32(yp, yv);

                    i2 += 8;
                }

                while i2 < m2 {
                    let yrp = yptr.add(i2);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    let ar0 = *pa0.add(i2);
                    let ai0 = *pa0.add(i2 + 1);
                    let ar1 = *pa1.add(i2);
                    let ai1 = *pa1.add(i2 + 1);

                    let xr0s = *xptr.add(2*(j+0)+0);
                    let xi0s = *xptr.add(2*(j+0)+1);
                    let xr1s = *xptr.add(2*(j+1)+0);
                    let xi1s = *xptr.add(2*(j+1)+1);

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
                let xr = vdupq_n_f32(*xptr.add(2*j+0));
                let xi = vdupq_n_f32(*xptr.add(2*j+1));

                let pac = aptr.add(lda2*j);

                let mut i2 = 0usize;

                while i2 + 8 <= m2 {
                    let yp = yptr.add(i2);
                    let mut yv: float32x4x2_t = vld2q_f32(yp);

                    let av = vld2q_f32(pac.add(i2));
                    yv.0 = vfmaq_f32(yv.0, xr, av.0);
                    yv.0 = vfmsq_f32(yv.0, xi, av.1);
                    yv.1 = vfmaq_f32(yv.1, xr, av.1);
                    yv.1 = vfmaq_f32(yv.1, xi, av.0);

                    vst2q_f32(yp, yv);

                    i2 += 8;
                }

                while i2 < m2 {
                    let ar = *pac.add(i2);
                    let ai = *pac.add(i2 + 1);

                    let yrp = yptr.add(i2);
                    let yip = yptr.add(i2 + 1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    let xrs = *xptr.add(2*j+0);
                    let xis = *xptr.add(2*j+1);

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
            let xr = *x.as_ptr().add(2*ix + 0);
            let xi = *x.as_ptr().add(2*ix + 1);
            if xr != 0.0 || xi != 0.0 {
                let col_ptr = a.as_ptr().add(2*(j*lda));
                let col = core::slice::from_raw_parts(col_ptr, 2*m);
                caxpy(m, [xr, xi], col, 1, y, incy);
            }
            ix = if incx >= 0 { ix + stepx } else { ix.wrapping_sub(stepx) };
        }
    }
}

