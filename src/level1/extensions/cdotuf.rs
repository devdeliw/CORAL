use core::arch::aarch64::{vld2q_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vaddvq_f32, float32x4x2_t};
use crate::level1::assert_length_helpers::required_len_ok_cplx;

#[inline(always)]
pub fn cdotuf(
    m   : usize,
    n   : usize,
    a   : &[f32],
    lda : usize,
    x   : &[f32],
    incx: isize,
    out : &mut [f32],
) {
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0, "incx must be nonzero");
    debug_assert!(required_len_ok_cplx(x.len(), m, incx), "x too short for m/incx");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m");
        let need = 2 * ((n - 1).saturating_mul(lda) + m);
        debug_assert!(a.len() >= need, "A panel too small for m,n,lda");
        debug_assert!(out.len() >= 2*n, "out too small for n");
    }

    unsafe {
        let (x_ptr, x_contig): (*const f32, Option<Vec<f32>>) = if incx == 1 {
            (x.as_ptr(), None)
        } else {
            let mut xb = Vec::<f32>::with_capacity(2*m);
            xb.set_len(2*m);

            let mut pxs = x.as_ptr();
            let step = incx as usize;

            for i in 0..m {
                let p = 2*i;
                *xb.get_unchecked_mut(p + 0) = *pxs.add(0);
                *xb.get_unchecked_mut(p + 1) = *pxs.add(1);
                pxs = pxs.add(2*step);
            }

            (xb.as_ptr(), Some(xb))
        };

        let ap = a.as_ptr();

        let mut j = 0usize;
        while j + 4 <= n {
            let pa0 = ap.add(2 * (j + 0) * lda);
            let pa1 = ap.add(2 * (j + 1) * lda);
            let pa2 = ap.add(2 * (j + 2) * lda);
            let pa3 = ap.add(2 * (j + 3) * lda);

            let mut r0 = vdupq_n_f32(0.0);
            let mut i0 = vdupq_n_f32(0.0);
            let mut r1 = vdupq_n_f32(0.0);
            let mut i1 = vdupq_n_f32(0.0);
            let mut r2 = vdupq_n_f32(0.0);
            let mut i2 = vdupq_n_f32(0.0);
            let mut r3 = vdupq_n_f32(0.0);
            let mut i3 = vdupq_n_f32(0.0);

            let mut i = 0usize;

            while i + 8 <= m {
                let p0 = 2*i;

                let xv0: float32x4x2_t = vld2q_f32(x_ptr.add(p0));
                let xr0 = xv0.0;
                let xi0 = xv0.1;

                let a0_0: float32x4x2_t = vld2q_f32(pa0.add(p0));
                r0 = vfmaq_f32(r0, a0_0.0, xr0);
                r0 = vfmsq_f32(r0, a0_0.1, xi0);
                i0 = vfmaq_f32(i0, a0_0.0, xi0);
                i0 = vfmaq_f32(i0, a0_0.1, xr0);

                let a1_0: float32x4x2_t = vld2q_f32(pa1.add(p0));
                r1 = vfmaq_f32(r1, a1_0.0, xr0);
                r1 = vfmsq_f32(r1, a1_0.1, xi0);
                i1 = vfmaq_f32(i1, a1_0.0, xi0);
                i1 = vfmaq_f32(i1, a1_0.1, xr0);

                let a2_0: float32x4x2_t = vld2q_f32(pa2.add(p0));
                r2 = vfmaq_f32(r2, a2_0.0, xr0);
                r2 = vfmsq_f32(r2, a2_0.1, xi0);
                i2 = vfmaq_f32(i2, a2_0.0, xi0);
                i2 = vfmaq_f32(i2, a2_0.1, xr0);

                let a3_0: float32x4x2_t = vld2q_f32(pa3.add(p0));
                r3 = vfmaq_f32(r3, a3_0.0, xr0);
                r3 = vfmsq_f32(r3, a3_0.1, xi0);
                i3 = vfmaq_f32(i3, a3_0.0, xi0);
                i3 = vfmaq_f32(i3, a3_0.1, xr0);

                let p1 = p0 + 8;

                let xv1: float32x4x2_t = vld2q_f32(x_ptr.add(p1));
                let xr1v = xv1.0;
                let xi1v = xv1.1;

                let a0_1: float32x4x2_t = vld2q_f32(pa0.add(p1));
                r0 = vfmaq_f32(r0, a0_1.0, xr1v);
                r0 = vfmsq_f32(r0, a0_1.1, xi1v);
                i0 = vfmaq_f32(i0, a0_1.0, xi1v);
                i0 = vfmaq_f32(i0, a0_1.1, xr1v);

                let a1_1: float32x4x2_t = vld2q_f32(pa1.add(p1));
                r1 = vfmaq_f32(r1, a1_1.0, xr1v);
                r1 = vfmsq_f32(r1, a1_1.1, xi1v);
                i1 = vfmaq_f32(i1, a1_1.0, xi1v);
                i1 = vfmaq_f32(i1, a1_1.1, xr1v);

                let a2_1: float32x4x2_t = vld2q_f32(pa2.add(p1));
                r2 = vfmaq_f32(r2, a2_1.0, xr1v);
                r2 = vfmsq_f32(r2, a2_1.1, xi1v);
                i2 = vfmaq_f32(i2, a2_1.0, xi1v);
                i2 = vfmaq_f32(i2, a2_1.1, xr1v);

                let a3_1: float32x4x2_t = vld2q_f32(pa3.add(p1));
                r3 = vfmaq_f32(r3, a3_1.0, xr1v);
                r3 = vfmsq_f32(r3, a3_1.1, xi1v);
                i3 = vfmaq_f32(i3, a3_1.0, xi1v);
                i3 = vfmaq_f32(i3, a3_1.1, xr1v);

                i += 8;
            }

            while i + 4 <= m {
                let p = 2*i;

                let xv: float32x4x2_t = vld2q_f32(x_ptr.add(p));
                let xr = xv.0;
                let xi = xv.1;

                let a0: float32x4x2_t = vld2q_f32(pa0.add(p));
                r0 = vfmaq_f32(r0, a0.0, xr);
                r0 = vfmsq_f32(r0, a0.1, xi);
                i0 = vfmaq_f32(i0, a0.0, xi);
                i0 = vfmaq_f32(i0, a0.1, xr);

                let a1: float32x4x2_t = vld2q_f32(pa1.add(p));
                r1 = vfmaq_f32(r1, a1.0, xr);
                r1 = vfmsq_f32(r1, a1.1, xi);
                i1 = vfmaq_f32(i1, a1.0, xi);
                i1 = vfmaq_f32(i1, a1.1, xr);

                let a2: float32x4x2_t = vld2q_f32(pa2.add(p));
                r2 = vfmaq_f32(r2, a2.0, xr);
                r2 = vfmsq_f32(r2, a2.1, xi);
                i2 = vfmaq_f32(i2, a2.0, xi);
                i2 = vfmaq_f32(i2, a2.1, xr);

                let a3: float32x4x2_t = vld2q_f32(pa3.add(p));
                r3 = vfmaq_f32(r3, a3.0, xr);
                r3 = vfmsq_f32(r3, a3.1, xi);
                i3 = vfmaq_f32(i3, a3.0, xi);
                i3 = vfmaq_f32(i3, a3.1, xr);

                i += 4;
            }

            let mut s0r = vaddvq_f32(r0);
            let mut s0i = vaddvq_f32(i0);
            let mut s1r = vaddvq_f32(r1);
            let mut s1i = vaddvq_f32(i1);
            let mut s2r = vaddvq_f32(r2);
            let mut s2i = vaddvq_f32(i2);
            let mut s3r = vaddvq_f32(r3);
            let mut s3i = vaddvq_f32(i3);

            while i < m {
                let p = 2*i;

                let xr = *x_ptr.add(p + 0);
                let xi = *x_ptr.add(p + 1);

                s0r += *pa0.add(p + 0) * xr - *pa0.add(p + 1) * xi;
                s0i += *pa0.add(p + 0) * xi + *pa0.add(p + 1) * xr;

                s1r += *pa1.add(p + 0) * xr - *pa1.add(p + 1) * xi;
                s1i += *pa1.add(p + 0) * xi + *pa1.add(p + 1) * xr;

                s2r += *pa2.add(p + 0) * xr - *pa2.add(p + 1) * xi;
                s2i += *pa2.add(p + 0) * xi + *pa2.add(p + 1) * xr;

                s3r += *pa3.add(p + 0) * xr - *pa3.add(p + 1) * xi;
                s3i += *pa3.add(p + 0) * xi + *pa3.add(p + 1) * xr;

                i += 1;
            }

            *out.get_unchecked_mut(2*(j + 0) + 0) += s0r;
            *out.get_unchecked_mut(2*(j + 0) + 1) += s0i;
            *out.get_unchecked_mut(2*(j + 1) + 0) += s1r;
            *out.get_unchecked_mut(2*(j + 1) + 1) += s1i;
            *out.get_unchecked_mut(2*(j + 2) + 0) += s2r;
            *out.get_unchecked_mut(2*(j + 2) + 1) += s2i;
            *out.get_unchecked_mut(2*(j + 3) + 0) += s3r;
            *out.get_unchecked_mut(2*(j + 3) + 1) += s3i;

            j += 4;
        }

        match n - j {
            3 => {
                let pa0 = ap.add(2 * (j + 0) * lda);
                let pa1 = ap.add(2 * (j + 1) * lda);
                let pa2 = ap.add(2 * (j + 2) * lda);

                let mut r0 = vdupq_n_f32(0.0);
                let mut i0 = vdupq_n_f32(0.0);
                let mut r1 = vdupq_n_f32(0.0);
                let mut i1 = vdupq_n_f32(0.0);
                let mut r2 = vdupq_n_f32(0.0);
                let mut i2v = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 8 <= m {
                    let p0 = 2*i;

                    let xv0: float32x4x2_t = vld2q_f32(x_ptr.add(p0));
                    let xr0 = xv0.0;
                    let xi0 = xv0.1;

                    let a0: float32x4x2_t = vld2q_f32(pa0.add(p0));
                    r0 = vfmaq_f32(r0, a0.0, xr0);
                    r0 = vfmsq_f32(r0, a0.1, xi0);
                    i0 = vfmaq_f32(i0, a0.0, xi0);
                    i0 = vfmaq_f32(i0, a0.1, xr0);

                    let a1: float32x4x2_t = vld2q_f32(pa1.add(p0));
                    r1 = vfmaq_f32(r1, a1.0, xr0);
                    r1 = vfmsq_f32(r1, a1.1, xi0);
                    i1 = vfmaq_f32(i1, a1.0, xi0);
                    i1 = vfmaq_f32(i1, a1.1, xr0);

                    let a2: float32x4x2_t = vld2q_f32(pa2.add(p0));
                    r2 = vfmaq_f32(r2, a2.0, xr0);
                    r2 = vfmsq_f32(r2, a2.1, xi0);
                    i2v = vfmaq_f32(i2v, a2.0, xi0);
                    i2v = vfmaq_f32(i2v, a2.1, xr0);

                    let p1 = p0 + 8;

                    let xv1: float32x4x2_t = vld2q_f32(x_ptr.add(p1));
                    let xr1v = xv1.0;
                    let xi1v = xv1.1;

                    let b0: float32x4x2_t = vld2q_f32(pa0.add(p1));
                    r0 = vfmaq_f32(r0, b0.0, xr1v);
                    r0 = vfmsq_f32(r0, b0.1, xi1v);
                    i0 = vfmaq_f32(i0, b0.0, xi1v);
                    i0 = vfmaq_f32(i0, b0.1, xr1v);

                    let b1: float32x4x2_t = vld2q_f32(pa1.add(p1));
                    r1 = vfmaq_f32(r1, b1.0, xr1v);
                    r1 = vfmsq_f32(r1, b1.1, xi1v);
                    i1 = vfmaq_f32(i1, b1.0, xi1v);
                    i1 = vfmaq_f32(i1, b1.1, xr1v);

                    let b2: float32x4x2_t = vld2q_f32(pa2.add(p1));
                    r2 = vfmaq_f32(r2, b2.0, xr1v);
                    r2 = vfmsq_f32(r2, b2.1, xi1v);
                    i2v = vfmaq_f32(i2v, b2.0, xi1v);
                    i2v = vfmaq_f32(i2v, b2.1, xr1v);

                    i += 8;
                }
                while i + 4 <= m {
                    let p = 2*i;

                    let xv: float32x4x2_t = vld2q_f32(x_ptr.add(p));
                    let xr = xv.0;
                    let xi = xv.1;

                    let a0 = vld2q_f32(pa0.add(p));
                    r0 = vfmaq_f32(r0, a0.0, xr);
                    r0 = vfmsq_f32(r0, a0.1, xi);
                    i0 = vfmaq_f32(i0, a0.0, xi);
                    i0 = vfmaq_f32(i0, a0.1, xr);

                    let a1 = vld2q_f32(pa1.add(p));
                    r1 = vfmaq_f32(r1, a1.0, xr);
                    r1 = vfmsq_f32(r1, a1.1, xi);
                    i1 = vfmaq_f32(i1, a1.0, xi);
                    i1 = vfmaq_f32(i1, a1.1, xr);

                    let a2 = vld2q_f32(pa2.add(p));
                    r2 = vfmaq_f32(r2, a2.0, xr);
                    r2 = vfmsq_f32(r2, a2.1, xi);
                    i2v = vfmaq_f32(i2v, a2.0, xi);
                    i2v = vfmaq_f32(i2v, a2.1, xr);

                    i += 4;
                }
                let mut s0r = vaddvq_f32(r0);
                let mut s0i = vaddvq_f32(i0);
                let mut s1r = vaddvq_f32(r1);
                let mut s1i = vaddvq_f32(i1);
                let mut s2r = vaddvq_f32(r2);
                let mut s2i = vaddvq_f32(i2v);

                while i < m {
                    let p = 2*i;

                    let xr = *x_ptr.add(p + 0);
                    let xi = *x_ptr.add(p + 1);

                    s0r += *pa0.add(p + 0) * xr - *pa0.add(p + 1) * xi;
                    s0i += *pa0.add(p + 0) * xi + *pa0.add(p + 1) * xr;

                    s1r += *pa1.add(p + 0) * xr - *pa1.add(p + 1) * xi;
                    s1i += *pa1.add(p + 0) * xi + *pa1.add(p + 1) * xr;

                    s2r += *pa2.add(p + 0) * xr - *pa2.add(p + 1) * xi;
                    s2i += *pa2.add(p + 0) * xi + *pa2.add(p + 1) * xr;

                    i += 1;
                }

                *out.get_unchecked_mut(2*(j + 0) + 0) += s0r;
                *out.get_unchecked_mut(2*(j + 0) + 1) += s0i;
                *out.get_unchecked_mut(2*(j + 1) + 0) += s1r;
                *out.get_unchecked_mut(2*(j + 1) + 1) += s1i;
                *out.get_unchecked_mut(2*(j + 2) + 0) += s2r;
                *out.get_unchecked_mut(2*(j + 2) + 1) += s2i;
            }
            2 => {
                let pa0 = ap.add(2 * (j + 0) * lda);
                let pa1 = ap.add(2 * (j + 1) * lda);

                let mut r0 = vdupq_n_f32(0.0);
                let mut i0 = vdupq_n_f32(0.0);
                let mut r1 = vdupq_n_f32(0.0);
                let mut i1 = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 8 <= m {
                    let p0 = 2*i;

                    let xv0: float32x4x2_t = vld2q_f32(x_ptr.add(p0));
                    let xr0 = xv0.0;
                    let xi0 = xv0.1;

                    let a0: float32x4x2_t = vld2q_f32(pa0.add(p0));
                    r0 = vfmaq_f32(r0, a0.0, xr0);
                    r0 = vfmsq_f32(r0, a0.1, xi0);
                    i0 = vfmaq_f32(i0, a0.0, xi0);
                    i0 = vfmaq_f32(i0, a0.1, xr0);

                    let a1: float32x4x2_t = vld2q_f32(pa1.add(p0));
                    r1 = vfmaq_f32(r1, a1.0, xr0);
                    r1 = vfmsq_f32(r1, a1.1, xi0);
                    i1 = vfmaq_f32(i1, a1.0, xi0);
                    i1 = vfmaq_f32(i1, a1.1, xr0);

                    let p1 = p0 + 8;

                    let xv1: float32x4x2_t = vld2q_f32(x_ptr.add(p1));
                    let xr1v = xv1.0;
                    let xi1v = xv1.1;

                    let b0: float32x4x2_t = vld2q_f32(pa0.add(p1));
                    r0 = vfmaq_f32(r0, b0.0, xr1v);
                    r0 = vfmsq_f32(r0, b0.1, xi1v);
                    i0 = vfmaq_f32(i0, b0.0, xi1v);
                    i0 = vfmaq_f32(i0, b0.1, xr1v);

                    let b1: float32x4x2_t = vld2q_f32(pa1.add(p1));
                    r1 = vfmaq_f32(r1, b1.0, xr1v);
                    r1 = vfmsq_f32(r1, b1.1, xi1v);
                    i1 = vfmaq_f32(i1, b1.0, xi1v);
                    i1 = vfmaq_f32(i1, b1.1, xr1v);

                    i += 8;
                }
                while i + 4 <= m {
                    let p = 2*i;

                    let xv: float32x4x2_t = vld2q_f32(x_ptr.add(p));
                    let xr = xv.0;
                    let xi = xv.1;

                    let a0 = vld2q_f32(pa0.add(p));
                    r0 = vfmaq_f32(r0, a0.0, xr);
                    r0 = vfmsq_f32(r0, a0.1, xi);
                    i0 = vfmaq_f32(i0, a0.0, xi);
                    i0 = vfmaq_f32(i0, a0.1, xr);

                    let a1 = vld2q_f32(pa1.add(p));
                    r1 = vfmaq_f32(r1, a1.0, xr);
                    r1 = vfmsq_f32(r1, a1.1, xi);
                    i1 = vfmaq_f32(i1, a1.0, xi);
                    i1 = vfmaq_f32(i1, a1.1, xr);

                    i += 4;
                }
                let mut s0r = vaddvq_f32(r0);
                let mut s0i = vaddvq_f32(i0);
                let mut s1r = vaddvq_f32(r1);
                let mut s1i = vaddvq_f32(i1);

                while i < m {
                    let p = 2*i;

                    let xr = *x_ptr.add(p + 0);
                    let xi = *x_ptr.add(p + 1);

                    s0r += *pa0.add(p + 0) * xr - *pa0.add(p + 1) * xi;
                    s0i += *pa0.add(p + 0) * xi + *pa0.add(p + 1) * xr;

                    s1r += *pa1.add(p + 0) * xr - *pa1.add(p + 1) * xi;
                    s1i += *pa1.add(p + 0) * xi + *pa1.add(p + 1) * xr;

                    i += 1;
                }

                *out.get_unchecked_mut(2*(j + 0) + 0) += s0r;
                *out.get_unchecked_mut(2*(j + 0) + 1) += s0i;
                *out.get_unchecked_mut(2*(j + 1) + 0) += s1r;
                *out.get_unchecked_mut(2*(j + 1) + 1) += s1i;
            }
            1 => {
                let pa0 = ap.add(2 * (j + 0) * lda);

                let mut r0 = vdupq_n_f32(0.0);
                let mut i0 = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 8 <= m {
                    let p0 = 2*i;

                    let xv0: float32x4x2_t = vld2q_f32(x_ptr.add(p0));
                    let xr0 = xv0.0;
                    let xi0 = xv0.1;

                    let a0: float32x4x2_t = vld2q_f32(pa0.add(p0));
                    r0 = vfmaq_f32(r0, a0.0, xr0);
                    r0 = vfmsq_f32(r0, a0.1, xi0);
                    i0 = vfmaq_f32(i0, a0.0, xi0);
                    i0 = vfmaq_f32(i0, a0.1, xr0);

                    let p1 = p0 + 8;

                    let xv1: float32x4x2_t = vld2q_f32(x_ptr.add(p1));
                    let xr1v = xv1.0;
                    let xi1v = xv1.1;

                    let b0: float32x4x2_t = vld2q_f32(pa0.add(p1));
                    r0 = vfmaq_f32(r0, b0.0, xr1v);
                    r0 = vfmsq_f32(r0, b0.1, xi1v);
                    i0 = vfmaq_f32(i0, b0.0, xi1v);
                    i0 = vfmaq_f32(i0, b0.1, xr1v);

                    i += 8;
                }
                while i + 4 <= m {
                    let p = 2*i;

                    let xv: float32x4x2_t = vld2q_f32(x_ptr.add(p));
                    let xr = xv.0;
                    let xi = xv.1;

                    let a0 = vld2q_f32(pa0.add(p));
                    r0 = vfmaq_f32(r0, a0.0, xr);
                    r0 = vfmsq_f32(r0, a0.1, xi);
                    i0 = vfmaq_f32(i0, a0.0, xi);
                    i0 = vfmaq_f32(i0, a0.1, xr);

                    i += 4;
                }
                let mut s0r = vaddvq_f32(r0);
                let mut s0i = vaddvq_f32(i0);

                while i < m {
                    let p = 2*i;

                    let xr = *x_ptr.add(p + 0);
                    let xi = *x_ptr.add(p + 1);

                    s0r += *pa0.add(p + 0) * xr - *pa0.add(p + 1) * xi;
                    s0i += *pa0.add(p + 0) * xi + *pa0.add(p + 1) * xr;

                    i += 1;
                }

                *out.get_unchecked_mut(2*(j + 0) + 0) += s0r;
                *out.get_unchecked_mut(2*(j + 0) + 1) += s0i;
            }
            _ => {}
        }

        core::mem::forget(x_contig);
    }
}

