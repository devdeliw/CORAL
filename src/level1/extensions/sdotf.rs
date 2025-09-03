use core::arch::aarch64::{vld1q_f32, vdupq_n_f32, vfmaq_f32, vaddvq_f32};
use crate::level1::assert_length_helpers::required_len_ok;

#[inline(always)]
pub fn sdotf(
    m   : usize,
    n   : usize,      
    a   : &[f32],     
    lda : usize,
    x   : &[f32],
    incx: isize,
    out : &mut [f32], 
) {
    // quick return
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0, "incx must be nonzero");
    debug_assert!(required_len_ok(x.len(), m, incx), "x too short for m/incx");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m");
        debug_assert!(
            a.len() >= (n - 1).saturating_mul(lda) + m,
            "A panel too small for m,n,lda"
        );
        debug_assert!(out.len() >= n, "out too small for n");
    }

    unsafe {
        let (x_ptr, x_contig): (*const f32, Option<Vec<f32>>) = if incx == 1 {
            (x.as_ptr(), None)
        } else {
            let mut xb = Vec::<f32>::with_capacity(m);
            xb.set_len(m);

            let mut pxs = x.as_ptr();
            let step    = incx as usize;

            for i in 0..m {
                *xb.get_unchecked_mut(i) = *pxs;
                pxs = pxs.add(step);
            }

            (xb.as_ptr(), Some(xb))
        };

        let ap = a.as_ptr();

        let mut j = 0usize;
        while j + 4 <= n {
            let pa0 = ap.add((j + 0) * lda);
            let pa1 = ap.add((j + 1) * lda);
            let pa2 = ap.add((j + 2) * lda);
            let pa3 = ap.add((j + 3) * lda);

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0usize;

            while i + 16 <= m {
                let x0 = vld1q_f32(x_ptr.add(i + 0));
                let x1 = vld1q_f32(x_ptr.add(i + 4));
                let x2 = vld1q_f32(x_ptr.add(i + 8));
                let x3 = vld1q_f32(x_ptr.add(i + 12));

                let a00 = vld1q_f32(pa0.add(i + 0));
                let a01 = vld1q_f32(pa0.add(i + 4));
                let a02 = vld1q_f32(pa0.add(i + 8));
                let a03 = vld1q_f32(pa0.add(i + 12));
                acc0 = vfmaq_f32(acc0, a00, x0);
                acc0 = vfmaq_f32(acc0, a01, x1);
                acc0 = vfmaq_f32(acc0, a02, x2);
                acc0 = vfmaq_f32(acc0, a03, x3);

                let b00 = vld1q_f32(pa1.add(i + 0));
                let b01 = vld1q_f32(pa1.add(i + 4));
                let b02 = vld1q_f32(pa1.add(i + 8));
                let b03 = vld1q_f32(pa1.add(i + 12));
                acc1 = vfmaq_f32(acc1, b00, x0);
                acc1 = vfmaq_f32(acc1, b01, x1);
                acc1 = vfmaq_f32(acc1, b02, x2);
                acc1 = vfmaq_f32(acc1, b03, x3);

                let c00 = vld1q_f32(pa2.add(i + 0));
                let c01 = vld1q_f32(pa2.add(i + 4));
                let c02 = vld1q_f32(pa2.add(i + 8));
                let c03 = vld1q_f32(pa2.add(i + 12));
                acc2 = vfmaq_f32(acc2, c00, x0);
                acc2 = vfmaq_f32(acc2, c01, x1);
                acc2 = vfmaq_f32(acc2, c02, x2);
                acc2 = vfmaq_f32(acc2, c03, x3);

                let d00 = vld1q_f32(pa3.add(i + 0));
                let d01 = vld1q_f32(pa3.add(i + 4));
                let d02 = vld1q_f32(pa3.add(i + 8));
                let d03 = vld1q_f32(pa3.add(i + 12));
                acc3 = vfmaq_f32(acc3, d00, x0);
                acc3 = vfmaq_f32(acc3, d01, x1);
                acc3 = vfmaq_f32(acc3, d02, x2);
                acc3 = vfmaq_f32(acc3, d03, x3);

                i += 16;
            }

            while i + 8 <= m {
                let x0 = vld1q_f32(x_ptr.add(i + 0));
                let x1 = vld1q_f32(x_ptr.add(i + 4));

                let a00 = vld1q_f32(pa0.add(i + 0));
                let a01 = vld1q_f32(pa0.add(i + 4));
                acc0 = vfmaq_f32(acc0, a00, x0);
                acc0 = vfmaq_f32(acc0, a01, x1);

                let b00 = vld1q_f32(pa1.add(i + 0));
                let b01 = vld1q_f32(pa1.add(i + 4));
                acc1 = vfmaq_f32(acc1, b00, x0);
                acc1 = vfmaq_f32(acc1, b01, x1);

                let c00 = vld1q_f32(pa2.add(i + 0));
                let c01 = vld1q_f32(pa2.add(i + 4));
                acc2 = vfmaq_f32(acc2, c00, x0);
                acc2 = vfmaq_f32(acc2, c01, x1);

                let d00 = vld1q_f32(pa3.add(i + 0));
                let d01 = vld1q_f32(pa3.add(i + 4));
                acc3 = vfmaq_f32(acc3, d00, x0);
                acc3 = vfmaq_f32(acc3, d01, x1);

                i += 8;
            }

            while i + 4 <= m {
                let x0 = vld1q_f32(x_ptr.add(i + 0));

                let a00 = vld1q_f32(pa0.add(i + 0));
                let b00 = vld1q_f32(pa1.add(i + 0));
                let c00 = vld1q_f32(pa2.add(i + 0));
                let d00 = vld1q_f32(pa3.add(i + 0));
                acc0 = vfmaq_f32(acc0, a00, x0);
                acc1 = vfmaq_f32(acc1, b00, x0);
                acc2 = vfmaq_f32(acc2, c00, x0);
                acc3 = vfmaq_f32(acc3, d00, x0);

                i += 4;
            }

            let mut s0 = vaddvq_f32(acc0);
            let mut s1 = vaddvq_f32(acc1);
            let mut s2 = vaddvq_f32(acc2);
            let mut s3 = vaddvq_f32(acc3);

            while i < m {
                let xi = *x_ptr.add(i);
                s0 += *pa0.add(i) * xi;
                s1 += *pa1.add(i) * xi;
                s2 += *pa2.add(i) * xi;
                s3 += *pa3.add(i) * xi;
                i += 1;
            }

            *out.get_unchecked_mut(j + 0) += s0;
            *out.get_unchecked_mut(j + 1) += s1;
            *out.get_unchecked_mut(j + 2) += s2;
            *out.get_unchecked_mut(j + 3) += s3;

            j += 4;
        }

        match n - j {
            3 => {
                let pa0 = ap.add((j + 0) * lda);
                let pa1 = ap.add((j + 1) * lda);
                let pa2 = ap.add((j + 2) * lda);

                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 16 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));
                    let x2 = vld1q_f32(x_ptr.add(i + 8));
                    let x3 = vld1q_f32(x_ptr.add(i + 12));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    let a02 = vld1q_f32(pa0.add(i + 8));
                    let a03 = vld1q_f32(pa0.add(i + 12));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);
                    acc0 = vfmaq_f32(acc0, a02, x2);
                    acc0 = vfmaq_f32(acc0, a03, x3);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    let b02 = vld1q_f32(pa1.add(i + 8));
                    let b03 = vld1q_f32(pa1.add(i + 12));
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    acc1 = vfmaq_f32(acc1, b01, x1);
                    acc1 = vfmaq_f32(acc1, b02, x2);
                    acc1 = vfmaq_f32(acc1, b03, x3);

                    let c00 = vld1q_f32(pa2.add(i + 0));
                    let c01 = vld1q_f32(pa2.add(i + 4));
                    let c02 = vld1q_f32(pa2.add(i + 8));
                    let c03 = vld1q_f32(pa2.add(i + 12));
                    acc2 = vfmaq_f32(acc2, c00, x0);
                    acc2 = vfmaq_f32(acc2, c01, x1);
                    acc2 = vfmaq_f32(acc2, c02, x2);
                    acc2 = vfmaq_f32(acc2, c03, x3);

                    i += 16;
                }
                while i + 8 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    acc1 = vfmaq_f32(acc1, b01, x1);

                    let c00 = vld1q_f32(pa2.add(i + 0));
                    let c01 = vld1q_f32(pa2.add(i + 4));
                    acc2 = vfmaq_f32(acc2, c00, x0);
                    acc2 = vfmaq_f32(acc2, c01, x1);

                    i += 8;
                }
                while i + 4 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let c00 = vld1q_f32(pa2.add(i + 0));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    acc2 = vfmaq_f32(acc2, c00, x0);
                    i += 4;
                }
                let mut s0 = vaddvq_f32(acc0);
                let mut s1 = vaddvq_f32(acc1);
                let mut s2 = vaddvq_f32(acc2);
                while i < m {
                    let xi = *x_ptr.add(i);
                    s0 += *pa0.add(i) * xi;
                    s1 += *pa1.add(i) * xi;
                    s2 += *pa2.add(i) * xi;
                    i += 1;
                }
                *out.get_unchecked_mut(j + 0) += s0;
                *out.get_unchecked_mut(j + 1) += s1;
                *out.get_unchecked_mut(j + 2) += s2;
            }
            2 => {
                let pa0 = ap.add((j + 0) * lda);
                let pa1 = ap.add((j + 1) * lda);

                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 16 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));
                    let x2 = vld1q_f32(x_ptr.add(i + 8));
                    let x3 = vld1q_f32(x_ptr.add(i + 12));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    let a02 = vld1q_f32(pa0.add(i + 8));
                    let a03 = vld1q_f32(pa0.add(i + 12));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);
                    acc0 = vfmaq_f32(acc0, a02, x2);
                    acc0 = vfmaq_f32(acc0, a03, x3);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    let b02 = vld1q_f32(pa1.add(i + 8));
                    let b03 = vld1q_f32(pa1.add(i + 12));
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    acc1 = vfmaq_f32(acc1, b01, x1);
                    acc1 = vfmaq_f32(acc1, b02, x2);
                    acc1 = vfmaq_f32(acc1, b03, x3);

                    i += 16;
                }
                while i + 8 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    acc1 = vfmaq_f32(acc1, b01, x1);

                    i += 8;
                }
                while i + 4 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let b00 = vld1q_f32(pa1.add(i + 0));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc1 = vfmaq_f32(acc1, b00, x0);
                    i += 4;
                }
                let mut s0 = vaddvq_f32(acc0);
                let mut s1 = vaddvq_f32(acc1);
                while i < m {
                    let xi = *x_ptr.add(i);
                    s0 += *pa0.add(i) * xi;
                    s1 += *pa1.add(i) * xi;
                    i += 1;
                }
                *out.get_unchecked_mut(j + 0) += s0;
                *out.get_unchecked_mut(j + 1) += s1;
            }
            1 => {
                let pa0 = ap.add((j + 0) * lda);
                let mut acc0 = vdupq_n_f32(0.0);

                let mut i = 0usize;
                while i + 16 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));
                    let x2 = vld1q_f32(x_ptr.add(i + 8));
                    let x3 = vld1q_f32(x_ptr.add(i + 12));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    let a02 = vld1q_f32(pa0.add(i + 8));
                    let a03 = vld1q_f32(pa0.add(i + 12));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);
                    acc0 = vfmaq_f32(acc0, a02, x2);
                    acc0 = vfmaq_f32(acc0, a03, x3);

                    i += 16;
                }
                while i + 8 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let x1 = vld1q_f32(x_ptr.add(i + 4));
                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    acc0 = vfmaq_f32(acc0, a01, x1);
                    i += 8;
                }
                while i + 4 <= m {
                    let x0 = vld1q_f32(x_ptr.add(i + 0));
                    let a00 = vld1q_f32(pa0.add(i + 0));
                    acc0 = vfmaq_f32(acc0, a00, x0);
                    i += 4;
                }
                let mut s0 = vaddvq_f32(acc0);
                while i < m {
                    let xi = *x_ptr.add(i);
                    s0 += *pa0.add(i) * xi;
                    i += 1;
                }
                *out.get_unchecked_mut(j + 0) += s0;
            }
            _ => {}
        }

        core::mem::forget(x_contig);
    }
}

