use core::arch::aarch64::{vld1q_f32, vdupq_n_f32, vfmaq_f32, vst1q_f32};
use crate::level1::assert_length_helpers::required_len_ok;
use crate::level1::saxpy::saxpy;

#[inline(always)]
pub fn saxpyf(
    m     : usize,
    n     : usize,
    alpha : f32,
    x     : &[f32],
    incx  : isize,
    a     : &[f32],
    lda   : usize,
    y     : &mut [f32],
    incy  : isize,
) {
    // quick return
    if m == 0 || n == 0 || alpha == 0.0 { return; }

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

            // f = 4 cols at a time 
            while j + 4 <= n {
                let x0 = alpha * *x.get_unchecked(j + 0);
                let x1 = alpha * *x.get_unchecked(j + 1);
                let x2 = alpha * *x.get_unchecked(j + 2);
                let x3 = alpha * *x.get_unchecked(j + 3);

                if x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0 {
                    j += 4;
                    continue;
                }

                let s0 = vdupq_n_f32(x0);
                let s1 = vdupq_n_f32(x1);
                let s2 = vdupq_n_f32(x2);
                let s3 = vdupq_n_f32(x3);

                let pa0 = a.as_ptr().add((j + 0) * lda);
                let pa1 = a.as_ptr().add((j + 1) * lda);
                let pa2 = a.as_ptr().add((j + 2) * lda);
                let pa3 = a.as_ptr().add((j + 3) * lda);

                let mut i = 0usize;

                while i + 16 <= m {
                    let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                    let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));
                    let mut y2 = vld1q_f32(y.as_ptr().add(i + 8));
                    let mut y3 = vld1q_f32(y.as_ptr().add(i + 12));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    let a02 = vld1q_f32(pa0.add(i + 8));
                    let a03 = vld1q_f32(pa0.add(i + 12));
                    y0 = vfmaq_f32(y0, s0, a00);
                    y1 = vfmaq_f32(y1, s0, a01);
                    y2 = vfmaq_f32(y2, s0, a02);
                    y3 = vfmaq_f32(y3, s0, a03);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    let b02 = vld1q_f32(pa1.add(i + 8));
                    let b03 = vld1q_f32(pa1.add(i + 12));
                    y0 = vfmaq_f32(y0, s1, b00);
                    y1 = vfmaq_f32(y1, s1, b01);
                    y2 = vfmaq_f32(y2, s1, b02);
                    y3 = vfmaq_f32(y3, s1, b03);

                    let c00 = vld1q_f32(pa2.add(i + 0));
                    let c01 = vld1q_f32(pa2.add(i + 4));
                    let c02 = vld1q_f32(pa2.add(i + 8));
                    let c03 = vld1q_f32(pa2.add(i + 12));
                    y0 = vfmaq_f32(y0, s2, c00);
                    y1 = vfmaq_f32(y1, s2, c01);
                    y2 = vfmaq_f32(y2, s2, c02);
                    y3 = vfmaq_f32(y3, s2, c03);

                    let d00 = vld1q_f32(pa3.add(i + 0));
                    let d01 = vld1q_f32(pa3.add(i + 4));
                    let d02 = vld1q_f32(pa3.add(i + 8));
                    let d03 = vld1q_f32(pa3.add(i + 12));
                    y0 = vfmaq_f32(y0, s3, d00);
                    y1 = vfmaq_f32(y1, s3, d01);
                    y2 = vfmaq_f32(y2, s3, d02);
                    y3 = vfmaq_f32(y3, s3, d03);

                    vst1q_f32(y.as_mut_ptr().add(i + 0),  y0);
                    vst1q_f32(y.as_mut_ptr().add(i + 4),  y1);
                    vst1q_f32(y.as_mut_ptr().add(i + 8),  y2);
                    vst1q_f32(y.as_mut_ptr().add(i + 12), y3);

                    i += 16;
                }

                while i + 8 <= m {
                    let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                    let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));

                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let a01 = vld1q_f32(pa0.add(i + 4));
                    y0 = vfmaq_f32(y0, s0, a00);
                    y1 = vfmaq_f32(y1, s0, a01);

                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let b01 = vld1q_f32(pa1.add(i + 4));
                    y0 = vfmaq_f32(y0, s1, b00);
                    y1 = vfmaq_f32(y1, s1, b01);

                    let c00 = vld1q_f32(pa2.add(i + 0));
                    let c01 = vld1q_f32(pa2.add(i + 4));
                    y0 = vfmaq_f32(y0, s2, c00);
                    y1 = vfmaq_f32(y1, s2, c01);

                    let d00 = vld1q_f32(pa3.add(i + 0));
                    let d01 = vld1q_f32(pa3.add(i + 4));
                    y0 = vfmaq_f32(y0, s3, d00);
                    y1 = vfmaq_f32(y1, s3, d01);

                    vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                    vst1q_f32(y.as_mut_ptr().add(i + 4), y1);
                    i += 8;
                }

                while i + 4 <= m {
                    let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                    let a00 = vld1q_f32(pa0.add(i + 0));
                    let b00 = vld1q_f32(pa1.add(i + 0));
                    let c00 = vld1q_f32(pa2.add(i + 0));
                    let d00 = vld1q_f32(pa3.add(i + 0));
                    y0 = vfmaq_f32(y0, s0, a00);
                    y0 = vfmaq_f32(y0, s1, b00);
                    y0 = vfmaq_f32(y0, s2, c00);
                    y0 = vfmaq_f32(y0, s3, d00);
                    vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                    i += 4;
                }

                while i < m {
                    let mut acc = *y.as_ptr().add(i);
                    acc += x0 * *pa0.add(i);
                    acc += x1 * *pa1.add(i);
                    acc += x2 * *pa2.add(i);
                    acc += x3 * *pa3.add(i);
                    *y.as_mut_ptr().add(i) = acc;
                    i += 1;
                }

                j += 4;
            }

            // leftover, also unrolled 
            let rem = n - j;
            match rem {
                3 => {
                    let x0 = alpha * *x.get_unchecked(j + 0);
                    let x1 = alpha * *x.get_unchecked(j + 1);
                    let x2 = alpha * *x.get_unchecked(j + 2);
                    if x0 == 0.0 && x1 == 0.0 && x2 == 0.0 { return; }

                    let s0 = vdupq_n_f32(x0);
                    let s1 = vdupq_n_f32(x1);
                    let s2 = vdupq_n_f32(x2);

                    let pa0 = a.as_ptr().add((j + 0) * lda);
                    let pa1 = a.as_ptr().add((j + 1) * lda);
                    let pa2 = a.as_ptr().add((j + 2) * lda);

                    let mut i = 0usize;
                    while i + 16 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));
                        let mut y2 = vld1q_f32(y.as_ptr().add(i + 8));
                        let mut y3 = vld1q_f32(y.as_ptr().add(i + 12));

                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        let a02 = vld1q_f32(pa0.add(i + 8));
                        let a03 = vld1q_f32(pa0.add(i + 12));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);
                        y2 = vfmaq_f32(y2, s0, a02);
                        y3 = vfmaq_f32(y3, s0, a03);

                        let b00 = vld1q_f32(pa1.add(i + 0));
                        let b01 = vld1q_f32(pa1.add(i + 4));
                        let b02 = vld1q_f32(pa1.add(i + 8));
                        let b03 = vld1q_f32(pa1.add(i + 12));
                        y0 = vfmaq_f32(y0, s1, b00);
                        y1 = vfmaq_f32(y1, s1, b01);
                        y2 = vfmaq_f32(y2, s1, b02);
                        y3 = vfmaq_f32(y3, s1, b03);

                        let c00 = vld1q_f32(pa2.add(i + 0));
                        let c01 = vld1q_f32(pa2.add(i + 4));
                        let c02 = vld1q_f32(pa2.add(i + 8));
                        let c03 = vld1q_f32(pa2.add(i + 12));
                        y0 = vfmaq_f32(y0, s2, c00);
                        y1 = vfmaq_f32(y1, s2, c01);
                        y2 = vfmaq_f32(y2, s2, c02);
                        y3 = vfmaq_f32(y3, s2, c03);

                        vst1q_f32(y.as_mut_ptr().add(i + 0),  y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4),  y1);
                        vst1q_f32(y.as_mut_ptr().add(i + 8),  y2);
                        vst1q_f32(y.as_mut_ptr().add(i + 12), y3);
                        i += 16;
                    }
                    while i + 8 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));

                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);

                        let b00 = vld1q_f32(pa1.add(i + 0));
                        let b01 = vld1q_f32(pa1.add(i + 4));
                        y0 = vfmaq_f32(y0, s1, b00);
                        y1 = vfmaq_f32(y1, s1, b01);

                        let c00 = vld1q_f32(pa2.add(i + 0));
                        let c01 = vld1q_f32(pa2.add(i + 4));
                        y0 = vfmaq_f32(y0, s2, c00);
                        y1 = vfmaq_f32(y1, s2, c01);

                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4), y1);
                        i += 8;
                    }
                    while i + 4 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let b00 = vld1q_f32(pa1.add(i + 0));
                        let c00 = vld1q_f32(pa2.add(i + 0));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y0 = vfmaq_f32(y0, s1, b00);
                        y0 = vfmaq_f32(y0, s2, c00);
                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        i += 4;
                    }
                    while i < m {
                        let mut acc = *y.as_ptr().add(i);
                        acc += x0 * *pa0.add(i);
                        acc += x1 * *pa1.add(i);
                        acc += x2 * *pa2.add(i);
                        *y.as_mut_ptr().add(i) = acc;
                        i += 1;
                    }
                }
                2 => {
                    let x0 = alpha * *x.get_unchecked(j + 0);
                    let x1 = alpha * *x.get_unchecked(j + 1);
                    if x0 == 0.0 && x1 == 0.0 { return; }

                    let s0 = vdupq_n_f32(x0);
                    let s1 = vdupq_n_f32(x1);

                    let pa0 = a.as_ptr().add((j + 0) * lda);
                    let pa1 = a.as_ptr().add((j + 1) * lda);

                    let mut i = 0usize;
                    while i + 16 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));
                        let mut y2 = vld1q_f32(y.as_ptr().add(i + 8));
                        let mut y3 = vld1q_f32(y.as_ptr().add(i + 12));

                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        let a02 = vld1q_f32(pa0.add(i + 8));
                        let a03 = vld1q_f32(pa0.add(i + 12));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);
                        y2 = vfmaq_f32(y2, s0, a02);
                        y3 = vfmaq_f32(y3, s0, a03);

                        let b00 = vld1q_f32(pa1.add(i + 0));
                        let b01 = vld1q_f32(pa1.add(i + 4));
                        let b02 = vld1q_f32(pa1.add(i + 8));
                        let b03 = vld1q_f32(pa1.add(i + 12));
                        y0 = vfmaq_f32(y0, s1, b00);
                        y1 = vfmaq_f32(y1, s1, b01);
                        y2 = vfmaq_f32(y2, s1, b02);
                        y3 = vfmaq_f32(y3, s1, b03);

                        vst1q_f32(y.as_mut_ptr().add(i + 0),  y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4),  y1);
                        vst1q_f32(y.as_mut_ptr().add(i + 8),  y2);
                        vst1q_f32(y.as_mut_ptr().add(i + 12), y3);
                        i += 16;
                    }
                    while i + 8 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));

                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);

                        let b00 = vld1q_f32(pa1.add(i + 0));
                        let b01 = vld1q_f32(pa1.add(i + 4));
                        y0 = vfmaq_f32(y0, s1, b00);
                        y1 = vfmaq_f32(y1, s1, b01);

                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4), y1);
                        i += 8;
                    }
                    while i + 4 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let b00 = vld1q_f32(pa1.add(i + 0));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y0 = vfmaq_f32(y0, s1, b00);
                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        i += 4;
                    }
                    while i < m {
                        let mut acc = *y.as_ptr().add(i);
                        acc += x0 * *pa0.add(i);
                        acc += x1 * *pa1.add(i);
                        *y.as_mut_ptr().add(i) = acc;
                        i += 1;
                    }
                }
                1 => {
                    let x0 = alpha * *x.get_unchecked(j + 0);
                    if x0 == 0.0 { return; }

                    let s0 = vdupq_n_f32(x0);
                    let pa0 = a.as_ptr().add((j + 0) * lda);

                    let mut i = 0usize;
                    while i + 16 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));
                        let mut y2 = vld1q_f32(y.as_ptr().add(i + 8));
                        let mut y3 = vld1q_f32(y.as_ptr().add(i + 12));

                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        let a02 = vld1q_f32(pa0.add(i + 8));
                        let a03 = vld1q_f32(pa0.add(i + 12));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);
                        y2 = vfmaq_f32(y2, s0, a02);
                        y3 = vfmaq_f32(y3, s0, a03);

                        vst1q_f32(y.as_mut_ptr().add(i + 0),  y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4),  y1);
                        vst1q_f32(y.as_mut_ptr().add(i + 8),  y2);
                        vst1q_f32(y.as_mut_ptr().add(i + 12), y3);
                        i += 16;
                    }
                    while i + 8 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let mut y1 = vld1q_f32(y.as_ptr().add(i + 4));
                        let a00 = vld1q_f32(pa0.add(i + 0));
                        let a01 = vld1q_f32(pa0.add(i + 4));
                        y0 = vfmaq_f32(y0, s0, a00);
                        y1 = vfmaq_f32(y1, s0, a01);
                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        vst1q_f32(y.as_mut_ptr().add(i + 4), y1);
                        i += 8;
                    }
                    while i + 4 <= m {
                        let mut y0 = vld1q_f32(y.as_ptr().add(i + 0));
                        let a00 = vld1q_f32(pa0.add(i + 0));
                        y0 = vfmaq_f32(y0, s0, a00);
                        vst1q_f32(y.as_mut_ptr().add(i + 0), y0);
                        i += 4;
                    }
                    while i < m {
                        let acc = *y.as_ptr().add(i) + x0 * *pa0.add(i);
                        *y.as_mut_ptr().add(i) = acc;
                        i += 1;
                    }
                }
                _ => {}
            }

            return;
        }

        // non unit stride 
        let stepx  = if incx > 0 { incx as usize } else { (-incx) as usize };
        let mut px = x.as_ptr().wrapping_add(if incx >= 0 { 0 } else { (n - 1) * stepx });

        for j in 0..n {
            let scaled = alpha * *px;
            if scaled != 0.0 {
                // a is column-major; contiguous 
                let col_ptr = a.as_ptr().add(j * lda);
                let col = core::slice::from_raw_parts(col_ptr, m);
                saxpy(m, scaled, col, 1, y, incy);
            }
            px = px.offset(incx);
        }
    }
}

