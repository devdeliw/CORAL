use core::arch::aarch64::{
    vld1q_f32, vst1q_f32, vdupq_n_f32, vfmaq_f32, vmulq_f32, vsubq_f32,
};
use crate::level1::assert_length_helpers::required_len_ok;

#[inline]
pub fn srot(n: usize, x: &mut [f32], incx: isize, y: &mut [f32], incy: isize, c: f32, s: f32) {
    // quick return
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_mut_ptr();
    let py = y.as_mut_ptr();

    // fast path
    if incx == 1 && incy == 1 {
        unsafe {
            let c2 = vdupq_n_f32(c);
            let s2 = vdupq_n_f32(s);

            let mut i = 0usize;
            while i + 32 <= n {
                let x0 = vld1q_f32(px.add(i +  0));
                let x1 = vld1q_f32(px.add(i +  4));
                let x2 = vld1q_f32(px.add(i +  8));
                let x3 = vld1q_f32(px.add(i + 12));
                let x4 = vld1q_f32(px.add(i + 16));
                let x5 = vld1q_f32(px.add(i + 20));
                let x6 = vld1q_f32(px.add(i + 24));
                let x7 = vld1q_f32(px.add(i + 28));

                let y0 = vld1q_f32(py.add(i +  0));
                let y1 = vld1q_f32(py.add(i +  4));
                let y2 = vld1q_f32(py.add(i +  8));
                let y3 = vld1q_f32(py.add(i + 12));
                let y4 = vld1q_f32(py.add(i + 16));
                let y5 = vld1q_f32(py.add(i + 20));
                let y6 = vld1q_f32(py.add(i + 24));
                let y7 = vld1q_f32(py.add(i + 28));

                let xn0 = vfmaq_f32(vmulq_f32(c2, x0), y0, s2);
                let xn1 = vfmaq_f32(vmulq_f32(c2, x1), y1, s2);
                let xn2 = vfmaq_f32(vmulq_f32(c2, x2), y2, s2);
                let xn3 = vfmaq_f32(vmulq_f32(c2, x3), y3, s2);
                let xn4 = vfmaq_f32(vmulq_f32(c2, x4), y4, s2);
                let xn5 = vfmaq_f32(vmulq_f32(c2, x5), y5, s2);
                let xn6 = vfmaq_f32(vmulq_f32(c2, x6), y6, s2);
                let xn7 = vfmaq_f32(vmulq_f32(c2, x7), y7, s2);

                let yn0 = vsubq_f32(vmulq_f32(c2, y0), vmulq_f32(s2, x0));
                let yn1 = vsubq_f32(vmulq_f32(c2, y1), vmulq_f32(s2, x1));
                let yn2 = vsubq_f32(vmulq_f32(c2, y2), vmulq_f32(s2, x2));
                let yn3 = vsubq_f32(vmulq_f32(c2, y3), vmulq_f32(s2, x3));
                let yn4 = vsubq_f32(vmulq_f32(c2, y4), vmulq_f32(s2, x4));
                let yn5 = vsubq_f32(vmulq_f32(c2, y5), vmulq_f32(s2, x5));
                let yn6 = vsubq_f32(vmulq_f32(c2, y6), vmulq_f32(s2, x6));
                let yn7 = vsubq_f32(vmulq_f32(c2, y7), vmulq_f32(s2, x7));

                vst1q_f32(py.add(i +  0), yn0);
                vst1q_f32(py.add(i +  4), yn1);
                vst1q_f32(py.add(i +  8), yn2);
                vst1q_f32(py.add(i + 12), yn3);
                vst1q_f32(py.add(i + 16), yn4);
                vst1q_f32(py.add(i + 20), yn5);
                vst1q_f32(py.add(i + 24), yn6);
                vst1q_f32(py.add(i + 28), yn7);

                vst1q_f32(px.add(i +  0), xn0);
                vst1q_f32(px.add(i +  4), xn1);
                vst1q_f32(px.add(i +  8), xn2);
                vst1q_f32(px.add(i + 12), xn3);
                vst1q_f32(px.add(i + 16), xn4);
                vst1q_f32(px.add(i + 20), xn5);
                vst1q_f32(px.add(i + 24), xn6);
                vst1q_f32(px.add(i + 28), xn7);

                i += 32;
            }
            while i + 4 <= n {
                let xv = vld1q_f32(px.add(i));
                let yv = vld1q_f32(py.add(i));
                let xn = vfmaq_f32(vmulq_f32(c2, xv), yv, s2);
                let yn = vsubq_f32(vmulq_f32(c2, yv), vmulq_f32(s2, xv));
                vst1q_f32(py.add(i), yn);
                vst1q_f32(px.add(i), xn);
                i += 4;
            }

            // tail 
            while i < n {
                let xi = *px.add(i);
                let yi = *py.add(i);
                let tmp = c * xi + s * yi;
                *py.add(i) = c * yi - s * xi;
                *px.add(i) = tmp;
                i += 1;
            }
        }
        return;
    }

    // non-unit stride
    unsafe {
        let mut ix: isize = if incx >= 0 { 0 } else { ((n - 1) as isize) * (-incx) };
        let mut iy: isize = if incy >= 0 { 0 } else { ((n - 1) as isize) * (-incy) };
        for _ in 0..n {
            let xi = *px.offset(ix);
            let yi = *py.offset(iy);
            let tmp = c * xi + s * yi;
            *py.offset(iy) = c * yi - s * xi;
            *px.offset(ix) = tmp;
            ix += incx;
            iy += incy;
        }
    }
}

