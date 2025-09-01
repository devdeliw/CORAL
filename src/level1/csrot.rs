use core::arch::aarch64::{
    vld1q_f32, vst1q_f32, vdupq_n_f32, vfmaq_f32, vmulq_f32, vsubq_f32,
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;

#[inline]
pub fn csrot(n: usize, x: &mut [f32], incx: isize, y: &mut [f32], incy: isize, c: f32, s: f32) {
    // quick return
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy (complex)");

    let px = x.as_mut_ptr();
    let py = y.as_mut_ptr();

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe {
            let c2 = vdupq_n_f32(c);
            let s2 = vdupq_n_f32(s);

            let len   = 2 * n; 
            let mut i = 0usize;
            while i + 16 <= len {
                let x0 = vld1q_f32(px.add(i +  0));
                let x1 = vld1q_f32(px.add(i +  4));
                let x2 = vld1q_f32(px.add(i +  8));
                let x3 = vld1q_f32(px.add(i + 12));

                let y0 = vld1q_f32(py.add(i +  0));
                let y1 = vld1q_f32(py.add(i +  4));
                let y2 = vld1q_f32(py.add(i +  8));
                let y3 = vld1q_f32(py.add(i + 12));

                let xn0 = vfmaq_f32(vmulq_f32(c2, x0), y0, s2);
                let xn1 = vfmaq_f32(vmulq_f32(c2, x1), y1, s2);
                let xn2 = vfmaq_f32(vmulq_f32(c2, x2), y2, s2);
                let xn3 = vfmaq_f32(vmulq_f32(c2, x3), y3, s2);

                let yn0 = vsubq_f32(vmulq_f32(c2, y0), vmulq_f32(s2, x0));
                let yn1 = vsubq_f32(vmulq_f32(c2, y1), vmulq_f32(s2, x1));
                let yn2 = vsubq_f32(vmulq_f32(c2, y2), vmulq_f32(s2, x2));
                let yn3 = vsubq_f32(vmulq_f32(c2, y3), vmulq_f32(s2, x3));

                vst1q_f32(py.add(i +  0), yn0);
                vst1q_f32(py.add(i +  4), yn1);
                vst1q_f32(py.add(i +  8), yn2);
                vst1q_f32(py.add(i + 12), yn3);

                vst1q_f32(px.add(i +  0), xn0);
                vst1q_f32(px.add(i +  4), xn1);
                vst1q_f32(px.add(i +  8), xn2);
                vst1q_f32(px.add(i + 12), xn3);

                i += 16;
            }

            while i + 4 <= len {
                let xv = vld1q_f32(px.add(i));
                let yv = vld1q_f32(py.add(i));

                let xn = vfmaq_f32(vmulq_f32(c2, xv), yv, s2);
                let yn = vsubq_f32(vmulq_f32(c2, yv), vmulq_f32(s2, xv));

                vst1q_f32(py.add(i), yn);
                vst1q_f32(px.add(i), xn);

                i += 4;
            }

            // tail 
            while i < len {
                let xi = *px.add(i);
                let yi = *py.add(i);

                let tmp    = c * xi + s * yi;      
                *py.add(i) = c * yi - s * xi;   
                *px.add(i) = tmp;

                i += 1;
            }
        }
        return;
    }

    // non unit stride 
    unsafe {
        let mut ix: isize = if incx >= 0 { 0 } else { ((n - 1) as isize) * (-incx) };
        let mut iy: isize = if incy >= 0 { 0 } else { ((n - 1) as isize) * (-incy) };

        for _ in 0..n {
            let xr = *px.offset(2*ix + 0);
            let xi = *px.offset(2*ix + 1);
            let yr = *py.offset(2*iy + 0);
            let yi = *py.offset(2*iy + 1);

            let xnr = c * xr + s * yr;
            let xni = c * xi + s * yi;

            let ynr = c * yr - s * xr;
            let yni = c * yi - s * xi;

            *px.offset(2*ix + 0) = xnr;
            *px.offset(2*ix + 1) = xni;
            *py.offset(2*iy + 0) = ynr;
            *py.offset(2*iy + 1) = yni;

            ix += incx;
            iy += incy;
        }
    }
}

