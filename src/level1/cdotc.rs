use core::arch::aarch64::{
    vld1q_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vaddvq_f32, vaddq_f32, vuzp1q_f32, vuzp2q_f32,
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;

#[inline]
pub fn cdotc(n: usize, x: &[f32], incx: isize, y: &[f32], incy: isize) -> [f32; 2] {
    if n == 0 { return [0.0, 0.0]; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_ptr();
    let py = y.as_ptr();

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe {
            let mut acc_re0 = vdupq_n_f32(0.0);
            let mut acc_im0 = vdupq_n_f32(0.0);
            let mut acc_re1 = vdupq_n_f32(0.0);
            let mut acc_im1 = vdupq_n_f32(0.0);

            let mut i = 0usize;
            let len = 2 * n; 

            while i + 16 <= len {
                let x0 = vld1q_f32(px.add(i + 0));
                let x1 = vld1q_f32(px.add(i + 4));
                let x2 = vld1q_f32(px.add(i + 8));
                let x3 = vld1q_f32(px.add(i + 12));
                let y0 = vld1q_f32(py.add(i + 0));
                let y1 = vld1q_f32(py.add(i + 4));
                let y2 = vld1q_f32(py.add(i + 8));
                let y3 = vld1q_f32(py.add(i + 12));

                let x_re0 = vuzp1q_f32(x0, x1);
                let x_im0 = vuzp2q_f32(x0, x1);
                let y_re0 = vuzp1q_f32(y0, y1);
                let y_im0 = vuzp2q_f32(y0, y1);

                acc_re0 = vfmaq_f32(acc_re0, x_re0, y_re0);
                acc_re0 = vfmaq_f32(acc_re0, x_im0, y_im0);
                // imag += xr*yi - xi*yr
                acc_im0 = vfmaq_f32(acc_im0, x_re0, y_im0);
                acc_im0 = vfmsq_f32(acc_im0, x_im0, y_re0);

                let x_re1 = vuzp1q_f32(x2, x3);
                let x_im1 = vuzp2q_f32(x2, x3);
                let y_re1 = vuzp1q_f32(y2, y3);
                let y_im1 = vuzp2q_f32(y2, y3);

                acc_re1 = vfmaq_f32(acc_re1, x_re1, y_re1);
                acc_re1 = vfmaq_f32(acc_re1, x_im1, y_im1);
                acc_im1 = vfmaq_f32(acc_im1, x_re1, y_im1);
                acc_im1 = vfmsq_f32(acc_im1, x_im1, y_re1);

                i += 16;
            }

            while i + 8 <= len {
                let x0 = vld1q_f32(px.add(i + 0));
                let x1 = vld1q_f32(px.add(i + 4));
                let y0 = vld1q_f32(py.add(i + 0));
                let y1 = vld1q_f32(py.add(i + 4));

                let x_re = vuzp1q_f32(x0, x1);
                let x_im = vuzp2q_f32(x0, x1);
                let y_re = vuzp1q_f32(y0, y1);
                let y_im = vuzp2q_f32(y0, y1);

                acc_re0 = vfmaq_f32(acc_re0, x_re, y_re);
                acc_re0 = vfmaq_f32(acc_re0, x_im, y_im);
                acc_im0 = vfmaq_f32(acc_im0, x_re, y_im);
                acc_im0 = vfmsq_f32(acc_im0, x_im, y_re);

                i += 8;
            }

            let acc_re = vaddq_f32(acc_re0, acc_re1);
            let acc_im = vaddq_f32(acc_im0, acc_im1);

            let mut real = vaddvq_f32(acc_re);
            let mut imag = vaddvq_f32(acc_im);

            // tail 
            while i < len {
                let xr = *px.add(i);
                let xi = *px.add(i + 1);
                let yr = *py.add(i);
                let yi = *py.add(i + 1);

                real += xr * yr + xi * yi;
                imag += xr * yi - xi * yr;

                i += 2;
            }

            return [real, imag];
        }
    }

    // non unit stride 
    unsafe {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        let stepx = 2 * incx;
        let stepy = 2 * incy;

        let mut ix: isize = if incx >= 0 { 0 } else { 2 * (n as isize - 1) * -incx };
        let mut iy: isize = if incy >= 0 { 0 } else { 2 * (n as isize - 1) * -incy };

        for _ in 0..n {
            let xr = *px.offset(ix);
            let xi = *px.offset(ix + 1);
            let yr = *py.offset(iy);
            let yi = *py.offset(iy + 1);

            real += xr * yr + xi * yi;
            imag += xr * yi - xi * yr;

            ix += stepx;
            iy += stepy;
        }

        [real, imag]
    }
}

