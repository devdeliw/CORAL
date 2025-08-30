use core::arch::aarch64::{
    vld1q_f64, vst1q_f64, 
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;

pub fn zswap(n: usize, x: &mut [f64], incx: isize, y: &mut [f64], incy: isize) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0);
    debug_assert!(required_len_ok_cplx(x.len(), n, incx));
    debug_assert!(required_len_ok_cplx(y.len(), n, incy));

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            let len = 2 * n; 
            let px = x.as_mut_ptr();
            let py = y.as_mut_ptr();
            let mut i = 0usize;

            while i + 8 <= len {
                let ax0 = vld1q_f64(px.add(i + 0));
                let ax1 = vld1q_f64(px.add(i + 2));
                let ax2 = vld1q_f64(px.add(i + 4));
                let ax3 = vld1q_f64(px.add(i + 6));

                let ay0 = vld1q_f64(py.add(i + 0));
                let ay1 = vld1q_f64(py.add(i + 2));
                let ay2 = vld1q_f64(py.add(i + 4));
                let ay3 = vld1q_f64(py.add(i + 6));

                vst1q_f64(py.add(i + 0),  ax0);
                vst1q_f64(py.add(i + 2),  ax1);
                vst1q_f64(py.add(i + 4),  ax2);
                vst1q_f64(py.add(i + 6), ax3);

                vst1q_f64(px.add(i + 0),  ay0);
                vst1q_f64(px.add(i + 2),  ay1);
                vst1q_f64(px.add(i + 4),  ay2);
                vst1q_f64(px.add(i + 6), ay3);

                i += 8;
            }

            while i + 2 <= len {
                let ax = vld1q_f64(px.add(i));
                let ay = vld1q_f64(py.add(i));
                vst1q_f64(py.add(i), ax);
                vst1q_f64(px.add(i), ay);
                i += 2;
            }

            // tail
            while i < len {
                let a = *px.add(i);
                *px.add(i) = *py.add(i);
                *py.add(i) = a;
                i += 1;
            }
            return;
        }

        // non unit stride 
        let px = x.as_mut_ptr();
        let py = y.as_mut_ptr();
        let stepx = (if incx > 0 { incx as usize } else { (-incx) as usize }) * 2;
        let stepy = (if incy > 0 { incy as usize } else { (-incy) as usize }) * 2;
        let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx };
        let mut iy = if incy >= 0 { 0usize } else { (n - 1) * stepy };

        for _ in 0..n {
            let a0 = *px.add(ix);
            *px.add(ix) = *py.add(iy);
            *py.add(iy) = a0;

            let a1 = *px.add(ix + 1);
            *px.add(ix + 1) = *py.add(iy + 1);
            *py.add(iy + 1) = a1;

            if incx >= 0 { ix += stepx } else { ix = ix.wrapping_sub(stepx) }
            if incy >= 0 { iy += stepy } else { iy = iy.wrapping_sub(stepy) }
        }
    }
}

