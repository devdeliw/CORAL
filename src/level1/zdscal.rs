use core::arch::aarch64::{ 
    vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64 
}; 

pub fn zdscal(n: usize, alpha: f64, x: &mut [f64], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let a_v = vdupq_n_f64(alpha);
            let mut i = 0usize;
            while i + 2 <= 2 * n {
                let v = vld1q_f64(p);
                let out = vmulq_f64(v, a_v);
                vst1q_f64(p, out);
                p = p.add(2);
                i += 2;
            }
            while i < 2 * n {
                *p = *p * alpha;
                *p.add(1) = *p.add(1) * alpha;
                p = p.add(2);
                i += 2;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let step = 2 * incx as usize;
            let mut idx = 0usize;
            let p = x.as_mut_ptr();
            for _ in 0..n {
                *p.add(idx)     = *p.add(idx) * alpha;
                *p.add(idx + 1) = *p.add(idx + 1) * alpha;
                idx += step;
            }
        }
    }
}

