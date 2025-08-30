use core::arch::aarch64::{ 
    vld1q_f64, vdupq_n_f64, vaddq_f64, vaddvq_f64, vabsq_f64,
};
use crate::level1::assert_length_helpers::required_len_ok;

pub fn dasum(n: usize, x: &[f64], incx: isize) -> f64 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    unsafe {
        // fast path 
        if incx == 1 {
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            let mut acc2 = vdupq_n_f64(0.0);
            let mut acc3 = vdupq_n_f64(0.0);

            let mut i = 0;
            while i + 8 <= n {
                let v0 = vabsq_f64(vld1q_f64(x.as_ptr().add(i)));
                let v1 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 2)));
                let v2 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 4)));
                let v3 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 6)));

                acc0 = vaddq_f64(acc0, v0);
                acc1 = vaddq_f64(acc1, v1);
                acc2 = vaddq_f64(acc2, v2);
                acc3 = vaddq_f64(acc3, v3);

                i += 8;
            }

            while i + 4 <= n { 
                let v0 = vabsq_f64(vld1q_f64(x.as_ptr().add(i))); 
                let v1 = vabsq_f64(vld1q_f64(x.as_ptr().add(i+2))); 

                acc0  = vaddq_f64(acc0, v0); 
                acc1  = vaddq_f64(acc1, v1);

                i += 4;
            }

            res += vaddvq_f64(acc0) 
                 + vaddvq_f64(acc1) 
                 + vaddvq_f64(acc2) 
                 + vaddvq_f64(acc3);

            while i < n {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;

            let mut idx = if incx > 0 { 0usize } else { (n - 1) * step } as isize;
            let delta   = if incx > 0 { step as isize } else { -(step as isize) };

            for _ in 0..n {
                let i = idx as usize;
                res += (*x.get_unchecked(i)).abs();
                idx += delta;
            }
        }
    }

    res
}
