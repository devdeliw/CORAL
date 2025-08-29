use core::arch::aarch64::{
    vld1q_f64, vdupq_n_f64, vaddq_f64, vaddvq_f64, vabsq_f64,
    vld1q_f32, vdupq_n_f32, vaddq_f32, vaddvq_f32, vabsq_f32,
};


pub fn sasum(n: usize, x: &[f32], incx: isize) -> f32 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    unsafe {
        // fast path 
        if incx == 1 {
            debug_assert!(x.len() >= n);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0;
            while i + 16 <= n {
                let v0 = vabsq_f32(vld1q_f32(x.as_ptr().add(i)));
                let v1 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 4)));
                let v2 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 8)));
                let v3 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 12)));

                acc0 = vaddq_f32(acc0, v0);
                acc1 = vaddq_f32(acc1, v1);
                acc2 = vaddq_f32(acc2, v2);
                acc3 = vaddq_f32(acc3, v3);

                i += 16;
            }

            res += vaddvq_f32(acc0) 
                 + vaddvq_f32(acc1) 
                 + vaddvq_f32(acc2) 
                 + vaddvq_f32(acc3);

            while i < n {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;
            debug_assert!(x.len() >= 1 + (n - 1) * step);

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


pub fn dasum(n: usize, x: &[f64], incx: isize) -> f64 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    unsafe {
        // fast path 
        if incx == 1 {
            debug_assert!(x.len() >= n);
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
            debug_assert!(x.len() >= 1 + (n - 1) * step);

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


pub fn scasum(n: usize, x: &[f32], incx: isize) -> f32 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    unsafe {
        // fast path 
        if incx == 1 {
            debug_assert!(x.len() >= 2 * n);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            let mut i = 0;
            let end = 2 * n;
            while i + 8 <= end {
                let v0 = vabsq_f32(vld1q_f32(x.as_ptr().add(i)));
                let v1 = vabsq_f32(vld1q_f32(x.as_ptr().add(i + 4)));
                acc0 = vaddq_f32(acc0, v0);
                acc1 = vaddq_f32(acc1, v1);
                i += 8;
            }

            res += vaddvq_f32(acc0) 
                 + vaddvq_f32(acc1);

            while i < end {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;
            debug_assert!(x.len() >= 2 * (1 + (n - 1) * step));

            let mut idx = if incx > 0 { 0usize } else { 2 * (n - 1) * step } as isize;
            let delta   = if incx > 0 { (2 * step) as isize } else { -((2 * step) as isize) };

            for _ in 0..n {
                let i = idx as usize;
                let re = *x.get_unchecked(i);
                let im = *x.get_unchecked(i + 1);
                res += re.abs() + im.abs();
                idx += delta;
            }
        }
    }

    res
}


pub fn dzasum(n: usize, x: &[f64], incx: isize) -> f64 {
    let mut res = 0.0;

    // quick return 
    if n == 0 || incx == 0 {
        return res;
    }

    unsafe {
        // fast path 
        if incx == 1 {
            debug_assert!(x.len() >= 2 * n);
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);

            let mut i = 0;
            let end = 2 * n;
            while i + 4 <= end {
                let v0 = vabsq_f64(vld1q_f64(x.as_ptr().add(i)));
                let v1 = vabsq_f64(vld1q_f64(x.as_ptr().add(i + 2)));
                acc0 = vaddq_f64(acc0, v0);
                acc1 = vaddq_f64(acc1, v1);
                i += 4;
            }

            res += vaddvq_f64(acc0) 
                 + vaddvq_f64(acc1);

            while i < end {
                res += (*x.as_ptr().add(i)).abs();
                i += 1;
            }
        } else {
            // non unit stride 
            let step = incx.unsigned_abs() as usize;
            debug_assert!(x.len() >= 2 * (1 + (n - 1) * step));

            let mut idx = if incx > 0 { 0usize } else { 2 * (n - 1) * step } as isize;
            let delta   = if incx > 0 { (2 * step) as isize } else { -((2 * step) as isize) };

            for _ in 0..n {
                let i = idx as usize;
                let re = *x.get_unchecked(i);
                let im = *x.get_unchecked(i + 1);
                res += re.abs() + im.abs();
                idx += delta;
            }
        }
    }

    res
}

