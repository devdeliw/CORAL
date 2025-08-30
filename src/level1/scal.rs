use core::arch::aarch64::{
    vdupq_n_f32, vld1q_f32, vst1q_f32, vmulq_f32,
    vrev64q_f32, vfmsq_f32, vfmaq_f32, vzip1q_f32, 
    vzip2q_f32, vcombine_f32, vget_low_f32,

    vdupq_n_f64, vld1q_f64, vst1q_f64, vmulq_f64,
    vextq_f64, vfmsq_f64, vfmaq_f64, vzip1q_f64
};


// real alpha, real vector
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(1);
                i += 1;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let mut p = x.as_mut_ptr();
            let step = incx as usize;
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(step);
                i += 1;
            }
        }
    }
}


pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(1);
                i += 1;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let mut p = x.as_mut_ptr();
            let step = incx as usize;
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(step);
                i += 1;
            }
        }
    }
}


// complex alpha, complex vector 
pub fn cscal(n: usize, alpha: [f32; 2], x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    let a_real = alpha[0];
    let a_imag = alpha[1];

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();

            let a_real_v = vdupq_n_f32(a_real);
            let a_imag_v = vdupq_n_f32(a_imag);

            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v_main = vld1q_f32(p);
                let v_swap = vrev64q_f32(v_main);

                let re = vfmsq_f32(vmulq_f32(v_main, a_real_v), v_swap, a_imag_v);
                let im = vfmaq_f32(vmulq_f32(v_swap, a_real_v), v_main, a_imag_v);

                let zip_lo = vzip1q_f32(re, im);
                let zip_hi = vzip2q_f32(re, im);
                let out = vcombine_f32(vget_low_f32(zip_lo), vget_low_f32(zip_hi));

                vst1q_f32(p, out);

                p = p.add(4);
                i += 4;
            }

            // tail
            while i < 2 * n {
                let real = *p;
                let imag = *p.add(1);
                *p        = a_real * real - a_imag * imag;
                *p.add(1) = a_real * imag + a_imag * real;
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
                let real = *p.add(idx);
                let imag = *p.add(idx + 1);
                *p.add(idx)     = a_real * real - a_imag * imag;
                *p.add(idx + 1) = a_real * imag + a_imag * real;
                idx += step;
            }
        }
    }
}


pub fn zscal(n: usize, alpha: [f64; 2], x: &mut [f64], incx: isize) {
    if n == 0 || incx <= 0 { return; }

    let ar = alpha[0];
    let ai = alpha[1];

    // fast path 
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let ar_v = vdupq_n_f64(ar);
            let ai_v = vdupq_n_f64(ai);

            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v0  = vld1q_f64(p);     
                let s0  = vextq_f64(v0, v0, 1); 
                let re0 = vfmsq_f64(vmulq_f64(v0, ar_v), s0, ai_v);
                let im0 = vfmaq_f64(vmulq_f64(s0, ar_v), v0, ai_v);
                let out0 = vzip1q_f64(re0, im0);
                vst1q_f64(p, out0);

                let v1  = vld1q_f64(p.add(2));  
                let s1  = vextq_f64(v1, v1, 1); 
                let re1 = vfmsq_f64(vmulq_f64(v1, ar_v), s1, ai_v);
                let im1 = vfmaq_f64(vmulq_f64(s1, ar_v), v1, ai_v);
                let out1 = vzip1q_f64(re1, im1); 
                vst1q_f64(p.add(2), out1);

                p = p.add(4);
                i += 4;
            }
            while i + 2 <= 2 * n {
                let v   = vld1q_f64(p);         
                let s   = vextq_f64(v, v, 1);   
                let re  = vfmsq_f64(vmulq_f64(v, ar_v), s, ai_v);
                let im  = vfmaq_f64(vmulq_f64(s, ar_v), v, ai_v);
                let out = vzip1q_f64(re, im);   
                vst1q_f64(p, out);
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
                let re = *p.add(idx);
                let im = *p.add(idx + 1);
                *p.add(idx)     = ar * re - ai * im;
                *p.add(idx + 1) = ar * im + ai * re;
                idx += step;
            }
        }
    }
}

// real alpha, complex vector 
pub fn csscal(n: usize, alpha: f32, x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let a_v = vdupq_n_f32(alpha);
            let mut i = 0usize;
            while i + 4 <= 2 * n {
                let v = vld1q_f32(p);
                let out = vmulq_f32(v, a_v);
                vst1q_f32(p, out);
                p = p.add(4);
                i += 4;
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

