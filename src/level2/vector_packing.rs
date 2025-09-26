/// Ensures the length of a buffer is always as large 
/// as the contents we will write to it. 
#[inline(always)] 
fn verify_len_f32(buffer: &mut Vec<f32>, n: usize) { 
    if buffer.len() != n { 
        buffer.resize(n, 0.0); 
    }
} 

fn verify_len_f64(buffer: &mut Vec<f64>, n: usize) { 
    if buffer.len() != n { 
        buffer.resize(n, 0.0); 
    }
}

// Given a strided vector `x`, writes `n` values to a contiguous 
// buffer in `dst` 
pub(crate) fn pack_f32(
    n       : usize, 
    x       : &[f32], 
    incx    : usize, 
    dst     : &mut Vec<f32>,
) { 
    unsafe { 
        if n == 0 { return; } 
        verify_len_f32(dst, n);

        let dst_ptr = dst.as_mut_ptr(); 

        if incx == 1 { 
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst_ptr, n);
            return; 
        }

        let mut i = 0; 
        let mut idx = 0;  

        while i + 4 <= n { 
            let p0 = x.as_ptr().add(idx); 
            let p1 = x.as_ptr().add(idx + incx); 
            let p2 = x.as_ptr().add(idx + 2 * incx); 
            let p3 = x.as_ptr().add(idx + 3 * incx); 

            *dst_ptr.add(i) = *p0; 
            *dst_ptr.add(i + 1) = *p1; 
            *dst_ptr.add(i + 2) = *p2; 
            *dst_ptr.add(i + 3) = *p3; 

            idx += 4 * incx; 
            i   += 4; 
        }

        while i < n { 
            *dst_ptr.add(i) = *x.as_ptr().add(idx); 

            idx += incx; 
            i   += 1; 
        }
    }
}

pub(crate) fn pack_f64(
    n       : usize, 
    x       : &[f64], 
    incx    : usize, 
    dst     : &mut Vec<f64>,
) { 
    unsafe { 
        if n == 0 { return; } 
        verify_len_f64(dst, n);

        let dst_ptr = dst.as_mut_ptr(); 

        if incx == 1 { 
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst_ptr, n);
            return; 
        }

        let mut i = 0; 
        let mut idx = 0;  

        while i + 4 <= n { 
            let p0 = x.as_ptr().add(idx); 
            let p1 = x.as_ptr().add(idx + incx); 
            let p2 = x.as_ptr().add(idx + 2 * incx); 
            let p3 = x.as_ptr().add(idx + 3 * incx); 

            *dst_ptr.add(i) = *p0; 
            *dst_ptr.add(i + 1) = *p1; 
            *dst_ptr.add(i + 2) = *p2; 
            *dst_ptr.add(i + 3) = *p3; 

            idx += 4 * incx; 
            i   += 4; 
        }

        while i < n { 
            *dst_ptr.add(i) = *x.as_ptr().add(idx); 

            idx += incx; 
            i   += 1; 
        }
    }
}

pub(crate) fn pack_c32(
    n: usize,
    x: &[f32],
    incx: usize,
    dst: &mut Vec<f32>,
) {
    unsafe {
        if n == 0 { return; }
        verify_len_f32(dst, 2 * n);

        if incx == 1 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();
        let mut i   = 0;      
        let mut idx = 0;   
        let s       = 2 * incx; 

        while i + 2 <= n {
            let p0 = x.as_ptr().add(idx);
            let p1 = x.as_ptr().add(idx + s);

            // write two complexes (4 scalars)
            *dst_ptr.add(2 * i)       = *p0;
            *dst_ptr.add(2 * i + 1)   = *p0.add(1);
            *dst_ptr.add(2 * i + 2)   = *p1;
            *dst_ptr.add(2 * i + 3)   = *p1.add(1);

            i   += 2;
            idx += 2 * s;
        }

        while i < n {
            let p = x.as_ptr().add(idx);
            *dst_ptr.add(2 * i)     = *p;
            *dst_ptr.add(2 * i + 1) = *p.add(1);

            i   += 1;
            idx += s;
        }
    }
}

pub(crate) fn pack_c64(
    n: usize,
    x: &[f64],
    incx: usize,
    dst: &mut Vec<f64>,
) {
    unsafe {
        if n == 0 { return; }
        verify_len_f64(dst, 2 * n);

        if incx == 1 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();
        let mut i   = 0;
        let mut idx = 0;
        let s       = 2 * incx;

        while i + 2 <= n {
            let p0 = x.as_ptr().add(idx);
            let p1 = x.as_ptr().add(idx + s);

            *dst_ptr.add(2 * i)       = *p0;
            *dst_ptr.add(2 * i + 1)   = *p0.add(1);
            *dst_ptr.add(2 * i + 2)   = *p1;
            *dst_ptr.add(2 * i + 3)   = *p1.add(1);

            i   += 2;
            idx += 2 * s;
        }

        while i < n {
            let p = x.as_ptr().add(idx);
            *dst_ptr.add(2 * i)     = *p;
            *dst_ptr.add(2 * i + 1) = *p.add(1);

            i   += 1;
            idx += s;
        }
    }
}

#[inline] 
#[cfg(target_arch = "aarch64")] 
pub(crate) fn pack_and_scale_f32( 
    n       : usize, 
    alpha   : f32, 
    x       : &[f32], 
    incx    : usize, 
    dst     : &mut Vec<f32> // destination buffer 
) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32}; 

    unsafe { 
        // quick return 
        if n == 0 { return; } 
        verify_len_f32(dst, n);

        if alpha == 0.0 { 
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, n); 
            return; 
        }

        if incx == 1 && alpha == 1.0 { 
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), n);
            return; 
        }

        let dst_ptr = dst.as_mut_ptr(); 

        // fast path 
        if incx == 1 { 
            let a = vdupq_n_f32(alpha); 

            let mut i = 0; 
            while i + 16 <= n { 
                let p  = x.as_ptr().add(i); 
                let x0 = vld1q_f32(p);
                let x1 = vld1q_f32(p.add(4));
                let x2 = vld1q_f32(p.add(8));
                let x3 = vld1q_f32(p.add(12));

                vst1q_f32(dst_ptr.add(i), vmulq_f32(x0, a));
                vst1q_f32(dst_ptr.add(i + 4), vmulq_f32(x1, a));
                vst1q_f32(dst_ptr.add(i + 8), vmulq_f32(x2, a));
                vst1q_f32(dst_ptr.add(i + 12), vmulq_f32(x3, a));

                i += 16; 
            } 

            while i + 4 <= n { 
                let p  = x.as_ptr().add(i);
                let x0 = vld1q_f32(p);

                vst1q_f32(dst_ptr.add(i), vmulq_f32(x0, a));

                i += 4;
            }

            while i < n { 
                *dst_ptr.add(i) = alpha * *x.as_ptr().add(i); 

                i += 1; 
            }
        } else { 
            // non unit stride 
            let mut i   = 0;
            let mut idx = 0; 
            while i + 4 <= n { 
                let p0 = x.as_ptr().add(idx); 
                let p1 = x.as_ptr().add(idx + incx); 
                let p2 = x.as_ptr().add(idx + 2 * incx); 
                let p3 = x.as_ptr().add(idx + 3 * incx); 

                *dst_ptr.add(i) = alpha * *p0; 
                *dst_ptr.add(i + 1) = alpha * *p1; 
                *dst_ptr.add(i + 2) = alpha * *p2; 
                *dst_ptr.add(i + 3) = alpha * *p3; 

                i   += 4;
                idx += 4 * incx; 
            }

            while i < n { 
                let p = x.as_ptr().add(idx); 
                *dst_ptr.add(i) = alpha * *p; 
    
                i   += 1; 
                idx += incx; 
            }
        }
    }
}

#[inline] 
#[cfg(target_arch = "aarch64")] 
pub(crate) fn pack_and_scale_f64( 
    n       : usize, 
    alpha   : f64, 
    x       : &[f64], 
    incx    : usize, 
    dst     : &mut Vec<f64> // destination buffer 
) {
    use core::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64}; 

    unsafe { 
        // quick return 
        if n == 0 { return; } 
        verify_len_f64(dst, n);

        if alpha == 0.0 { 
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, n); 
            return; 
        }

        if incx == 1 && alpha == 1.0 { 
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), n);
            return; 
        }

        let dst_ptr = dst.as_mut_ptr(); 

        // fast path 
        if incx == 1 { 
            let a = vdupq_n_f64(alpha); 

            let mut i = 0; 
            while i + 8 <= n { 
                let p  = x.as_ptr().add(i); 
                let x0 = vld1q_f64(p);
                let x1 = vld1q_f64(p.add(2));
                let x2 = vld1q_f64(p.add(4));
                let x3 = vld1q_f64(p.add(6));

                vst1q_f64(dst_ptr.add(i),      vmulq_f64(x0, a));
                vst1q_f64(dst_ptr.add(i + 2),  vmulq_f64(x1, a));
                vst1q_f64(dst_ptr.add(i + 4),  vmulq_f64(x2, a));
                vst1q_f64(dst_ptr.add(i + 6),  vmulq_f64(x3, a));

                i += 8; 
            } 

            while i + 2 <= n { 
                let p  = x.as_ptr().add(i);
                let x0 = vld1q_f64(p);

                vst1q_f64(dst_ptr.add(i), vmulq_f64(x0, a));

                i += 2;
            }

            while i < n { 
                *dst_ptr.add(i) = alpha * *x.as_ptr().add(i); 

                i += 1; 
            }
        } else { 
            // non unit stride 
            let mut i   = 0;
            let mut idx = 0; 
            while i + 4 <= n { 
                let p0 = x.as_ptr().add(idx); 
                let p1 = x.as_ptr().add(idx + incx); 
                let p2 = x.as_ptr().add(idx + 2 * incx); 
                let p3 = x.as_ptr().add(idx + 3 * incx); 

                *dst_ptr.add(i)     = alpha * *p0; 
                *dst_ptr.add(i + 1) = alpha * *p1; 
                *dst_ptr.add(i + 2) = alpha * *p2; 
                *dst_ptr.add(i + 3) = alpha * *p3; 

                i   += 4;
                idx += 4 * incx; 
            }

            while i < n { 
                let p = x.as_ptr().add(idx); 
                *dst_ptr.add(i) = alpha * *p; 
    
                i   += 1; 
                idx += incx; 
            }
        }
    }
} 

#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn pack_and_rscale_c32(
    n     : usize,
    alpha : f32,
    x     : &[f32],
    incx  : usize,
    dst   : &mut Vec<f32>,
) {
    use core::arch::aarch64::{
        float32x4x2_t, vdupq_n_f32, vld2q_f32, vmulq_f32, vst2q_f32,
    };

    unsafe {
        if n == 0 { return; }
        verify_len_f32(dst, 2 * n);

        if alpha == 0.0 {
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, 2 * n);
            return;
        }
        if incx == 1 && alpha == 1.0 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();

        if incx == 1 {
            let a = vdupq_n_f32(alpha);

            let mut i = 0;
            while i + 8 <= 2 * n {
                let p  = x.as_ptr().add(i);
                let mut z: float32x4x2_t = vld2q_f32(p); 
                z.0 = vmulq_f32(z.0, a);
                z.1 = vmulq_f32(z.1, a);
                vst2q_f32(dst_ptr.add(i), z);
                i += 8;
            }
            while i < 2 * n {
                *dst_ptr.add(i) = alpha * *x.as_ptr().add(i);
                i += 1;
            }
        } else {
            let mut i   = 0;
            let mut idx = 0;
            let s       = 2 * incx;

            while i < n {
                let pr = x.as_ptr().add(idx);
                *dst_ptr.add(2 * i)     = alpha * *pr;
                *dst_ptr.add(2 * i + 1) = alpha * *pr.add(1);
                i   += 1;
                idx += s;
            }
        }
    }
}

#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn pack_and_scale_c32(
    n        : usize,
    alpha    : [f32; 2], 
    x        : &[f32],
    incx     : usize,
    dst      : &mut Vec<f32>,
) {
    use core::arch::aarch64::{
        float32x4x2_t, vdupq_n_f32, vld2q_f32, vmulq_f32, vfmaq_f32, vfmsq_f32, vst2q_f32,
    };

    let alpha_re = alpha[0]; 
    let alpha_im = alpha[1]; 

    unsafe {
        if n == 0 { return; }
        verify_len_f32(dst, 2 * n);

        if alpha_re == 0.0 && alpha_im == 0.0 {
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, 2 * n);
            return;
        }
        if incx == 1 && alpha_re == 1.0 && alpha_im == 0.0 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();

        if incx == 1 {
            let ar = vdupq_n_f32(alpha_re);
            let ai = vdupq_n_f32(alpha_im);

            let mut i = 0;
            while i + 8 <= 2 * n {
                let p  = x.as_ptr().add(i);
                let a: float32x4x2_t = vld2q_f32(p);

                // out_re = ar*re - ai*im
                let mut out_re = vmulq_f32(a.0, ar);
                out_re = vfmsq_f32(out_re, a.1, ai);

                // out_im = ar*im + ai*re
                let mut out_im = vmulq_f32(a.1, ar);
                out_im = vfmaq_f32(out_im, a.0, ai);

                let mut z = a;
                z.0 = out_re;
                z.1 = out_im;
                vst2q_f32(dst_ptr.add(i), z);

                i += 8;
            }

            while i < 2 * n {
                let xr = *x.as_ptr().add(i);
                let xi = *x.as_ptr().add(i + 1);
                *dst_ptr.add(i)     = alpha_re * xr - alpha_im * xi;
                *dst_ptr.add(i + 1) = alpha_re * xi + alpha_im * xr;
                i += 2;
            }
        } else {
            let mut i   = 0;
            let mut idx = 0;
            let s       = 2 * incx;

            while i < n {
                let xr = *x.as_ptr().add(idx);
                let xi = *x.as_ptr().add(idx + 1);

                *dst_ptr.add(2 * i)     = alpha_re * xr - alpha_im * xi;
                *dst_ptr.add(2 * i + 1) = alpha_re * xi + alpha_im * xr;

                i   += 1;
                idx += s;
            }
        }
    }
}

#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn pack_and_rscale_c64(
    n     : usize,
    alpha : f64,
    x     : &[f64],    
    incx  : usize,
    dst   : &mut Vec<f64>,
) {
    use core::arch::aarch64::{
        float64x2x2_t, vdupq_n_f64, vld2q_f64, vmulq_f64, vst2q_f64,
    };

    unsafe {
        if n == 0 { return; }
        verify_len_f64(dst, 2 * n);

        if alpha == 0.0 {
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, 2 * n);
            return;
        }
        if incx == 1 && alpha == 1.0 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();

        if incx == 1 {
            let a = vdupq_n_f64(alpha);

            let mut i = 0;
            while i + 4 <= 2 * n {
                let p  = x.as_ptr().add(i);
                let mut z: float64x2x2_t = vld2q_f64(p);
                z.0 = vmulq_f64(z.0, a);
                z.1 = vmulq_f64(z.1, a);
                vst2q_f64(dst_ptr.add(i), z);
                i += 4;
            }

            while i < 2 * n {
                *dst_ptr.add(i) = alpha * *x.as_ptr().add(i);
                i += 1;
            }
        } else {
            let mut i   = 0;
            let mut idx = 0;
            let s       = 2 * incx;

            while i < n {
                let pr = x.as_ptr().add(idx);
                *dst_ptr.add(2 * i)     = alpha * *pr;
                *dst_ptr.add(2 * i + 1) = alpha * *pr.add(1);
                i   += 1;
                idx += s;
            }
        }
    }
}

#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn pack_and_scale_c64(
    n        : usize,
    alpha    : [f64; 2], 
    x        : &[f64],
    incx     : usize,   
    dst      : &mut Vec<f64>,
) {
    use core::arch::aarch64::{
        float64x2x2_t, vdupq_n_f64, vld2q_f64, vmulq_f64, vfmaq_f64, vfmsq_f64, vst2q_f64,
    };

    let alpha_re = alpha[0]; 
    let alpha_im = alpha[1]; 

    unsafe {
        if n == 0 { return; }
        verify_len_f64(dst, 2 * n);

        if alpha_re == 0.0 && alpha_im == 0.0 {
            core::ptr::write_bytes(dst.as_mut_ptr(), 0, 2 * n);
            return;
        }
        if incx == 1 && alpha_re == 1.0 && alpha_im == 0.0 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), 2 * n);
            return;
        }

        let dst_ptr = dst.as_mut_ptr();

        if incx == 1 {
            let ar = vdupq_n_f64(alpha_re);
            let ai = vdupq_n_f64(alpha_im);

            let mut i = 0;
            while i + 4 <= 2 * n {
                let p  = x.as_ptr().add(i);
                let a: float64x2x2_t = vld2q_f64(p); 

                // out_re = ar*re - ai*im
                let mut out_re = vmulq_f64(a.0, ar);
                out_re = vfmsq_f64(out_re, a.1, ai);

                // out_im = ar*im + ai*re
                let mut out_im = vmulq_f64(a.1, ar);
                out_im = vfmaq_f64(out_im, a.0, ai);

                let mut z = a;
                z.0 = out_re;
                z.1 = out_im;
                vst2q_f64(dst_ptr.add(i), z);

                i += 4;
            }

            while i < 2 * n {
                let xr = *x.as_ptr().add(i);
                let xi = *x.as_ptr().add(i + 1);
                *dst_ptr.add(i)     = alpha_re * xr - alpha_im * xi;
                *dst_ptr.add(i + 1) = alpha_re * xi + alpha_im * xr;
                i += 2;
            }
        } else {
            let mut i   = 0;
            let mut idx = 0;
            let s       = 2 * incx;

            while i < n {
                let xr = *x.as_ptr().add(idx);
                let xi = *x.as_ptr().add(idx + 1);

                *dst_ptr.add(2 * i)     = alpha_re * xr - alpha_im * xi;
                *dst_ptr.add(2 * i + 1) = alpha_re * xi + alpha_im * xr;

                i   += 1;
                idx += s;
            }
        }
    }
}

/// Writes back contiguous buffer to strided vector 
#[inline(always)] 
pub(crate) fn write_back_f32( 
    n       : usize, 
    buffer  : &[f32], 
    x       : &mut [f32], 
    incx    : usize, 
) { 
    unsafe { 
        if n == 0 { return; } 

        let mut ptr = x.as_mut_ptr(); 
        for i in 0..n { 
            *ptr = *buffer.get_unchecked(i); 
            ptr  = ptr.add(incx); 
        }
    }
} 

#[inline(always)] 
pub(crate) fn write_back_f64( 
    n       : usize, 
    buffer  : &[f64], 
    x       : &mut [f64], 
    incx    : usize, 
) { 
    unsafe { 
        if n == 0 { return; } 

        let mut ptr = x.as_mut_ptr(); 
        for i in 0..n { 
            *ptr = *buffer.get_unchecked(i); 
            ptr  = ptr.add(incx); 
        }
    }
}

#[inline(always)]
pub(crate) fn write_back_c32(
    n      : usize,
    buffer : &[f32],
    x      : &mut [f32],
    incx   : usize,
) {
    unsafe {
        if n == 0 { return; }

        let mut ptr = x.as_mut_ptr();
        for i in 0..n {
            *ptr           = *buffer.get_unchecked(2 * i);
            *ptr.add(1)    = *buffer.get_unchecked(2 * i + 1);
            ptr = ptr.add(2 * incx);
        }
    }
}

#[inline(always)]
pub(crate) fn write_back_c64(
    n      : usize,
    buffer : &[f64],
    x      : &mut [f64],
    incx   : usize,
) {
    unsafe {
        if n == 0 { return; }

        let mut ptr = x.as_mut_ptr();
        for i in 0..n {
            *ptr           = *buffer.get_unchecked(2 * i);
            *ptr.add(1)    = *buffer.get_unchecked(2 * i + 1);
            ptr = ptr.add(2 * incx);
        }
    }
}
