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
