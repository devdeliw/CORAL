use core::arch::aarch64::{ 
    vld1q_f32, vdupq_n_f32, vaddvq_f32, vabsq_f32, vmulq_f32, vmaxq_f32, vmaxvq_f32, vfmaq_f32, 
};
use crate::level1::nrm2_helpers::upd_f32;

pub fn snrm2(n: usize, x: &[f32], incx: isize) -> f32 { 
    // quick return 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    let mut scale : f32 = 0.0; 
    let mut ssq   : f32 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
            debug_assert!(x.len() >= n); 
            let mut i = 0; 
            while i + 16 <= n { 
                let v0 = vld1q_f32(x.as_ptr().add(i)); 
                let v1 = vld1q_f32(x.as_ptr().add(i + 4)); 
                let v2 = vld1q_f32(x.as_ptr().add(i + 8)); 
                let v3 = vld1q_f32(x.as_ptr().add(i + 12));  

                let a0 = vabsq_f32(v0); 
                let a1 = vabsq_f32(v1); 
                let a2 = vabsq_f32(v2); 
                let a3 = vabsq_f32(v3); 

                let m01 = vmaxq_f32(a0, a1); 
                let m23 = vmaxq_f32(a2, a3); 
                let m   = vmaxq_f32(m01, m23); 
                let chunk_max = vmaxvq_f32(m); 

                if chunk_max > 0.0 { 
                    let inv  = 1.0f32 / chunk_max; 
                    let vinv = vdupq_n_f32(inv); 

                    // normalize
                    let n0 = vmulq_f32(a0, vinv); 
                    let n1 = vmulq_f32(a1, vinv); 
                    let n2 = vmulq_f32(a2, vinv); 
                    let n3 = vmulq_f32(a3, vinv); 

                    let mut s = vdupq_n_f32(0.0);
                    s = vfmaq_f32(s, n0, n0);
                    s = vfmaq_f32(s, n1, n1); 
                    s = vfmaq_f32(s, n2, n2); 
                    s = vfmaq_f32(s, n3, n3); 

                    let chunk_ssq = vaddvq_f32(s);
                    upd_f32(&mut scale, &mut ssq, chunk_max, chunk_ssq);
                }
                
                i += 16; 
            }

            // tail
            while i < n { 
                let xi = *x.as_ptr().add(i);  
                if xi != 0.0 { 
                    let absxi = xi.abs(); 
                    if scale < absxi { 
                        let r = scale / absxi; 
                        ssq = 1.0 + ssq * (r * r); 
                        scale = absxi; 
                    } else if scale > 0.0 { 
                        let r = absxi / scale;
                        ssq += r * r; 
                    } else { 
                        scale = absxi; 
                    }
                }
                i += 1; 
            }
        } else { 
            // non unit stride
            let step = incx.unsigned_abs() as usize; 
            debug_assert!(x.len() >= 1 + (n - 1) * step); 

            let mut idx: isize = if incx > 0 { 0 } else { (n - 1) * step } as isize; 
            let delta: isize   = if incx > 0 { step as isize } else { -(step as isize) }; 

            for _ in 0..n { 
                let xi = *x.get_unchecked(idx as usize);
                if xi != 0.0 { 
                    let absxi = xi.abs(); 
                    if scale < absxi { 
                        let r = scale / absxi; 
                        ssq   = 1.0 + ssq * (r * r); 
                        scale = absxi; 
                    } else if scale > 0.0 { 
                        let r = absxi / scale; 
                        ssq  += r * r; 
                    } else { 
                        scale = absxi;
                    }
                }
                idx += delta;
            }
        }
    }

    scale * ssq.sqrt()
}

