use core::arch::aarch64::{
    vld1q_f32, vdupq_n_f32, vaddvq_f32, vabsq_f32, vmulq_f32, vmaxq_f32, vmaxvq_f32, vfmaq_f32, 
};
use crate::level1::nrm2_helpers::upd_f32; 

#[inline]
pub fn scnrm2(n: usize, x: &[f32], incx: isize) -> f32 {
    // quick return 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    let mut scale : f32 = 0.0; 
    let mut ssq   : f32 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
            let end = 2 * n; 
            debug_assert!(x.len() >= end); 

            let mut i = 0; 
            while i + 8 <= end { 
                let v0 = vld1q_f32(x.as_ptr().add(i)); 
                let v1 = vld1q_f32(x.as_ptr().add(i + 4)); 

                let a0 = vabsq_f32(v0); 
                let a1 = vabsq_f32(v1); 

                let m   = vmaxq_f32(a0, a1); 
                let chunk_max = vmaxvq_f32(m); 

                if chunk_max > 0.0 { 
                    let inv  = 1.0f32 / chunk_max; 
                    let vinv = vdupq_n_f32(inv); 

                    // normalize
                    let n0 = vmulq_f32(a0, vinv); 
                    let n1 = vmulq_f32(a1, vinv); 

                    let mut s = vdupq_n_f32(0.0);
                    s = vfmaq_f32(s, n0, n0);
                    s = vfmaq_f32(s, n1, n1);

                    let chunk_ssq = vaddvq_f32(s);
                    upd_f32(&mut scale, &mut ssq, chunk_max, chunk_ssq);
                }
                
                i += 8; 
            }

            // tail over remaining scalars
            while i < end { 
                let v = *x.as_ptr().add(i);  
                if v != 0.0 { 
                    let a = v.abs(); 
                    if scale < a { 
                        let r = scale / a; 
                        ssq = 1.0 + ssq * (r * r); 
                        scale = a; 
                    } else if scale > 0.0 { 
                        let r = a / scale;
                        ssq += r * r; 
                    } else { 
                        scale = a; 
                    }
                }
                i += 1; 
            }
        } else { 
            // non unit stride 
            let s = incx.unsigned_abs() as usize; 
            debug_assert!(x.len() >= 2 * (1 + (n - 1) * s)); 

            let mut idx:  isize = if incx > 0 { 0 } else { (2 * (n - 1) * s) as isize }; 
            let     dlt:  isize = if incx > 0 { (2 * s) as isize } else { -((2 * s) as isize) }; 

            for _ in 0..n { 
                let re = *x.get_unchecked(idx as usize);
                if re != 0.0 { 
                    let a = re.abs(); 
                    if scale < a { 
                        let r = scale / a; 
                        ssq   = 1.0 + ssq * (r * r); 
                        scale = a; 
                    } else if scale > 0.0 { 
                        let r = a / scale; 
                        ssq  += r * r; 
                    } else { 
                        scale = a;
                    }
                }

                let im = *x.get_unchecked(idx as usize + 1);
                if im != 0.0 { 
                    let a = im.abs(); 
                    if scale < a { 
                        let r = scale / a; 
                        ssq   = 1.0 + ssq * (r * r); 
                        scale = a; 
                    } else if scale > 0.0 { 
                        let r = a / scale; 
                        ssq  += r * r; 
                    } else { 
                        scale = a;
                    }
                }

                idx += dlt;
            }
        }
    }

    scale * ssq.sqrt()
}
