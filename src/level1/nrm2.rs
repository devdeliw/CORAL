use core::arch::aarch64::{
    vld1q_f64, vdupq_n_f64, vaddvq_f64, vabsq_f64, vmulq_f64, vmaxq_f64, vmaxvq_f64, vfmaq_f64,  
    vld1q_f32, vdupq_n_f32, vaddvq_f32, vabsq_f32, vmulq_f32, vmaxq_f32, vmaxvq_f32, vfmaq_f32, 
};

#[inline(always)] 
fn upd_f32(scale: &mut f32, ssq: &mut f32, cmax: f32, cssq: f32) {
    if *scale < cmax {

        let r  = *scale / cmax;
        *ssq   = *ssq * (r*r) + cssq;
        *scale = cmax;

    } else if *scale > 0.0 {

        let r = cmax / *scale;
        *ssq += cssq * (r*r);

    } else {

        *scale = cmax;
        *ssq   = cssq;

    }
}

#[inline(always)] 
fn upd_f64(scale: &mut f64, ssq: &mut f64, cmax: f64, cssq: f64) {
    if *scale < cmax {

        let r  = *scale / cmax;
        *ssq   = *ssq * (r*r) + cssq;
        *scale = cmax;

    } else if *scale > 0.0 {

        let r = cmax / *scale;
        *ssq += cssq * (r*r);

    } else {

        *scale = cmax;
        *ssq   = cssq;

    }
}
 

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


pub fn dnrm2(n: usize, x: &[f64], incx: isize) -> f64 { 
    // quick return 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    let mut scale : f64 = 0.0; 
    let mut ssq   : f64 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
            debug_assert!(x.len() >= n); 
            let mut i = 0; 
            while i + 8 <= n { 
                let v0 = vld1q_f64(x.as_ptr().add(i)); 
                let v1 = vld1q_f64(x.as_ptr().add(i + 2)); 
                let v2 = vld1q_f64(x.as_ptr().add(i + 4)); 
                let v3 = vld1q_f64(x.as_ptr().add(i + 6));  

                let a0 = vabsq_f64(v0); 
                let a1 = vabsq_f64(v1); 
                let a2 = vabsq_f64(v2); 
                let a3 = vabsq_f64(v3); 

                let m01 = vmaxq_f64(a0, a1); 
                let m23 = vmaxq_f64(a2, a3); 
                let m   = vmaxq_f64(m01, m23); 
                let chunk_max = vmaxvq_f64(m); 

                if chunk_max > 0.0 { 
                    let inv  = 1.0f64 / chunk_max; 
                    let vinv = vdupq_n_f64(inv); 

                    // normalize
                    let n0 = vmulq_f64(a0, vinv); 
                    let n1 = vmulq_f64(a1, vinv); 
                    let n2 = vmulq_f64(a2, vinv); 
                    let n3 = vmulq_f64(a3, vinv); 

                    let mut s = vdupq_n_f64(0.0);
                    s = vfmaq_f64(s, n0, n0);
                    s = vfmaq_f64(s, n1, n1); 
                    s = vfmaq_f64(s, n2, n2); 
                    s = vfmaq_f64(s, n3, n3); 

                    let chunk_ssq = vaddvq_f64(s);
                    upd_f64(&mut scale, &mut ssq, chunk_max, chunk_ssq);
                }
                
                i += 8; 
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


pub fn scnrm2(n: usize, x: &[f32], incx: isize) -> f32 { 
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


pub fn dznrm2(n: usize, x: &[f64], incx: isize) -> f64 { 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    let mut scale : f64 = 0.0; 
    let mut ssq   : f64 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
            let end = 2 * n; 
            debug_assert!(x.len() >= end); 

            let mut i = 0; 
            while i + 4 <= end { 
                let v0 = vld1q_f64(x.as_ptr().add(i)); 
                let v1 = vld1q_f64(x.as_ptr().add(i + 2)); 

                let a0 = vabsq_f64(v0); 
                let a1 = vabsq_f64(v1); 

                let m   = vmaxq_f64(a0, a1); 
                let chunk_max = vmaxvq_f64(m); 

                if chunk_max > 0.0 { 
                    let inv  = 1.0f64 / chunk_max; 
                    let vinv = vdupq_n_f64(inv); 

                    // normalize
                    let n0 = vmulq_f64(a0, vinv); 
                    let n1 = vmulq_f64(a1, vinv); 

                    let mut s = vdupq_n_f64(0.0);
                    s = vfmaq_f64(s, n0, n0);
                    s = vfmaq_f64(s, n1, n1);

                    let chunk_ssq = vaddvq_f64(s);
                    upd_f64(&mut scale, &mut ssq, chunk_max, chunk_ssq);
                }
                
                i += 4; 
            }

            // tail 
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

