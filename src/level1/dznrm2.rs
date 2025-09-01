//! Computes the Euclidean norm of a complex double precision vector.
//!
//! This function implements the BLAS [`dznrm2`] routine, returning
//! sqrt(sum(|Re(x[i])|^2 + |Im(x[i])|^2)) over `n` complex elements of the
//! input vector `x` with a specified stride.
//!
//! # Arguments
//! - `n`    : Number of complex elements in the vector.
//! - `x`    : Input slice containing interleaved complex vector elements
//!            `[re0, im0, re1, im1, ...]`.
//! - `incx` : Stride between consecutive complex elements of `x`
//!            (measured in complex numbers; every step advances two scalar idxs).
//!
//! # Returns
//! - `f64` Euclidean norm of the selected complex vector elements.
//!
//! # Notes
//! - Uses the scaled sum-of-squares algorithm to avoid overflow and underflow.
//! - For `incx == 1`, [`dznrm2`] uses unrolled NEON SIMD instructions for optimized
//!   performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `incx == 0`, the function returns `0.0f64`.
//!
//! # Author
//! Deval Deliwala


use core::arch::aarch64::{
    vld1q_f64, vdupq_n_f64, vaddvq_f64, vabsq_f64, vmulq_f64, vmaxq_f64, vmaxvq_f64, vfmaq_f64,  
};
use crate::level1::nrm2_helpers::upd_f64; 
 

#[inline]
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

                let m = vmaxq_f64(a0, a1); 
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
                        ssq   = 1.0 + ssq * (r * r); 
                        scale = a; 
                    } else if scale > 0.0 { 
                        let r = a / scale;
                        ssq  += r * r; 
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

