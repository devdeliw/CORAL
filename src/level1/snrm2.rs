//! Computes the Euclidean norm NRM2 of a single precision vector.
//!
//! ```text 
//! sqrt(x[0]^2 + x[1]^2 + ... + x[n-1]^2)
//! ```
//!
//! This function implements the BLAS [`snrm2`] routine, over `n` elements 
//! of the input vector `x` with a specified stride.
//!
//! # Arguments
//! - `n`    (usize)  : Number of elements in the vector.
//! - `x`    (&[f32]) : Input slice containing vector elements.
//! - `incx` (usize)  : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - `f32` Euclidean norm of the selected vector elements.
//!
//! # Notes
//! - Uses the scaled sum-of-squares algorithm to avoid overflow and underflow.
//! - For `incx == 1`, [`snrm2`] uses unrolled NEON SIMD instructions for optimized 
//!   performance on AArch64.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `incx == 0`, the function returns `0.0f32`.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{ 
    vld1q_f32,
    vdupq_n_f32,
    vaddvq_f32,
    vabsq_f32, 
    vmulq_f32, 
    vmaxq_f32, 
    vmaxvq_f32, 
    vfmaq_f32, 
};
use crate::level1::nrm2_helpers::upd_f32;
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline]
#[cfg(target_arch = "aarch64")] 
pub fn snrm2(
    n       : usize, 
    x       : &[f32], 
    incx    : usize
) -> f32 { 
    // quick return 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    let mut scale : f32 = 0.0; 
    let mut ssq   : f32 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
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
                    let inv  = 1.0 / chunk_max; 
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
                        ssq   = 1.0 + ssq * (r * r); 
                        scale = absxi; 
                    } else if scale > 0.0 { 
                        let r = absxi / scale;
                        ssq  += r * r; 
                    } else { 
                        scale = absxi; 
                    }
                }
                i += 1; 
            }
        } else { 
            // non unit stride
            
            let mut ix = 0; 
            for _ in 0..n { 
                let xi = *x.get_unchecked(ix);

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

                ix += incx;
            }
        }
    }

    scale * ssq.sqrt()
}

