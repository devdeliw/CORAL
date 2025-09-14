//! Computes the Euclidean norm of a complex single precision vector.
//!
//! This function implements the BLAS [`scnrm2`] routine, returning
//! sqrt(sum(|Re(x[i])|^2 + |Im(x[i])|^2)) over `n` complex elements of the
//! input vector `x` with a specified stride.
//!
//! # Arguments
//! - `n`    (usize)  : Number of complex elements in the vector.
//! - `x`    (&[f32]) : Input slice containing interleaved complex vector elements
//!                   | `[re0, im0, re1, im1, ...]`.
//! - `incx` (usize)  : Stride between consecutive complex elements of `x`
//!                     (measured in complex numbers; every step advances two scalar idxs).
//!
//! # Returns
//! - `f32` Euclidean norm of the selected complex vector elements.
//!
//! # Notes
//! - Uses the scaled sum-of-squares algorithm to avoid overflow and underflow.
//! - For `incx == 1`, [`scnrm2`] uses unrolled NEON SIMD instructions for optimized
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
use crate::level1::assert_length_helpers::required_len_ok_cplx; 

#[inline]
#[cfg(target_arch = "aarch64")] 
pub fn scnrm2(
    n       : usize,
    x       : &[f32], 
    incx    : usize
) -> f32 {
    // quick return 
    if n == 0 || incx == 0 { 
        return 0.0; 
    } 

    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");

    let mut scale : f32 = 0.0; 
    let mut ssq   : f32 = 1.0; 

    unsafe { 
        // fast path 
        if incx == 1 { 
            let end = 2 * n; 

            let mut i = 0; 
            while i + 8 <= end { 
                let v0 = vld1q_f32(x.as_ptr().add(i)); 
                let v1 = vld1q_f32(x.as_ptr().add(i + 4)); 

                let a0 = vabsq_f32(v0); 
                let a1 = vabsq_f32(v1); 

                let m = vmaxq_f32(a0, a1); 
                let chunk_max = vmaxvq_f32(m); 

                if chunk_max > 0.0 { 
                    let inv  = 1.0 / chunk_max; 
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
            let mut ix = 0; 
            for _ in 0..n { 
                let re = *x.get_unchecked(ix);
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

                let im = *x.get_unchecked(ix + 1);
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

                ix += incx * 2;
            }
        }
    }

    scale * ssq.sqrt()
}
