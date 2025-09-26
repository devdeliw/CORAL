//! Applies a modified Givens rotation ROTM to two single precision vectors.
//!
//! This function implements the BLAS [`srotm`] routine, updating the elements of 
//! vectors `x` and `y` using the modified Givens transformation defined by the 
//! parameter array `param`. The transformation form depends on `param[0]` (`flag`):
//! 
//! - `-2.0` : Identity (no operation).
//! - `-1.0` : General 2x2 matrix with parameters `h11, h12, h21, h22`.
//! - ` 0.0` : Simplified form with implicit ones on the diagonal.
//! - `+1.0` : Alternate simplified form with fixed off-diagonal structure.
//!
//! # Arguments
//! - `n`     (usize)      : Number of elements to process.
//! - `x`     (&mut [f32]) : First input/output slice containing vector elements.
//! - `incx`  (usize)      : Stride between consecutive elements of `x`.
//! - `y`     (&mut [f32]) : Second input/output slice containing vector elements.
//! - `incy`  (usize)      : Stride between consecutive elements of `y`.
//! - `param` ([f32; 5])   : Array of 5 parameters defining the modified Givens rotation
//!                          (`flag, h11, h21, h12, h22`).
//!
//! # Returns
//! - Nothing. The contents of `x` and `y` are updated in place.
//!
//! # Notes
//! - For `flag = -2.0`, the routine exits immediately without modifying inputs.
//! - For unit strides, [`srotm`] uses unrolled NEON SIMD instructions for optimized 
//!   performance on AArch64. 
//! - For non-unit strides, it falls back to scalar loops.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f32,
    vst1q_f32, 
    vdupq_n_f32,
    vfmaq_f32,
    vmulq_f32, 
    vsubq_f32,
};
use crate::level1::assert_length_helpers::required_len_ok;


#[inline]
#[cfg(target_arch = "aarch64")] 
pub fn srotm(
    n       : usize, 
    x       : &mut [f32], 
    incx    : usize, 
    y       : &mut [f32], 
    incy    : usize, 
    param   : &[f32; 5]
) {
    // quick return
    if n == 0 { return; }

    let flag = param[0];

    // quick return 
    // identity
    if flag == -2.0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_mut_ptr();
    let py = y.as_mut_ptr();

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe {
            if flag < 0.0 {

                // `flag` = -1 
                // [h11,  h12]
                // [h21,  h22]
                let h11v = vdupq_n_f32(param[1]);
                let h21v = vdupq_n_f32(param[2]);
                let h12v = vdupq_n_f32(param[3]);
                let h22v = vdupq_n_f32(param[4]);

                let mut i = 0;
                while i + 32 <= n {
                    let x0 = vld1q_f32(px.add(i + 0));
                    let x1 = vld1q_f32(px.add(i + 4));
                    let x2 = vld1q_f32(px.add(i + 8));
                    let x3 = vld1q_f32(px.add(i + 12));
                    let x4 = vld1q_f32(px.add(i + 16));
                    let x5 = vld1q_f32(px.add(i + 20));
                    let x6 = vld1q_f32(px.add(i + 24));
                    let x7 = vld1q_f32(px.add(i + 28));

                    let y0 = vld1q_f32(py.add(i + 0));
                    let y1 = vld1q_f32(py.add(i + 4));
                    let y2 = vld1q_f32(py.add(i + 8));
                    let y3 = vld1q_f32(py.add(i + 12));
                    let y4 = vld1q_f32(py.add(i + 16));
                    let y5 = vld1q_f32(py.add(i + 20));
                    let y6 = vld1q_f32(py.add(i + 24));
                    let y7 = vld1q_f32(py.add(i + 28));

                    let xn0 = vfmaq_f32(vmulq_f32(h11v, x0), y0, h12v);
                    let xn1 = vfmaq_f32(vmulq_f32(h11v, x1), y1, h12v);
                    let xn2 = vfmaq_f32(vmulq_f32(h11v, x2), y2, h12v);
                    let xn3 = vfmaq_f32(vmulq_f32(h11v, x3), y3, h12v);
                    let xn4 = vfmaq_f32(vmulq_f32(h11v, x4), y4, h12v);
                    let xn5 = vfmaq_f32(vmulq_f32(h11v, x5), y5, h12v);
                    let xn6 = vfmaq_f32(vmulq_f32(h11v, x6), y6, h12v);
                    let xn7 = vfmaq_f32(vmulq_f32(h11v, x7), y7, h12v);

                    let yn0 = vfmaq_f32(vmulq_f32(h22v, y0), x0, h21v);
                    let yn1 = vfmaq_f32(vmulq_f32(h22v, y1), x1, h21v);
                    let yn2 = vfmaq_f32(vmulq_f32(h22v, y2), x2, h21v);
                    let yn3 = vfmaq_f32(vmulq_f32(h22v, y3), x3, h21v);
                    let yn4 = vfmaq_f32(vmulq_f32(h22v, y4), x4, h21v);
                    let yn5 = vfmaq_f32(vmulq_f32(h22v, y5), x5, h21v);
                    let yn6 = vfmaq_f32(vmulq_f32(h22v, y6), x6, h21v);
                    let yn7 = vfmaq_f32(vmulq_f32(h22v, y7), x7, h21v);

                    vst1q_f32(py.add(i + 0), yn0);
                    vst1q_f32(py.add(i + 4 ), yn1);
                    vst1q_f32(py.add(i + 8), yn2);
                    vst1q_f32(py.add(i + 12), yn3);
                    vst1q_f32(py.add(i + 16), yn4);
                    vst1q_f32(py.add(i + 20), yn5);
                    vst1q_f32(py.add(i + 24), yn6);
                    vst1q_f32(py.add(i + 28), yn7);

                    vst1q_f32(px.add(i + 0), xn0);
                    vst1q_f32(px.add(i + 4), xn1);
                    vst1q_f32(px.add(i + 8), xn2);
                    vst1q_f32(px.add(i + 12), xn3);
                    vst1q_f32(px.add(i + 16), xn4);
                    vst1q_f32(px.add(i + 20), xn5);
                    vst1q_f32(px.add(i + 24), xn6);
                    vst1q_f32(px.add(i + 28), xn7);

                    i += 32;
                }

                while i + 4 <= n {
                    let xv = vld1q_f32(px.add(i));
                    let yv = vld1q_f32(py.add(i));

                    let xn = vfmaq_f32(vmulq_f32(h11v, xv), yv, h12v);
                    let yn = vfmaq_f32(vmulq_f32(h22v, yv), xv, h21v);

                    vst1q_f32(py.add(i), yn);
                    vst1q_f32(px.add(i), xn);

                    i += 4;
                }

                // tail
                while i < n {
                    let xi = *px.add(i);
                    let yi = *py.add(i);
                    let xn = param[1] * xi + param[3] * yi; 
                    let yn = param[2] * xi + param[4] * yi; 
                    *py.add(i) = yn;
                    *px.add(i) = xn;
                    i += 1;
                }
            } else if flag == 0.0 {
        
                // `flag` = 0
                // [1.0, h12] 
                // [h21, 1.0]
                let h12v = vdupq_n_f32(param[3]);
                let h21v = vdupq_n_f32(param[2]);

                let mut i = 0;
                while i + 32 <= n {
                    let x0 = vld1q_f32(px.add(i + 0));
                    let x1 = vld1q_f32(px.add(i + 4));
                    let x2 = vld1q_f32(px.add(i + 8));
                    let x3 = vld1q_f32(px.add(i + 12));
                    let x4 = vld1q_f32(px.add(i + 16));
                    let x5 = vld1q_f32(px.add(i + 20));
                    let x6 = vld1q_f32(px.add(i + 24));
                    let x7 = vld1q_f32(px.add(i + 28));

                    let y0 = vld1q_f32(py.add(i));
                    let y1 = vld1q_f32(py.add(i + 4));
                    let y2 = vld1q_f32(py.add(i + 8));
                    let y3 = vld1q_f32(py.add(i + 12));
                    let y4 = vld1q_f32(py.add(i + 16));
                    let y5 = vld1q_f32(py.add(i + 20));
                    let y6 = vld1q_f32(py.add(i + 24));
                    let y7 = vld1q_f32(py.add(i + 28));

                    let xn0 = vfmaq_f32(x0, y0, h12v);
                    let xn1 = vfmaq_f32(x1, y1, h12v);
                    let xn2 = vfmaq_f32(x2, y2, h12v);
                    let xn3 = vfmaq_f32(x3, y3, h12v);
                    let xn4 = vfmaq_f32(x4, y4, h12v);
                    let xn5 = vfmaq_f32(x5, y5, h12v);
                    let xn6 = vfmaq_f32(x6, y6, h12v);
                    let xn7 = vfmaq_f32(x7, y7, h12v);

                    let yn0 = vfmaq_f32(y0, x0, h21v);
                    let yn1 = vfmaq_f32(y1, x1, h21v);
                    let yn2 = vfmaq_f32(y2, x2, h21v);
                    let yn3 = vfmaq_f32(y3, x3, h21v);
                    let yn4 = vfmaq_f32(y4, x4, h21v);
                    let yn5 = vfmaq_f32(y5, x5, h21v);
                    let yn6 = vfmaq_f32(y6, x6, h21v);
                    let yn7 = vfmaq_f32(y7, x7, h21v);

                    vst1q_f32(py.add(i + 0), yn0);
                    vst1q_f32(py.add(i + 4), yn1);
                    vst1q_f32(py.add(i + 8), yn2);
                    vst1q_f32(py.add(i + 12), yn3);
                    vst1q_f32(py.add(i + 16), yn4);
                    vst1q_f32(py.add(i + 20), yn5);
                    vst1q_f32(py.add(i + 24), yn6);
                    vst1q_f32(py.add(i + 28), yn7);

                    vst1q_f32(px.add(i + 0), xn0);
                    vst1q_f32(px.add(i + 4), xn1);
                    vst1q_f32(px.add(i + 8), xn2);
                    vst1q_f32(px.add(i + 12), xn3);
                    vst1q_f32(px.add(i + 16), xn4);
                    vst1q_f32(px.add(i + 20), xn5);
                    vst1q_f32(px.add(i + 24), xn6);
                    vst1q_f32(px.add(i + 28), xn7);

                    i += 32;
                }

                while i + 4 <= n {
                    let xv = vld1q_f32(px.add(i));
                    let yv = vld1q_f32(py.add(i));

                    let xn = vfmaq_f32(xv, yv, h12v); 
                    let yn = vfmaq_f32(yv, xv, h21v); 

                    vst1q_f32(py.add(i), yn);
                    vst1q_f32(px.add(i), xn);

                    i += 4;
                }

                // tail 
                while i < n {
                    let xi = *px.add(i);
                    let yi = *py.add(i);
                    let xn = xi + param[3] * yi;
                    let yn = param[2] * xi + yi; 
                    *py.add(i) = yn;
                    *px.add(i) = xn;
                    i += 1;
                }
            } else {

                // `flag` = +1  
                // [h11,  1.0]  
                // [-1.0, h22]  
                let h11v = vdupq_n_f32(param[1]);
                let h22v = vdupq_n_f32(param[4]);

                let mut i = 0;
                while i + 32 <= n {
                    let x0 = vld1q_f32(px.add(i + 0));
                    let x1 = vld1q_f32(px.add(i + 4));
                    let x2 = vld1q_f32(px.add(i + 8));
                    let x3 = vld1q_f32(px.add(i + 12));
                    let x4 = vld1q_f32(px.add(i + 16));
                    let x5 = vld1q_f32(px.add(i + 20));
                    let x6 = vld1q_f32(px.add(i + 24));
                    let x7 = vld1q_f32(px.add(i + 28));

                    let y0 = vld1q_f32(py.add(i + 0));
                    let y1 = vld1q_f32(py.add(i + 4));
                    let y2 = vld1q_f32(py.add(i + 8));
                    let y3 = vld1q_f32(py.add(i + 12));
                    let y4 = vld1q_f32(py.add(i + 16));
                    let y5 = vld1q_f32(py.add(i + 20));
                    let y6 = vld1q_f32(py.add(i + 24));
                    let y7 = vld1q_f32(py.add(i + 28));

                    let xn0 = vfmaq_f32(y0, x0, h11v);
                    let xn1 = vfmaq_f32(y1, x1, h11v);
                    let xn2 = vfmaq_f32(y2, x2, h11v);
                    let xn3 = vfmaq_f32(y3, x3, h11v);
                    let xn4 = vfmaq_f32(y4, x4, h11v);
                    let xn5 = vfmaq_f32(y5, x5, h11v);
                    let xn6 = vfmaq_f32(y6, x6, h11v);
                    let xn7 = vfmaq_f32(y7, x7, h11v);

                    let yn0 = vsubq_f32(vmulq_f32(h22v, y0), x0);
                    let yn1 = vsubq_f32(vmulq_f32(h22v, y1), x1);
                    let yn2 = vsubq_f32(vmulq_f32(h22v, y2), x2);
                    let yn3 = vsubq_f32(vmulq_f32(h22v, y3), x3);
                    let yn4 = vsubq_f32(vmulq_f32(h22v, y4), x4);
                    let yn5 = vsubq_f32(vmulq_f32(h22v, y5), x5);
                    let yn6 = vsubq_f32(vmulq_f32(h22v, y6), x6);
                    let yn7 = vsubq_f32(vmulq_f32(h22v, y7), x7);

                    vst1q_f32(py.add(i + 0), yn0);
                    vst1q_f32(py.add(i + 4), yn1);
                    vst1q_f32(py.add(i + 8), yn2);
                    vst1q_f32(py.add(i + 12), yn3);
                    vst1q_f32(py.add(i + 16), yn4);
                    vst1q_f32(py.add(i + 20), yn5);
                    vst1q_f32(py.add(i + 24), yn6);
                    vst1q_f32(py.add(i + 28), yn7);

                    vst1q_f32(px.add(i + 0), xn0);
                    vst1q_f32(px.add(i + 4), xn1);
                    vst1q_f32(px.add(i + 8), xn2);
                    vst1q_f32(px.add(i + 12), xn3);
                    vst1q_f32(px.add(i + 16), xn4);
                    vst1q_f32(px.add(i + 20), xn5);
                    vst1q_f32(px.add(i + 24), xn6);
                    vst1q_f32(px.add(i + 28), xn7);

                    i += 32;
                }

                while i + 4 <= n {
                    let xv = vld1q_f32(px.add(i));
                    let yv = vld1q_f32(py.add(i));

                    let xn = vfmaq_f32(yv, xv, h11v);            
                    let yn = vsubq_f32(vmulq_f32(h22v, yv), xv);

                    vst1q_f32(py.add(i), yn);
                    vst1q_f32(px.add(i), xn);

                    i += 4;
                }

                // tail 
                while i < n {
                    let xi = *px.add(i);
                    let yi = *py.add(i);
                    let xn = param[1] * xi + yi;     
                    let yn = (-xi) + param[4] * yi; 
                    *py.add(i) = yn;
                    *px.add(i) = xn;
                    i += 1;
                }
            }
        }
        return;
    }

    // non unit stride
    unsafe {
        let mut ix = 0; 
        let mut iy = 0;

        if flag < 0.0 {
            let h11 = param[1];
            let h21 = param[2];
            let h12 = param[3];
            let h22 = param[4];

            for _ in 0..n {
                let xi = *px.add(ix);
                let yi = *py.add(iy);

                let xn = h11 * xi + h12 * yi;
                let yn = h21 * xi + h22 * yi;

                *py.add(iy) = yn;
                *px.add(ix) = xn;

                ix += incx;
                iy += incy;
            }
        } else if flag == 0.0 {
            let h12 = param[3];
            let h21 = param[2];

            for _ in 0..n {
                let xi = *px.add(ix);
                let yi = *py.add(iy);

                let xn = xi + h12 * yi;
                let yn = h21 * xi + yi;

                *py.add(iy) = yn;
                *px.add(ix) = xn;

                ix += incx;
                iy += incy;
            }
        } else {
            let h11 = param[1];
            let h22 = param[4];

            for _ in 0..n {
                let xi = *px.add(ix);
                let yi = *py.add(iy);

                let xn = h11 * xi + yi;
                let yn = -xi + h22 * yi;

                *py.add(iy) = yn;
                *px.add(ix) = xn;

                ix += incx;
                iy += incy;
            }
        }
    }
}

