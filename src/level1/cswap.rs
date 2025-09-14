//! Swaps elements of two complex single precision vectors.
//!
//! This function implements the BLAS [`cswap`] routine, exchanging elements of 
//! two input complex vectors `x` and `y` over `n` entries with specified strides.
//!
//! # Arguments 
//! - `n`    (usize)      : Number of complex elements to swap. 
//! - `x`    (&mut [f32]) : First input/output slice containing interleaved complex vector elements 
//!                       | `[re0, im0, re1, im1, ...]`. 
//! - `incx` (usize)      : Stride between consecutive complex elements of `x` 
//!                       | (measured in complex numbers; every step advances two scalar idxs). 
//! - `y`    (&mut [f32]) : Second input/output slice containing interleaved complex vector elements 
//!                       | `[re0, im0, re1, im1, ...]`. 
//! - `incy` (usize)      : Stride between consecutive complex elements of `y` 
//!                         (measured in complex numbers; every step advances two scalar idxs). 
//!
//! # Returns 
//! - Nothing. The contents of `x` and `y` are swapped in place.
//!
//! # Notes 
//! - For `incx == 1 && incy == 1`, [`cswap`] uses unrolled NEON SIMD instructions 
//!   for optimized performance on AArch64. 
//! - For non-unit or negative strides, the function falls back to scalar iteration. 
//! - If `n == 0`, the function returns immediately without modifying input slices.
//!
//! # Author 
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f32, 
    vst1q_f32, 
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;


#[inline(always)]
#[cfg(target_arch = "aarch64")] 
pub fn cswap(
    n       : usize, 
    x       : &mut [f32],
    incx    : usize, 
    y       : &mut [f32], 
    incy    : usize
) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx > 0 && incy > 0);
    debug_assert!(required_len_ok_cplx(x.len(), n, incx));
    debug_assert!(required_len_ok_cplx(y.len(), n, incy));

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            let len = 2 * n; 
            let px  = x.as_mut_ptr();
            let py  = y.as_mut_ptr();

            let mut i = 0;
            while i + 16 <= len {
                let ax0 = vld1q_f32(px.add(i + 0));
                let ax1 = vld1q_f32(px.add(i + 4));
                let ax2 = vld1q_f32(px.add(i + 8));
                let ax3 = vld1q_f32(px.add(i + 12));

                let ay0 = vld1q_f32(py.add(i + 0));
                let ay1 = vld1q_f32(py.add(i + 4));
                let ay2 = vld1q_f32(py.add(i + 8));
                let ay3 = vld1q_f32(py.add(i + 12));

                vst1q_f32(py.add(i + 0),  ax0);
                vst1q_f32(py.add(i + 4),  ax1);
                vst1q_f32(py.add(i + 8),  ax2);
                vst1q_f32(py.add(i + 12), ax3);

                vst1q_f32(px.add(i + 0),  ay0);
                vst1q_f32(px.add(i + 4),  ay1);
                vst1q_f32(px.add(i + 8),  ay2);
                vst1q_f32(px.add(i + 12), ay3);

                i += 16;
            }

            while i + 4 <= len {
                let ax = vld1q_f32(px.add(i));
                let ay = vld1q_f32(py.add(i));
                vst1q_f32(py.add(i), ax);
                vst1q_f32(px.add(i), ay);
                i += 4;
            }

            // tail
            while i < len {
                let a = *px.add(i);
                *px.add(i) = *py.add(i);
                *py.add(i) = a;
                i += 1;
            }
            return;
        }

        // non unit stride 
        let px = x.as_mut_ptr();
        let py = y.as_mut_ptr();

        let mut ix = 0; 
        let mut iy = 0;

        for _ in 0..n {
            let a0 = *px.add(ix);
            *px.add(ix) = *py.add(iy);
            *py.add(iy) = a0;

            let a1 = *px.add(ix + 1);
            *px.add(ix + 1) = *py.add(iy + 1);
            *py.add(iy + 1) = a1;

            ix += incx * 2;
            iy += incy * 2;
        }
    }
}
