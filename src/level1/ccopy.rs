//! Copies a complex single precision vector into another.
//!
//! This function implements the BLAS [`ccopy`] routine, copying `n` complex elements from
//! the input vector `x` into the output vector `y` with specified strides.
//!
//! # Arguments
//! - `n`    (usize)      : Number of complex elements to copy.
//! - `x`    (&[f32])     : Input slice containing interleaved complex vector elements
//!                       | `[re0, im0, re1, im1, ...]`.
//! - `incx` (usize)      : Stride between consecutive complex elements of `x`
//!                       | (measured in complex numbers; every step advances two scalar idxs).
//! - `y`    (&mut [f32]) : Output slice to receive copied complex elements
//!                       `[re0, im0, re1, im1, ...]`.
//! - `incy` (usize)      : Stride between consecutive complex elements of `y`
//!                       (measured in complex numbers; every step advances two scalar idxs).
//!
//! # Returns
//! - Nothing. The contents of `y` are overwritten with elements from `x`.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`ccopy`] uses `core::ptr::copy_nonoverlapping` for fast
//!   contiguous memory copying of real and imag parts.
//! - For non unit or negative strides, the function falls back to a scalar loop.
//! - If `n == 0`, the function returns immediately without modifying `y`.
//!
//! # Author
//! Deval Deliwala


use crate::level1::assert_length_helpers::required_len_ok_cplx; 

#[inline(always)]
pub fn ccopy(
    n       : usize, 
    x       : &[f32],
    incx    : usize,
    y       : &mut [f32],
    incy    : usize
) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy (complex)");

    unsafe {
        if incx == 1 && incy == 1 {
            core::ptr::copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), 2 * n);
            return;
        }

        let mut ix = 0; 
        let mut iy = 0; 
        for _ in 0..n {
            *y.get_unchecked_mut(iy)     = *x.get_unchecked(ix);
            *y.get_unchecked_mut(iy + 1) = *x.get_unchecked(ix + 1);
            ix += incx * 2;
            iy += incy * 2;
        }
    }
}
