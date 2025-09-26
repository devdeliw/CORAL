//! COPY Copies a double precision vector into another.
//!
//! This function implements the BLAS [`dcopy`] routine, copying `n` elements from
//! the input vector `x` into the output vector `y` with specified strides.
//!
//! # Arguments
//! - `n`    (usize)      : Number of elements to copy.
//! - `x`    (&[f64])     : Input slice containing vector elements.
//! - `incx` (usize)      : Stride between consecutive elements of `x`.
//! - `y`    (&mut [f64]) : Output slice to receive copied elements.
//! - `incy` (usize)      : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are overwritten with elements from `x`.
//!
//! # Notes
//! - For `incx == 1 && incy == 1`, [`dcopy`] uses `core::ptr::copy` for fast
//!   contiguous memory copying.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0`, the function returns immediately without modifying `y`.
//!
//! # Author
//! Deval Deliwala

use crate::level1::assert_length_helpers::required_len_ok; 

#[inline(always)]
pub fn dcopy(
    n       : usize,
    x       : &[f64],
    incx    : usize, 
    y       : &mut [f64], 
    incy    : usize
) {
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            core::ptr::copy(x.as_ptr(), y.as_mut_ptr(), n);
            return;
        }

        // non unit stride 
        let mut ix = 0; 
        let mut iy = 0; 
        for _ in 0..n {
            *y.get_unchecked_mut(iy) = *x.get_unchecked(ix);
            ix += incx;
            iy += incy;
        }
    }
}
