//! Double precision GEMV.
//!
//! ```text
//! NoTranspose         => y = alpha * A * x + beta * y
//! Transpose/ConjTrans => y = alpha * A^T * x + beta * y
//! ```
//!
//! Dispatches to [`sgemv_notranspose`] or [`sgemv_transpose`].

use crate::level2::{ 
    enums::CoralTranspose, 
    dgemv_transpose::dgemv_transpose, 
    dgemv_notranspose::dgemv_notranspose, 
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn dgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : f64, 
    matrix  : &[f64], 
    lda     : usize, 
    x       : &[f64], 
    incx    : usize, 
    beta    : f64, 
    y       : &mut [f64], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => dgemv_notranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => dgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => dgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}

