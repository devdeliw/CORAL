//! Single precision GEMV.
//!
//! ```text
//! NoTranspose         => y = alpha * A * x + beta * y
//! Transpose/ConjTrans => y = alpha * A^T * x + beta * y
//! ```
//!
//! Dispatches to [`sgemv_notranspose`] or [`sgemv_transpose`].

use crate::level2::{ 
    enums::CoralTranspose, 
    sgemv_transpose::sgemv_transpose, 
    sgemv_notranspose::sgemv_notranspose, 
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn sgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : f32, 
    matrix  : &[f32], 
    lda     : usize, 
    x       : &[f32], 
    incx    : usize, 
    beta    : f32, 
    y       : &mut [f32], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => sgemv_notranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => sgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => sgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}
