//! Single precision complex GEMV.
//!
//! ```text
//! NoTranspose   => y = alpha * A * x + beta * y
//! Transpose     => y = alpha * A^T * x + beta * y
//! ConjTranspose => y = alpha * A^H * x + beta * y
//! ```
//!
//! Dispatches to [`cgemv_notranspose`] or [`cgemv_transpose`] or [`cgemv_conjtranspose`].

use crate::level2::{ 
    enums::CoralTranspose, 
    cgemv_transpose::cgemv_transpose, 
    cgemv_notranspose::cgemv_notranspose, 
    cgemv_conjtranspose::cgemv_conjtranspose
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn cgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f32; 2], 
    matrix  : &[f32], 
    lda     : usize, 
    x       : &[f32], 
    incx    : usize, 
    beta    : [f32; 2], 
    y       : &mut [f32], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => cgemv_notranspose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => cgemv_transpose    (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => cgemv_conjtranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}

