//! Double precision complex GEMV.
//!
//! ```text
//! NoTranspose   => y = alpha * A * x + beta * y
//! Transpose     => y = alpha * A^T * x + beta * y
//! ConjTranspose => y = alpha * A^H * x + beta * y
//! ```
//!
//! Dispatches to [`zgemv_notranspose`] or [`zgemv_transpose`] or [`zgemv_conjtranspose`].

use crate::level2::{ 
    enums::CoralTranspose, 
    zgemv_notranspose::zgemv_notranspose, 
    zgemv_transpose::zgemv_transpose, 
    zgemv_conjtranspose::zgemv_conjtranspose
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn zgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f64; 2], 
    matrix  : &[f64], 
    lda     : usize, 
    x       : &[f64], 
    incx    : usize, 
    beta    : [f64; 2], 
    y       : &mut [f64], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => zgemv_notranspose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => zgemv_transpose    (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => zgemv_conjtranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}


