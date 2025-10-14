//! Double precision GEMV.
//!
//! ```text
//! NoTranspose         => y = alpha * A * x + beta * y
//! Transpose/ConjTrans => y = alpha * A^T * x + beta * y
//! ```
//!
//! `A` is a column-major matrix.
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  (f64)        : Scalar multiplier applied to the product `A * x`.
//! - `matrix` (&[f64])     : Input slice containing the matrix `A`.
//! - `lda`    (usize)      : Leading dimension of `A`.
//! - `x`      (&[f64])     : Input vector of length `n_cols`.
//! - `incx`   (usize)      : Stride between consecutive elements of `x`.
//! - `beta`   (f64)        : Scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f64]) : Input/output vector of length `n_rows`.
//! - `incy`   (usize)      : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place. 

use crate::enums::CoralTranspose; 
use crate::level2::{ 
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

