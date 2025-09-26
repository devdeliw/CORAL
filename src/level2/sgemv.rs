//! Single precision GEMV.
//!
//! ```text
//! NoTranspose         => y = alpha * A * x + beta * y
//! Transpose/ConjTrans => y = alpha * A^T * x + beta * y
//! ```
//!
//! `A` is column-major. 
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  (f32)        : Scalar multiplier applied to the product `A * x`.
//! - `matrix` (&[f32])     : Input slice containing the matrix `A`.
//! - `lda`    (usize)      : Leading dimension of `A`.
//! - `x`      (&[f32])     : Input vector of length `n_cols`.
//! - `incx`   (usize)      : Stride between consecutive elements of `x`.
//! - `beta`   (f32)        : Scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f32]) : Input/output vector of length `n_rows`.
//! - `incy`   (usize)      : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place.


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
