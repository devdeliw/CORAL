//! Single precision complex GEMV.
//!
//! ```text
//! NoTranspose   => y = alpha * A * x + beta * y
//! Transpose     => y = alpha * A^T * x + beta * y
//! ConjTranspose => y = alpha * A^H * x + beta * y
//! ```
//! 
//! `A` is an interleaved column-major matrix. `[re, im, ...]`
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  ([f32; 2])   : Complex scalar multiplier applied to the product `A * x`.
//! - `matrix` (&[f32])     : Input slice containing the interleaved matrix `A`.
//! - `lda`    (usize)      : Leading dimension of `A`; complex units. 
//! - `x`      (&[f32])     : Input complex vector of length `n_cols`.
//! - `incx`   (usize)      : Stride between consecutive complex elements of `x`; complex units. 
//! - `beta`   ([f32; 2])   : Complex scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f32]) : Input/output complex vector of length `n_rows`.
//! - `incy`   (usize)      : Stride between consecutive complex elements of `y`; complex units. 
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place.

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

