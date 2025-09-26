//! Performs a single precision triangular matrix–vector multiply (TRMV).
//!
//! ```text
//! x := op(A) * x, op(A) is A or A^T 
//! ```
//!
//! This function implements the BLAS [`strmv`] routine for both **upper** and **lower**
//! triangular matrices. 
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to use `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f32])          : Input slice containing the triangular matrix `A`. 
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f32])      : Input/output slice containing the vector `x`.
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place. 
//!
//! # Notes
//! - The computation is routed to either [`strumv`] or [`strlmv`] based on `uplo`.
//! - The kernel is optimized for AArch64 NEON targets 
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level2::{ 
    strlmv::strlmv, 
    strumv::strumv, 
    enums::{CoralDiagonal, CoralTranspose, CoralTriangular},
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn strmv( 
    uplo        : CoralTriangular, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    n           : usize, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    match uplo { 
        CoralTriangular::UpperTriangular => strumv(n, diagonal, transpose, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => strlmv(n, diagonal, transpose, matrix, lda, x, incx), 
    }
}
