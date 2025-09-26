//! Performs a double precision triangular matrix–vector multiply (TRMV).
//!
//! ```text
//! x := op(A) * x; op(A) is A or A^T
//! ```
//!
//! This function implements the BLAS [`dtrmv`] routine for both **upper** and **lower**
//! triangular matrices, computing the in-place product `x := op(A) * x`, where `op(A)` is
//! either `A` or `A^T`.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to use `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f64])          : Input slice containing the triangular matrix `A`.
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f64])      : Input/output slice containing the vector `x`
//!                                   | updated in place.
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place. 
//!
//! # Notes
//! - The kernel is optimized for AArch64 NEON targets 
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level2::{ 
    dtrlmv::dtrlmv, 
    dtrumv::dtrumv, 
    enums::{CoralDiagonal, CoralTranspose, CoralTriangular},
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn dtrmv( 
    uplo        : CoralTriangular, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    n           : usize, 
    matrix      : &[f64], 
    lda         : usize, 
    x           : &mut [f64], 
    incx        : usize, 
) { 
    match uplo { 
        CoralTriangular::UpperTriangular => dtrumv(n, diagonal, transpose, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => dtrlmv(n, diagonal, transpose, matrix, lda, x, incx), 
    }
}

