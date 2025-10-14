//! Performs a double precision complex triangular matrix–vector multiply (ZTRMV).
//!
//! ```text
//! x := op(A) * x,  op(A) is A, A^T, or A^H
//! ```
//!
//! This function implements the BLAS [`ztrmv`] routine for both **upper** and **lower**
//! triangular matrices. 
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to use `A`, `A^T`, or `A^H`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f64])          : Input slice containing the triangular matrix `A`. 
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f64])      : Input/output slice containing the vector `x`.
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

use crate::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use crate::level2::{ 
    ztrlmv::ztrlmv, 
    ztrumv::ztrumv, 
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn ztrmv( 
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
        CoralTriangular::UpperTriangular => ztrumv(n, diagonal, transpose, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => ztrlmv(n, diagonal, transpose, matrix, lda, x, incx), 
    }
}

