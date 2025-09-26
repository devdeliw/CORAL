//! Performs a double precision triangular solve (TRSV).
//!
//! ```text
//! solves op(A) * x = b for x, op(A) is A or A^T 
//! ```
//!
//! This function implements the BLAS [`dtrsv`] routine for both **upper** and **lower**
//! triangular systems. 
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f64])          : Input slice containing the triangular matrix `A`
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f64])      : Input/output slice containing the right-hand side `b` on
//!                                   | entry and the solution `x` on exit.
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
    enums::{CoralTriangular, CoralDiagonal, CoralTranspose}, 
    dtrlsv::dtrlsv, 
    dtrusv::dtrusv,
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn dtrsv( 
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
        CoralTriangular::UpperTriangular => dtrusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => dtrlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}

