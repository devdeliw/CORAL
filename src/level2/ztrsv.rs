//! Performs a double precision complex triangular solve (TRSV).
//!
//! ```text 
//! solves op(A) * x = b for x, where op(A) is A, A^T, or A^H
//! ```
//!
//! This function implements the BLAS [`ztrsv`] routine for both **upper** and **lower**
//! triangular systems.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with `A`, `A^T`, or `A^H`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the interleaved square matrix `A`.
//! - `matrix`      (&[f64])          : Input slice containing the triangular matrix `A`.
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f64])      : Input/output slice containing the right-hand side `b` on
//!                                   | entry and the solution `x` on exit.
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. `x` is updated in place with the solution. 
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
    ztrlsv::ztrlsv, 
    ztrusv::ztrusv,
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn ztrsv( 
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
        CoralTriangular::UpperTriangular => ztrusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => ztrlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}

