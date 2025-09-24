//! Performs a double precision triangular solve (TRSV).
//!
//! This function implements the BLAS [`dtrsv`] routine for both **upper** and **lower**
//! triangular systems, solving the in-place system `op(A) * x = b` for `x`, where `op(A)` is
//! either `A` or `A^T`.
//!
//! Internally, this dispatches to [`dtrusv`] or [`dtrlsv`]
//! depending on the specified `uplo` parameter.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order (dimension) of the square matrix `A`.
//! - `matrix`      (&[f64])          : Input slice containing the triangular matrix `A` in
//!                                   | column-major layout.
//! - `lda`         (usize)           : Leading dimension (stride between columns) of `A`.
//! - `x`           (&mut [f64])      : Input/output slice containing the right-hand side `b` on
//!                                   | entry and the solution `x` on exit (updated in place).
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are overwritten with the solution to `op(A) * x = b`.
//!
//! # Notes
//! - The computation is routed to either [`dtrusv`] or [`dtrlsv`] based on `uplo`.
//! - The kernel is optimized for AArch64 NEON targets and assumes column-major memory layout.
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

