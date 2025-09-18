//! Performs a single precision triangular solve (TRSV).
//!
//! This function implements the BLAS [`strsv`] routine for both **upper** and **lower**
//! triangular systems, solving the in-place system `op(A) * x = b` for `x`, where `op(A)` is
//! either `A` or `A^T`.
//!
//! Internally, this dispatches to [`strusv`] or [`strlsv`]
//! depending on the specified `uplo` parameter.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order (dimension) of the square matrix `A`.
//! - `matrix`      (&[f32])          : Input slice containing the triangular matrix `A` in
//!                                   | column-major layout.
//! - `lda`         (usize)           : Leading dimension (stride between columns) of `A`.
//! - `x`           (&mut [f32])      : Input/output slice containing the right-hand side `b` on
//!                                   | entry and the solution `x` on exit (updated in place).
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are overwritten with the solution to `op(A) * x = b`.
//!
//! # Notes
//! - The computation is routed to either [`strusv`] or [`strlsv`] based on `uplo`.
//! - The kernel is optimized for AArch64 NEON targets and assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level2::{
    enums::{CoralTriangular, CoralDiagonal, CoralTranspose}, 
    strlsv::strlsv, 
    strusv::strusv,
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn strsv( 
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
        CoralTriangular::UpperTriangular => strusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => strlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}
