//! Performs a single precision triangular matrix–vector multiply (TRMV).
//!
//! This function implements the BLAS [`strmv`] routine for both **upper** and **lower**
//! triangular matrices, computing the in-place product `x := op(A) * x`, where `op(A)` is
//! either `A` or `A^T`.
//!
//! Internally, this dispatches to [`strumv`] or [`strlmv`]
//! depending on the specified `uplo` parameter.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to use `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order (dimension) of the square matrix `A`.
//! - `matrix`      (&[f32])          : Input slice containing the triangular matrix `A` in
//!                                   | column-major layout.
//! - `lda`         (usize)           : Leading dimension (stride between columns) of `A`.
//! - `x`           (&mut [f32])      : Input/output slice containing the vector `x`, updated in place.
//! - `incx`        (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place as `x := op(A) * x`.
//!
//! # Notes
//! - The computation is routed to either [`strumv`] or [`strlmv`] based on `uplo`.
//! - The kernel is optimized for AArch64 NEON targets and assumes column-major memory layout.
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
