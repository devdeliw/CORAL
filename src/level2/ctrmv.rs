//! Performs a single precision complex triangular matrix–vector multiply (CTRMV).
//!
//! ```text
//! x := op(A) * x,  op(A) is A, A^T, or A^H
//! ```
//!
//! This function implements the BLAS [`ctrmv`] routine for both **upper** and **lower**
//! triangular matrices. 
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to use `A`, `A^T`, or `A^H`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f32])          : Input slice containing the interleaved triangular complex matrix `A`.
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f32])      : Input/output slice containing the interleaved complex vector `x`.
//! - `incx`        (usize)           : Stride between consecutive complex elements of `x`.
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
    ctrlmv::ctrlmv, 
    ctrumv::ctrumv, 
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn ctrmv( 
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
        CoralTriangular::UpperTriangular => ctrumv(n, diagonal, transpose, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => ctrlmv(n, diagonal, transpose, matrix, lda, x, incx), 
    }
}

