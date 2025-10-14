//! Performs a single precision triangular solve (TRSV).
//!
//! ```text 
//! solves op(A) * x = b for x, op(A) is A or A^T 
//! ```
//!
//! This function implements the BLAS [`strsv`] routine for both **upper** and **lower**
//! triangular systems.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether `A` is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with `A` or `A^T`.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix `A`.
//! - `matrix`      (&[f32])          : Input slice containing the triangular matrix `A` in
//!                                   | column-major layout.
//! - `lda`         (usize)           : Leading dimension of `A`.
//! - `x`           (&mut [f32])      : Input/output slice containing the right-hand side `b` on
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

use crate::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use crate::level2::{
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
