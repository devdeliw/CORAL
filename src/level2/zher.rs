//! Performs a complex double precision Hermitian rank-1 update (HER).
//!
//! This function implements the BLAS [`zher`] routine, computing
//!
//! ```text
//!     A := alpha * x * x^H + A
//! ```
//!
//! where `A` is an `n x n` **Hermitian** column-major matrix and only the triangle
//! indicated by `uplo` is referenced/updated. `x` is a complex vector of length `n`
//! stored as interleaved `[re, im, re, im, ...]`.
//!
//! Internally, this uses a fast path for the **unit-stride** case (`incx == 1`)
//! that applies a triangular [`zaxpy`] into each column, and falls back to a general
//! call with arbitrary strides.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Dimension of the matrix `A`.
//! - `alpha`  (f64)             : Real scalar multiplier applied to the outer product `x * x^H`.
//! - `x`      (&[f64])          : Input slice containing the complex vector `x`; interleaved.
//! - `incx`   (usize)           : Stride between consecutive complex elements of `x`.
//! - `matrix` (&mut [f64])      : Input/output slice containing the matrix `A` in column-major layout
//!                              | as interleaved complex scalars; specified triangle updated in place.
//! - `lda`    (usize)           : Leading dimension in complex elements of `A`.
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place
//!   within the specified triangle.
//!
//! # Notes
//! - Optimized for AArch64 NEON targets; fast path uses SIMD via the level1 [`zaxpy`] kernel.
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level1::zaxpy::zaxpy;
// assert length helpers
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;
use crate::level2::enums::CoralTriangular;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn zher(
    uplo    : CoralTriangular,
    n       : usize,
    alpha   : f64,
    x       : &[f64],
    incx    : usize,
    matrix  : &mut [f64],
    lda     : usize,
) {
    // quick returns
    if n == 0 || alpha == 0.0 {
        return;
    }

    debug_assert!(incx > 0, "incx stride must be nonzero");
    debug_assert!(lda >= n, "leading dimension must be >= n");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x not large enough for n/incx");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not large enough for given n x n and lda"
    );

    // fast path 
    if incx == 1 {
        match uplo {
            CoralTriangular::UpperTriangular => {
                // for each column j
                // A[0..=j, j] += (alpha * conj(x[j])) * x[0..=j]
                for j in 0..n {
                    let xjr = unsafe { *x.get_unchecked(2*j) };
                    let xji = unsafe { *x.get_unchecked(2*j + 1) };
                    let aj  = [alpha * xjr, -alpha * xji];
                    if aj[0] != 0.0 || aj[1] != 0.0 {
                        // length = j+1
                        let col_start = j * lda;

                        zaxpy(
                            j + 1,
                            aj,
                            &x[..],
                            1,
                            &mut matrix[2*col_start .. 2*(col_start + (j + 1))],
                            1,
                        );

                        // force diagonal imaginary part to zero
                        matrix[2*(j * lda + j) + 1] = 0.0;
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                // for each column j 
                // A[j..n, j] += (alpha * conj(x[j])) * x[j..n]
                for j in 0..n {
                    let xjr = unsafe { *x.get_unchecked(2*j) };
                    let xji = unsafe { *x.get_unchecked(2*j + 1) };
                    let aj  = [alpha * xjr, -alpha * xji];
                    if aj[0] != 0.0 || aj[1] != 0.0 {
                        // start at diagonal 
                        // row j 
                        let col_start = j * lda + j; 

                        zaxpy(
                            n - j,
                            aj,
                            &x[2*j..],
                            1,
                            &mut matrix[2*col_start .. 2*(j * lda + n)],
                            1,
                        );

                        // force diagonal imaginary part to zero
                        matrix[2*(j * lda + j) + 1] = 0.0;
                    }
                }
            }
        }
        return;
    }

    // general path 
    let a_ptr = matrix.as_mut_ptr();
    let x_ptr = x.as_ptr();

    unsafe {
        match uplo {
            CoralTriangular::UpperTriangular => {
                // column j; update rows i = 0..=j
                for j in 0..n {
                    //  aj = alpha * conj(x[j])
                    let base = 2 * j * incx;
                    let xjr  = *x_ptr.add(base);
                    let xji  = *x_ptr.add(base + 1);
                    let aj   = [alpha * xjr, -alpha * xji];

                    if aj[0] != 0.0 || aj[1] != 0.0 {
                        // pointers to top of column j and beginning of x
                        let col_start = j * lda; 

                        // length = j + 1
                        zaxpy(
                            j + 1,
                            aj,
                            std::slice::from_raw_parts(x_ptr, x.len()),
                            incx,
                            std::slice::from_raw_parts_mut(
                                a_ptr.add(2*col_start),
                                2*(j + 1)
                            ),
                            1,
                        );

                        // force diagonal imaginary part to zero
                        *a_ptr.add(2*(j * lda + j) + 1) = 0.0;
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                // column j; update rows i = j..n-1
                for j in 0..n {
                    //  aj = alpha * conj(x[j])
                    let base = 2 * j * incx;
                    let xjr  = *x_ptr.add(base);
                    let xji  = *x_ptr.add(base + 1);
                    let aj   = [alpha * xjr, -alpha * xji];

                    if aj[0] != 0.0 || aj[1] != 0.0 {
                        // (row j, col j) 
                        let col_start = j * lda + j;

                        // length = n - j
                        zaxpy(
                            n - j,
                            aj,
                            std::slice::from_raw_parts(x_ptr.add(2*j*incx), x.len() - 2*j*incx),
                            incx,
                            std::slice::from_raw_parts_mut(
                                a_ptr.add(2*col_start),
                                2*(n - j)
                            ),
                            1,
                        );

                        // force diagonal imaginary part to zero
                        *a_ptr.add(2*(j * lda + j) + 1) = 0.0;
                    }
                }
            }
        }
    }
}

