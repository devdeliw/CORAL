//! Performs a single precision symmetric rank-1 update (SYR).
//!
//! This function implements the BLAS [`ssyr`] routine, computing
//!
//! ```text
//!     A := alpha * x * x^T + A
//! ```
//!
//! where `A` is an `n x n` **symmetric** column-major matrix and only the triangle
//! indicated by `uplo` is referenced/updated. `x` is a vector of length `n`.
//!
//! Internally, this uses a fast path for the **unit-stride** case (`incx == 1`)
//! that applies a triangular [`saxpy`] into each column, and falls back to a general
//! pointer-walk loop with fused multiply-add (FMA) for arbitrary strides.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Dimension of the matrix `A`.
//! - `alpha`  (f32)             : Scalar multiplier applied to the outer product `x * x^T`.
//! - `x`      (&[f32])          : Input slice containing the vector `x`.
//! - `incx`   (usize)           : Stride between consecutive elements of `x`.
//! - `matrix` (&mut [f32])      : Input/output slice containing the matrix `A` in column-major layout;
//!                              | updated in place (only the specified triangle is touched).
//! - `lda`    (usize)           : Leading dimension (stride between columns) of `A`.
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place as `A := alpha * x * x^T + A`
//!   within the specified triangle.
//!
//! # Notes
//! - Optimized for AArch64 NEON targets; fast path uses SIMD via the level1 [`saxpy`] kernel.
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level1::saxpy::saxpy;
// assert length helpers
use crate::level1::assert_length_helpers::required_len_ok;
use crate::level2::assert_length_helpers::required_len_ok_matrix;

// triangle selector
use crate::level2::enums::CoralTriangular;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn ssyr(
    uplo    : CoralTriangular,
    n       : usize,
    alpha   : f32,
    x       : &[f32],
    incx    : usize,
    matrix  : &mut [f32],
    lda     : usize,
) {
    // quick returns
    if n == 0 || alpha == 0.0 {
        return;
    }

    debug_assert!(incx > 0, "incx stride must be nonzero");
    debug_assert!(lda >= n, "leading dimension must be >= n");
    debug_assert!(required_len_ok(x.len(), n, incx), "x not large enough for n/incx");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not large enough for given n x n and lda"
    );

    // fast path 
    if incx == 1 {
        match uplo {
            CoralTriangular::UpperTriangular => {
                // for each columin j
                // A[0..=j, j] += (alpha * x[j]) * x[0..=j]
                for j in 0..n {
                    let aj = alpha * unsafe { *x.get_unchecked(j) };
                    if aj != 0.0 {
                        // length = j+1
                        let col_start = j * lda;

                        saxpy(
                            j + 1,
                            aj,
                            &x[..],
                            1,
                            &mut matrix[col_start..col_start + (j + 1)],
                            1,
                        );
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                // for each column j 
                // A[j..n, j] += (alpha * x[j]) * x[j..n]
                for j in 0..n {
                    let aj = alpha * unsafe { *x.get_unchecked(j) };
                    if aj != 0.0 {
                        // start at diagonal 
                        // row j 
                        let col_start = j * lda + j; 

                        saxpy(
                            n - j,
                            aj,
                            &x[j..n],
                            1,
                            &mut matrix[col_start..j * lda + n],
                            1,
                        );
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
                    //  aj = alpha * x[j]
                    let aj = alpha * *x_ptr.add(j * incx);

                    if aj != 0.0 {
                        // pointers to top of column j and beginning of x
                        let mut a_col = a_ptr.add(j * lda); 
                        let mut xi = x_ptr;    

                        // length = j + 1
                        for _i in 0..=j {
                            *a_col = (*xi).mul_add(aj, *a_col);
                            a_col  = a_col.add(1);   
                            xi     = xi.add(incx);         
                        }
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                // column j; update rows i = j..n-1
                for j in 0..n {
                    //  aj = alpha * x[j]
                    let aj = alpha * *x_ptr.add(j * incx);

                    if aj != 0.0 {
                        // (row j, col j) 
                        let mut a_col = a_ptr.add(j * lda + j);
                        let mut xi    = x_ptr.add(j * incx);

                        // length = n - j
                        for _i in j..n {
                            *a_col = (*xi).mul_add(aj, *a_col);
                            a_col  = a_col.add(1);      
                            xi     = xi.add(incx);      
                        }
                    }
                }
            }
        }
    }
}

