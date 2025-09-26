//! Performs a double precision symmetric rank-2 update (SYR2).
//!
//! BLAS [`dsyr2`] computes
//!
//! ```text
//!     A := alpha * (x * y^T + y * x^T) + A
//! ```
//!
//! where `A` is an `n x n` **symmetric** column-major matrix and only the triangle
//! indicated by `uplo` is referenced/updated.
//!
//! Fast path for unit strides uses two triangular [`daxpy`] streams per column.
//! General path uses a pointer-walk for arbitrary strides.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Dimension of the matrix `A`.
//! - `alpha`  (f64)             : Scalar multiplier applied to the outer product `x * x^T`.
//! - `x`      (&[f64])          : Input slice containing the vector `x`.
//! - `incx`   (usize)           : Stride between consecutive elements of `x`.
//! - `y`      (&[f64])          : Input slice containing the vector `y`
//! - `incy`   (usize)           : Stride between consecutive elements of `y`.
//! - `matrix` (&mut [f64])      : Input/output slice containing the matrix `A`.
//!                              | updated in place. 
//! - `lda`    (usize)           : Leading dimension of `A`.
//
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place within the specified triangle.
//!
//! # Notes
//! - Optimized for AArch64 NEON targets; fast path uses SIMD via the level1 [`daxpy`] kernel.
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub
//!
//! # Author
//! Deval Deliwala

use crate::level1::daxpy::daxpy;
use crate::level1::assert_length_helpers::required_len_ok;
use crate::level2::assert_length_helpers::required_len_ok_matrix;
use crate::level2::enums::CoralTriangular;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn dsyr2(
    uplo    : CoralTriangular,
    n       : usize,
    alpha   : f64,
    x       : &[f64],
    incx    : usize,
    y       : &[f64],
    incy    : usize,
    matrix  : &mut [f64],
    lda     : usize,
) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    debug_assert!(incx > 0 && incy > 0, "incx/incy strides must be nonzero");
    debug_assert!(lda >= n, "leading dimension must be >= n");
    debug_assert!(required_len_ok(x.len(), n, incx), "x not large enough for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y not large enough for n/incy");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not large enough for given n x n and lda"
    );

    // fast path
    if incx == 1 && incy == 1 {
        match uplo {
            CoralTriangular::UpperTriangular => {
                for j in 0..n {
                    let aj_y = alpha * unsafe { *y.get_unchecked(j) };
                    let aj_x = alpha * unsafe { *x.get_unchecked(j) };
                    if aj_y != 0.0 {
                        let col_start = j * lda;
                        daxpy(
                            j + 1,
                            aj_y,
                            &x[..],
                            1,
                            &mut matrix[col_start..col_start + (j + 1)],
                            1,
                        );
                    }
                    if aj_x != 0.0 {
                        let col_start = j * lda;
                        daxpy(
                            j + 1,
                            aj_x,
                            &y[..],
                            1,
                            &mut matrix[col_start..col_start + (j + 1)],
                            1,
                        );
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                for j in 0..n {
                    let aj_y = alpha * unsafe { *y.get_unchecked(j) };
                    let aj_x = alpha * unsafe { *x.get_unchecked(j) };
                    if aj_y != 0.0 {
                        let col_start = j * lda + j;
                        daxpy(
                            n - j,
                            aj_y,
                            &x[j..n],
                            1,
                            &mut matrix[col_start..j * lda + n],
                            1,
                        );
                    }
                    if aj_x != 0.0 {
                        let col_start = j * lda + j;
                        daxpy(
                            n - j,
                            aj_x,
                            &y[j..n],
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
    let y_ptr = y.as_ptr();

    unsafe {
        match uplo {
            CoralTriangular::UpperTriangular => {
                for j in 0..n {
                    let aj_y = alpha * *y_ptr.add(j * incy);
                    let aj_x = alpha * *x_ptr.add(j * incx);
                    if aj_y == 0.0 && aj_x == 0.0 {
                        continue;
                    }

                    let mut a_col = a_ptr.add(j * lda);
                    let mut xi    = x_ptr;
                    let mut yi    = y_ptr;

                    for _i in 0..=j {
                        *a_col = (*xi).mul_add(aj_y, *a_col);
                        *a_col = (*yi).mul_add(aj_x, *a_col);
                        a_col  = a_col.add(1);
                        xi     = xi.add(incx);
                        yi     = yi.add(incy);
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                for j in 0..n {
                    let aj_y = alpha * *y_ptr.add(j * incy);
                    let aj_x = alpha * *x_ptr.add(j * incx);
                    if aj_y == 0.0 && aj_x == 0.0 {
                        continue;
                    }

                    let mut a_col = a_ptr.add(j * lda + j);
                    let mut xi    = x_ptr.add(j * incx);
                    let mut yi    = y_ptr.add(j * incy);

                    for _i in j..n {
                        *a_col = (*xi).mul_add(aj_y, *a_col);
                        *a_col = (*yi).mul_add(aj_x, *a_col);
                        a_col  = a_col.add(1);
                        xi     = xi.add(incx);
                        yi     = yi.add(incy);
                    }
                }
            }
        }
    }
}

