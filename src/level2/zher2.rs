//! Performs a complex double precision Hermitian rank-2 update (HER2).
//!
//! ```text
//!     A := alpha * (x * y^H) + conj(alpha) * (y * x^H) + A
//! ```
//!
//! where `A` is an `n x n` **Hermitian** interleaved column-major matrix, `[re, im, ...]`. 
//! Only the triangle indicated by `uplo` is referenced/updated. `x` and `y` are complex vectors 
//! of length `n`. 
//!
//! Internally, this uses a fast path for the **unit-stride** case (`incx == 1 && incy == 1`)
//! that applies two triangular [`zaxpy`] streams per column, and falls back to a general
//! call with arbitrary strides.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Order of the matrix `A`.
//! - `alpha`  (f64)             : Real scalar multiplier applied to the outer product `x * x^H`.
//! - `x`      (&[f64])          : Input slice containing the complex vector `x`.
//! - `incx`   (usize)           : Stride between consecutive complex elements of `x`.
//! - `y`      (&[f64])          : Input slice containing the complex vector `y`.
//! - `incy`   (usize)           : Stride between consecutive complex elements of `y`.
//! - `matrix` (&mut [f64])      : Input/output slice containing the matrix `A`.
//!                              | specified triangle updated in place.
//! - `lda`    (usize)           : Leading dimension of `A`.
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place within the specified triangle.
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
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;
use crate::enums::CoralTriangular;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn zher2(
    uplo    : CoralTriangular,
    n       : usize,
    alpha   : [f64; 2],
    x       : &[f64],
    incx    : usize,
    y       : &[f64],
    incy    : usize,
    matrix  : &mut [f64],
    lda     : usize,
) {
    // quick returns
    if n == 0 || (alpha[0] == 0.0 && alpha[1] == 0.0) {
        return;
    }

    debug_assert!(incx > 0 && incy > 0, "incx/incy strides must be nonzero");
    debug_assert!(lda >= n, "leading dimension must be >= n");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x not large enough for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y not large enough for n/incy");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not large enough for given n x n and lda"
    );

    // fast path 
    if incx == 1 && incy == 1 {
        match uplo {
            CoralTriangular::UpperTriangular => {
                // for each column j
                // A[0..=j, j] += (alpha * conj(y[j])) * x[0..=j]  
                //             +  (conj(alpha) * conj(x[j])) * y[0..=j]
                for j in 0..n {
                    let xjr = unsafe { *x.get_unchecked(2*j) };
                    let xji = unsafe { *x.get_unchecked(2*j + 1) };
                    let yjr = unsafe { *y.get_unchecked(2*j) };
                    let yji = unsafe { *y.get_unchecked(2*j + 1) };

                    // aj_y = alpha * conj(y[j]) = (ar+iai)*(yjr - i yji)
                    let ar = alpha[0];
                    let ai = alpha[1];
                    let aj_y = [ar * yjr + ai * yji, -ar * yji + ai * yjr];

                    // aj_x = conj(alpha) * conj(x[j]) = (ar - i ai)*(xjr - i xji)
                    let aj_x = [ar * xjr - ai * xji, -ar * xji - ai * xjr];

                    // length = j+1
                    let col_start = j * lda;

                    if aj_y[0] != 0.0 || aj_y[1] != 0.0 {
                        zaxpy(
                            j + 1,
                            aj_y,
                            &x[..],
                            1,
                            &mut matrix[2*col_start .. 2*(col_start + (j + 1))],
                            1,
                        );
                    }
                    if aj_x[0] != 0.0 || aj_x[1] != 0.0 {
                        zaxpy(
                            j + 1,
                            aj_x,
                            &y[..],
                            1,
                            &mut matrix[2*col_start .. 2*(col_start + (j + 1))],
                            1,
                        );
                    }

                    // force diagonal imaginary part to zero
                    matrix[2*(j * lda + j) + 1] = 0.0;
                }
            }
            CoralTriangular::LowerTriangular => {
                // for each column j 
                // A[j..n, j] += (alpha * conj(y[j])) * x[j..n] 
                //            +  (conj(alpha) * conj(x[j])) * y[j..n]
                for j in 0..n {
                    let xjr = unsafe { *x.get_unchecked(2*j) };
                    let xji = unsafe { *x.get_unchecked(2*j + 1) };
                    let yjr = unsafe { *y.get_unchecked(2*j) };
                    let yji = unsafe { *y.get_unchecked(2*j + 1) };

                    let ar = alpha[0];
                    let ai = alpha[1];
                    let aj_y = [ar * yjr + ai * yji, -ar * yji + ai * yjr];
                    let aj_x = [ar * xjr - ai * xji, -ar * xji - ai * xjr];

                    // start at diagonal 
                    // row j 
                    let col_start = j * lda + j; 

                    if aj_y[0] != 0.0 || aj_y[1] != 0.0 {
                        zaxpy(
                            n - j,
                            aj_y,
                            &x[2*j..],
                            1,
                            &mut matrix[2*col_start .. 2*(j * lda + n)],
                            1,
                        );
                    }
                    if aj_x[0] != 0.0 || aj_x[1] != 0.0 {
                        zaxpy(
                            n - j,
                            aj_x,
                            &y[2*j..],
                            1,
                            &mut matrix[2*col_start .. 2*(j * lda + n)],
                            1,
                        );
                    }

                    // force diagonal imaginary part to zero
                    matrix[2*(j * lda + j) + 1] = 0.0;
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
                // column j; update rows i = 0..=j
                for j in 0..n {
                    //  aj_y = alpha * conj(y[j]); 
                    //  aj_x = conj(alpha) * conj(x[j])
                    let by   = 2 * j * incy;
                    let bx   = 2 * j * incx;
                    let yjr  = *y_ptr.add(by);
                    let yji  = *y_ptr.add(by + 1);
                    let xjr  = *x_ptr.add(bx);
                    let xji  = *x_ptr.add(bx + 1);

                    let ar = alpha[0];
                    let ai = alpha[1];

                    let aj_y = [ar * yjr + ai * yji, -ar * yji + ai * yjr];
                    let aj_x = [ar * xjr - ai * xji, -ar * xji - ai * xjr];

                    if (aj_y[0] != 0.0 || aj_y[1] != 0.0) || (aj_x[0] != 0.0 || aj_x[1] != 0.0) {
                        let col_start = j * lda; 

                        // length = j + 1
                        if aj_y[0] != 0.0 || aj_y[1] != 0.0 {
                            zaxpy(
                                j + 1,
                                aj_y,
                                std::slice::from_raw_parts(x_ptr, x.len()),
                                incx,
                                std::slice::from_raw_parts_mut(
                                    a_ptr.add(2*col_start),
                                    2*(j + 1)
                                ),
                                1,
                            );
                        }
                        if aj_x[0] != 0.0 || aj_x[1] != 0.0 {
                            zaxpy(
                                j + 1,
                                aj_x,
                                std::slice::from_raw_parts(y_ptr, y.len()),
                                incy,
                                std::slice::from_raw_parts_mut(
                                    a_ptr.add(2*col_start),
                                    2*(j + 1)
                                ),
                                1,
                            );
                        }

                        // force diagonal imaginary part to zero
                        *a_ptr.add(2*(j * lda + j) + 1) = 0.0;
                    }
                }
            }
            CoralTriangular::LowerTriangular => {
                // column j; update rows i = j..n-1
                for j in 0..n {
                    //  aj_y = alpha * conj(y[j]); 
                    //  aj_x = conj(alpha) * conj(x[j])
                    let by   = 2 * j * incy;
                    let bx   = 2 * j * incx;
                    let yjr  = *y_ptr.add(by);
                    let yji  = *y_ptr.add(by + 1);
                    let xjr  = *x_ptr.add(bx);
                    let xji  = *x_ptr.add(bx + 1);

                    let ar = alpha[0];
                    let ai = alpha[1];

                    let aj_y = [ar * yjr + ai * yji, -ar * yji + ai * yjr];
                    let aj_x = [ar * xjr - ai * xji, -ar * xji - ai * xjr];

                    if (aj_y[0] != 0.0 || aj_y[1] != 0.0) || (aj_x[0] != 0.0 || aj_x[1] != 0.0) {
                        // (row j, col j) 
                        let col_start = j * lda + j;

                        // length = n - j
                        if aj_y[0] != 0.0 || aj_y[1] != 0.0 {
                            zaxpy(
                                n - j,
                                aj_y,
                                std::slice::from_raw_parts(
                                    x_ptr.add(2*j*incx),
                                    x.len() - 2*j*incx
                                ),
                                incx,
                                std::slice::from_raw_parts_mut(
                                    a_ptr.add(2*col_start),
                                    2*(n - j)
                                ),
                                1,
                            );
                        }
                        if aj_x[0] != 0.0 || aj_x[1] != 0.0 {
                            zaxpy(
                                n - j,
                                aj_x,
                                std::slice::from_raw_parts(
                                    y_ptr.add(2*j*incy), 
                                    y.len() - 2*j*incy
                                ),
                                incy,
                                std::slice::from_raw_parts_mut(
                                    a_ptr.add(2*col_start),
                                    2*(n - j)
                                ),
                                1,
                            );
                        }

                        // force diagonal imaginary part to zero
                        *a_ptr.add(2*(j * lda + j) + 1) = 0.0;
                    }
                }
            }
        }
    }
}

