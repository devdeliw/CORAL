//! Performs a complex double precision rank-1 matrix update (GERC).
//!
//! This function implements the BLAS [`zgerc`] routine, computing the outer-product update
//!
//! ```text
//!     A := alpha * x * y^H + A
//! ```
//!
//! where `A` is an `n_rows x n_cols` column-major matrix, `x` is a vector of length
//! `n_rows`, and `y` is a vector of length `n_cols`.  
//!
//! Internally, this uses a fast path for the **unit-stride** case (`incx == 1` and `incy == 1`)
//! that applies a scaled [`zaxpy`] into each column, and falls back to a general pointer-walk
//! loop with fused multiply-add (FMA) for arbitrary strides.
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  ([f64; 2])   : Complex scalar multiplier applied to the outer product `x * y^H`.
//! - `x`      (&[f64])     : Input slice containing interleaved complex vector `x` elements.
//! - `incx`   (usize)      : Stride between consecutive complex elements of `x`.
//! - `y`      (&[f64])     : Input slice containing interleaved complex vector `y` elements.
//! - `incy`   (usize)      : Stride between consecutive complex elements of `y`.
//! - `matrix` (&mut [f64]) : Input slice containing interleaved complex matrix `A`; updated in place. 
//! - `lda`    (usize)      : Leading dimension of `A`.
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place as `A := alpha * x * y^H + A`.
//!
//! # Notes
//! - Optimized for AArch64 NEON targets; fast path uses SIMD, have not made portable.
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

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn zgerc( 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f64; 2], 
    x       : &[f64], 
    incx    : usize, 
    y       : &[f64], 
    incy    : usize, 
    matrix  : &mut [f64], 
    lda     : usize, 
) { 
    // quick return 
    if n_rows == 0 || n_cols == 0 || (alpha[0] == 0.0 && alpha[1] == 0.0) { return; } 

    debug_assert!(incx > 0 && incy > 0, "incx/incy strides must be nonzero"); 
    debug_assert!(lda >= n_rows, "leading dimension must be >= n_rows"); 
    debug_assert!(required_len_ok_cplx(x.len(), n_rows, incx), "x not large enough for n_rows/incx"); 
    debug_assert!(required_len_ok_cplx(y.len(), n_cols, incy), "y not large enough for n_cols/incy"); 
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n_rows, n_cols, lda), 
        "matrix not large enough for given n_rows x n_cols and lda"
    ); 

    // fast path 
    if incx == 1 && incy == 1 { 
        // A[:, j] += (alpha * conj(y[j])) * x 
        for col_idx in 0..n_cols { 
            let y_re = unsafe { *y.get_unchecked(2 * col_idx) }; 
            let y_im = unsafe { *y.get_unchecked(2 * col_idx + 1) }; 
            let coeff = [ 
                alpha[0] * y_re + alpha[1] * y_im, 
                -alpha[0] * y_im + alpha[1] * y_re, 
            ]; 
            if coeff[0] != 0.0 || coeff[1] != 0.0 { 
                zaxpy(
                    n_rows, 
                    coeff,
                    x,
                    1,
                    &mut matrix[2 * col_idx * lda .. 2 * col_idx * lda + 2 * n_rows],
                    1
                ); 
            }
        }

        return; 
    }

    // general path 
    let mat_ptr = matrix.as_mut_ptr(); 
    let x_ptr   = x.as_ptr(); 
    let y_ptr   = y.as_ptr(); 

    for col_idx in 0..n_cols { 
        let y_off = col_idx * incy * 2; 
        let y_re  = unsafe { *y_ptr.add(y_off) }; 
        let y_im  = unsafe { *y_ptr.add(y_off + 1) }; 
        let coeff_re = alpha[0] * y_re + alpha[1] * y_im; 
        let coeff_im = -alpha[0] * y_im + alpha[1] * y_re; 

        if coeff_re != 0.0 || coeff_im != 0.0 {
            unsafe { 
                let mut mat_col_ptr = mat_ptr.add(2 * col_idx * lda); 

                let mut x_row_ptr = x_ptr; 
                for _ in 0..n_rows { 
                    let xr = *x_row_ptr; 
                    let xi = *x_row_ptr.add(1); 

                    let pr = xr * coeff_re - xi * coeff_im; 
                    let pi = xr * coeff_im + xi * coeff_re; 

                    *mat_col_ptr       = *mat_col_ptr       + pr;
                    *mat_col_ptr.add(1)= *mat_col_ptr.add(1)+ pi;

                    mat_col_ptr  = mat_col_ptr.add(2);
                    x_row_ptr    = x_row_ptr.add(2 * incx); 
                }
            }
        }
    }
}

