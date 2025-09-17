//! Performs a single precision rank-1 matrix update (GER).
//!
//! This function implements the BLAS [`sger`] routine, computing the outer-product update
//!
//! ```text
//!     A := alpha * x * y^T + A
//! ```
//!
//! where `A` is an `n_rows × n_cols` column-major matrix, `x` is a vector of length
//! `n_rows`, and `y` is a vector of length `n_cols`.  
//!
//! Internally, this uses a fast path for the **unit-stride** case (`incx == 1` and `incy == 1`)
//! that applies a scaled [`saxpy`] into each column, and falls back to a general pointer-walk
//! loop with fused multiply-add (FMA) for arbitrary strides.
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  (f32)        : Scalar multiplier applied to the outer product `x * y^T`.
//! - `x`      (&[f32])     : Input slice containing the vector `x`.
//! - `incx`   (usize)      : Stride between consecutive elements of `x`.
//! - `y`      (&[f32])     : Input slice containing the vector `y`.
//! - `incy`   (usize)      : Stride between consecutive elements of `y`.
//! - `matrix` (&mut [f32]) : Input/output slice containing the matrix `A` in column-major layout,
//!                         | updated in place.
//! - `lda`    (usize)      : Leading dimension (stride between columns) of `A`.
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place as `A := alpha * x * y^T + A`.
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


use crate::level1::saxpy::saxpy;

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn sger( 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : f32, 
    x       : &[f32], 
    incx    : usize, 
    y       : &[f32], 
    incy    : usize, 
    matrix  : &mut [f32], 
    lda     : usize, 
) { 
    // quick return 
    if n_rows == 0 || n_cols == 0 || alpha == 0.0 { return; } 

    debug_assert!(incx > 0 && incy > 0, "incx/incy strides must be nonzero"); 
    debug_assert!(lda >= n_rows, "leading dimension must be >= n_rows"); 
    debug_assert!(required_len_ok(x.len(), n_rows, incx), "x not large enough for n_rows/incx"); 
    debug_assert!(required_len_ok(y.len(), n_cols, incy), "y not large enough for n_cols/incy"); 
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n_rows, n_cols, lda), 
        "matrix not large enough for given n_rows x n_cols and lda"
    ); 

    // fast path 
    if incx == 1 && incy == 1 { 
        // A[:, j] += (alpha * y[j]) * x 
        for col_idx in 0..n_cols { 
            let coeff = alpha * unsafe { *y.get_unchecked(col_idx) }; 
            if coeff != 0.0 { 
                saxpy(n_rows, coeff, x, 1, &mut matrix[col_idx * lda..col_idx * lda + n_rows], 1); 
            }
        }

        return; 
    }

    let mat_ptr = matrix.as_mut_ptr(); 
    let x_ptr   = x.as_ptr(); 
    let y_ptr   = y.as_ptr(); 

    for col_idx in 0..n_cols { 
        let coeff = alpha * unsafe { *y_ptr.add(col_idx * incy) }; 
        if coeff != 0.0 {
            unsafe { 
                let mut mat_col_ptr = mat_ptr.add(col_idx * lda); 

                // contiguous x 
                let mut x_row_ptr = x_ptr; 
                for _ in 0..n_rows { 
                    *mat_col_ptr = (*x_row_ptr).mul_add(coeff, *mat_col_ptr);
                    mat_col_ptr  = mat_col_ptr.add(1);
                    x_row_ptr    = x_row_ptr.add(incx); 
                }
            }
        }
    }
}


