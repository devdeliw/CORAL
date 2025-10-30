//! `GER`. Performs a complex single precision rank-1 matrix update.
//!
//! This function implements the BLAS [`cgerc`] routine, computing the outer-product update
//!
//! \\[ 
//! A := \alpha x y^{H} + A. 
//! \\]
//!
//! where $A$ is an `n_rows x n_cols` interleaved column-major matrix, `[re, im, ...]`, 
//! $x$ is a vector of length `n_rows`, and $y$ is a vector of length `n_cols`.  
//!
//! Internally, this uses a fast path for the unit-stride case (`incx == 1` and `incy == 1`)
//! that applies a scaled [`caxpy`] into each column, and falls back to a general pointer-walk
//! loop for arbitrary strides.
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows ($m$) in the matrix $A$.
//! - `n_cols` (usize)      : Number of columns ($n$) in the matrix $A$.
//! - `alpha`  ([f32; 2])   : Complex scalar multiplier applied to the outer product $x y^H$.
//! - `x`      (&[f32])     : Input slice containing interleaved complex vector $x$ elements.
//! - `incx`   (usize)      : Stride between consecutive complex elements of $x$.
//! - `y`      (&[f32])     : Input slice containing interleaved complex vector $y$ elements.
//! - `incy`   (usize)      : Stride between consecutive complex elements of $y$. 
//! - `matrix` (&mut [f32]) : Input slice containing interleaved complex matrix $A$.
//! - `lda`    (usize)      : Leading dimension of $A$; complex units. 
//!
//! # Returns
//! - Nothing. The contents of `matrix` are updated in place. 
//!
//! # Author
//! Deval Deliwala
//! 
//! # Example
//! ```rust
//! use coral::level2::cgerc;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!
//!     let alpha = [0.5, 0.0];
//!     let x     = vec![1.0, 0.0, 0.0, 1.0];  // (1, i)
//!     let incx  = 1;
//!     let y     = vec![1.0, -1.0, 2.0, 0.0]; // (1 - i, 2)
//!     let incy  = 1;
//!
//!     let mut a = vec![0.0; 2 * m * n];
//!     let lda = m;
//!
//!     cgerc(m, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }
//! ```

use crate::level1::caxpy::caxpy;

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok_cplx; 
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn cgerc( 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f32; 2], 
    x       : &[f32], 
    incx    : usize, 
    y       : &[f32], 
    incy    : usize, 
    matrix  : &mut [f32], 
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
                caxpy(
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
