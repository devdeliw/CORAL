//! Performs a single precision general matrix–vector multiply (GEMV) in the form:
//! 
//! ```text
//!     y := alpha * A * x + beta * y
//! ```
//!
//! where `A` is an `n_rows` × `n_cols` column-major matrix, `x` is a vector of
//! length `n_cols`, and `y` is a vector of length `n_rows`.  
//!
//! This function implements the BLAS [`crate::level2::sgemv`] routine for the
//! **no-transpose** case, optimized for AArch64 NEON architectures with blocking and 
//! panel packing.
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows (m) in the matrix `A`.
//! - `n_cols` (usize)      : Number of columns (n) in the matrix `A`.
//! - `alpha`  (f32)        : Scalar multiplier applied to the product `A * x`.
//! - `matrix` (&[f32])     : Input slice containing the matrix `A`, stored in column-major
//!                         | order with leading dimension `lda`.
//! - `lda`    (usize)      : Leading dimension of `A` (stride between successive columns).
//! - `x`      (&[f32])     : Input vector of length `n_cols`, with stride `incx`.
//! - `incx`   (usize)      : Stride between consecutive elements of `x`.
//! - `beta`   (f32)        : Scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f32]) : Input/output vector of length `n_rows`, with stride `incy`.
//! - `incy`   (usize)      : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place to contain the result
//!   `alpha * A * x + beta * y`.
//!
//! # Notes
//! - If `n_rows == 0` or `n_cols == 0`, the function returns immediately.
//! - If `alpha == 0.0 && beta == 1.0`, the function returns immediately (no change).
//! - When `lda == n_rows`, the matrix is stored contiguously, and a **fast path**
//!   is taken using a single fused [`saxpyf`] call.
//! - Otherwise, the routine falls back to a **blocked algorithm**, iterating over
//!   panels of size `MC x NC` with contiguous packing into temporary buffers.
//!
//! # Author
//! Deval Deliwala

use core::slice; 
use crate::level1::sscal::sscal;  
use crate::level1_special::saxpyf::saxpyf;

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix;  

// contiguous packing helpers 
use crate::level2::{
    vector_packing::{
        pack_f32, 
        pack_and_scale_f32, 
        write_back_f32, 
    },
    panel_packing::pack_panel, 
};

const MC: usize = 64; 
const NC: usize = 128;  

#[inline] 
#[cfg(target_arch = "aarch64")] 
pub(crate) fn sgemv_notranspose( 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : f32, 
    matrix  : &[f32], 
    lda     : usize, 
    x       : &[f32], 
    incx    : usize, 
    beta    : f32, 
    y       : &mut [f32], 
    incy    : usize,
) { 
    // quick return 
    if n_cols == 0 || n_rows == 0  { return; }
    if alpha == 0.0 && beta == 1.0 { return; }

    debug_assert!(incx > 0 && incy > 0, "vector increments must be nonzero"); 
    debug_assert!(lda >= n_rows, "matrix leading dimension must be >= n_rows"); 
    debug_assert!(required_len_ok(x.len(), n_cols, incx), "x too short for m/incx (NO TRANSPOSE)");
    debug_assert!(required_len_ok(y.len(), n_rows, incy), "y too short for n/incy (NO TRANSPOSE)");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n_rows, n_cols, lda),
        "matrix too short for given n_rows/n_cols and lda (NO TRANSPOSE)"
    ); 

    // y := beta * y 
    if beta != 1.0  { sscal(n_rows, beta, y, incy); }
    if alpha == 0.0 { return; }

    // pack x into contiguous buffer and scale by alpha 
    let mut xbuffer: Vec<f32> = Vec::new(); 
    pack_and_scale_f32(n_cols, alpha, x, incx, &mut xbuffer);

    // pack y into contiguous buffer if incy != 1 
    let (mut ybuffer, mut packed_y): (Vec<f32>, bool) = (Vec::new(), false); 
    let y_slice: &mut [f32] = if incy == 1 { y } else { 
        packed_y = true; 
        pack_f32(n_rows, y, incy, &mut ybuffer); 

        // slice lives as long as y_buffer 
        // entire scope 
        ybuffer.as_mut_slice() 
    };
        
    // fast path 
    // rows contiguous 
    if lda == n_rows { 
        unsafe { 
            let matrix_len  = n_cols * n_rows;  
            let matrix_view = slice::from_raw_parts(matrix.as_ptr(), matrix_len); 

            // y_slice[0..n_rows] += A xbuffer 
            saxpyf(n_rows, n_cols, &xbuffer, 1, matrix_view, lda, y_slice, 1); 
        }
    } else {
        // general case  
        //  
        //  for each row_panel of height MC 
        //      for each col_panel of width NC 
        //          update the MC slice of y using the x rows in this panel  
        let mut apack: Vec<f32> = Vec::new();

        let mut row_idx = 0;
        while row_idx < n_rows {
            // rows in this outer block
            let mb_eff = core::cmp::min(MC, n_rows - row_idx);

            // contiguous y sub-slice for this row block 
            let y_sub: &mut [f32] = &mut y_slice[row_idx..row_idx + mb_eff];

            // base slice starting at row_idx to end of matrix view
            let a_row_base = unsafe {
                slice::from_raw_parts(
                    matrix.as_ptr().add(row_idx),
                    (n_cols - 1) * lda + (n_rows - row_idx),
                )
            };

            let mut col_idx = 0;
            while col_idx < n_cols {
                // cols in this outer block
                let nb_eff = core::cmp::min(NC, n_cols - col_idx);

                // contiguous pack A[row_idx..row_idx+mb_eff, col_idx..col_idx+nb_eff]
                pack_panel(
                    &mut apack,
                    a_row_base,
                    mb_eff,
                    col_idx,
                    nb_eff,
                    1,      
                    lda,
                );

                // one fused saxpy call per block 
                // saxpyf implements its own MR/NR microblocking 
                // y_sub[0..mb_eff] += apack(mb_eff x nb_eff) x[col_idx..col_idx+nb_eff]
                saxpyf(
                    mb_eff,                                   
                    nb_eff,                                   
                    &xbuffer[col_idx..col_idx + nb_eff],    
                    1,
                    &apack,                                   
                    mb_eff,                                   
                    y_sub,                                    
                    1,
                );

                col_idx += nb_eff;
            }

            row_idx += mb_eff;
        }
    }

    if packed_y { 
        write_back_f32(n_rows, &ybuffer, y, incy);
    }
}
