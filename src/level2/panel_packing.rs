//! Packs a rectangular panel of columns from a strided matrix into a contiguous
//! column-major buffer.
//!
//! - `panel`       : Destination buffer, resized to `n_rows * n_cols`.
//! - `matrix`      : Source matrix stored in strided format.
//! - `n_rows`      : Number of rows in the panel.
//! - `col_idx`     : Starting column index in the source matrix.
//! - `n_cols`      : Number of columns in the panel.
//! - `row_stride`  : Stride between consecutive elements within a column.
//! - `col_stride`  : Stride between consecutive columns.
#[inline(always)] 
fn pack_panel<T: Copy>( 
    panel       : &mut Vec<T>, 
    matrix      : &[T],
    n_rows      : usize, 
    col_idx     : usize, 
    n_cols      : usize, 
    row_stride  : usize, 
    col_stride  : usize, 
) { 
    unsafe { 
        // initialize empty buffer 
        let total = n_rows * n_cols; 
        panel.clear();
        panel.reserve_exact(total); 
        panel.set_len(total);
        
        for col in 0..n_cols { 
            // start pointer to top of col
            let col_base = matrix.as_ptr().add((col_idx + col) * col_stride);

            // slice of panel current col will fill 
            let dst_col = &mut panel[col * n_rows .. (col + 1) * n_rows]; 

            if row_stride == 1 { 
                // fast path 
                core::ptr::copy_nonoverlapping(
                    col_base, 
                    dst_col.as_mut_ptr(), 
                    n_rows,
                );
            } else { 
                // row stride 
                for row in 0..n_rows { 
                    *dst_col.get_unchecked_mut(row) = *col_base.add(row * row_stride); 
                }
            }
        }
    }
}

/// Packs a rectangular panel of columns from a strided interleaved-complex matrix into a
/// contiguous column-major buffer.
///
/// The source and destination are interleaved `[re, im, re, im, ...]`.
///
/// - `panel`       : Destination buffer; resized to `2 * n_rows * n_cols`.
/// - `matrix`      : Source complex matrix. 
/// - `n_rows`      : Number of *complex* rows in the panel.
/// - `col_idx`     : Starting column index in *complex* columns.
/// - `n_cols`      : Number of *complex* columns in the panel.
/// - `row_stride`  : Stride between consecutive *complex* elements within a column.
/// - `col_stride`  : Stride between consecutive *complex* columns.
#[inline(always)]
fn pack_panel_cplx<T: Copy>(
    panel      : &mut Vec<T>,
    matrix     : &[T],
    n_rows     : usize,
    col_idx    : usize,
    n_cols     : usize,
    row_stride : usize,   
    col_stride : usize, 
) {
    unsafe {
        let total_scalars = 2 * n_rows * n_cols;
        panel.clear();
        panel.reserve_exact(total_scalars);
        panel.set_len(total_scalars);

        let s_row = 2 * row_stride;
        let s_col = 2 * col_stride;

        for col in 0..n_cols {
            // start pointer to top of col 
            let col_base = matrix.as_ptr().add((col_idx + col) * s_col);

            // slice of panel current col will fill 
            let dst_col = &mut panel[(2 * col * n_rows)..(2 * (col + 1) * n_rows)];

            if row_stride == 1 {
                core::ptr::copy_nonoverlapping(col_base, dst_col.as_mut_ptr(), 2 * n_rows);
            } else {
                let mut dptr = dst_col.as_mut_ptr();
                for r in 0..n_rows {
                    let sptr = col_base.add(r * s_row);
                    *dptr = *sptr;              
                    *dptr.add(1) = *sptr.add(1); 
                    dptr = dptr.add(2);
                }
            }
        }
    }
}


#[inline(always)]
pub(crate) fn pack_panel_f32(
    panel   : &mut Vec<f32>, 
    matrix  : &[f32], 
    n_rows  : usize, 
    col_idx : usize,
    n_cols  : usize, 
    row_stride: usize, 
    col_stride: usize,
) {
    pack_panel::<f32>(panel, matrix, n_rows, col_idx, n_cols, row_stride, col_stride);
}

#[inline(always)]
pub(crate) fn pack_panel_f64(
    panel   : &mut Vec<f64>, 
    matrix  : &[f64], 
    n_rows  : usize,
    col_idx : usize,
    n_cols  : usize, 
    row_stride: usize,
    col_stride: usize,
) {
    pack_panel::<f64>(panel, matrix, n_rows, col_idx, n_cols, row_stride, col_stride);
}

#[inline(always)]
pub(crate) fn pack_panel_c32(
    panel      : &mut Vec<f32>,
    matrix     : &[f32],
    n_rows     : usize,
    col_idx    : usize,
    n_cols     : usize,
    row_stride : usize,   
    col_stride : usize,   
) {
    pack_panel_cplx::<f32>(panel, matrix, n_rows, col_idx, n_cols, row_stride, col_stride);
}

#[inline(always)]
pub(crate) fn pack_panel_c64(
    panel      : &mut Vec<f64>,
    matrix     : &[f64],
    n_rows     : usize,
    col_idx    : usize,
    n_cols     : usize,
    row_stride : usize,
    col_stride : usize,
) {
    pack_panel_cplx::<f64>(panel, matrix, n_rows, col_idx, n_cols, row_stride, col_stride);
}


