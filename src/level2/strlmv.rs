use core::slice;
use crate::level2::enums::{CoralTranspose, CoralDiag};  

// fused level 1 
use crate::level1_special::{saxpyf::saxpyf, sdotf::sdotf};

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix; 

// mini kernels 
use crate::level2::trmv_kernels::single_add_and_scale; 

const NB: usize = 64; 

#[inline(always)]
fn compute_lower_block_notranspose( 
    buf_len     : usize,
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    y_block     : *mut f32, 
) { 
    let mut xbuffer: [f32; NB] = [0.0; NB];

    unsafe { 
        core::ptr::copy_nonoverlapping(
            x_block, 
            xbuffer.as_mut_ptr(), 
            buf_len
        ); 
    } 

    let mut buffer: [f32; NB]  = [0.0; NB]; 

    unsafe { 
        for k in 0..buf_len { 
            let scale  = xbuffer[k]; 
            let column = mat_block.add(k * lda); 

            // strict lower part
            // rows (k+1..buf_len)
            let below = buf_len.saturating_sub(k + 1);
            single_add_and_scale(
                buffer.as_mut_ptr().add(k + 1), 
                column.add(k + 1), 
                below, 
                scale
            ); 

            if unit_diag { 
                *buffer.get_unchecked_mut(k) += scale; 
            } else { 
                *buffer.get_unchecked_mut(k) += *column.add(k) * scale; 
            }
        }

        core::ptr::copy_nonoverlapping(
            buffer.as_ptr(),
            y_block,
            buf_len
        );
    }
}

#[inline(always)]
fn compute_lower_block_transpose( 
    buf_len     : usize,
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    y_block     : *mut f32, 
) { 
    let mut xbuffer: [f32; NB] = [0.0; NB];

    unsafe { 
        core::ptr::copy_nonoverlapping(
            x_block, 
            xbuffer.as_mut_ptr(), 
            buf_len
        ); 
    } 

    let mut buffer: [f32; NB]  = [0.0; NB]; 

    unsafe { 
        for k in 0..buf_len {
            // pointer to L[..buf_len, k] 
            let column = mat_block.add(k * lda); 

            // accumulates sum_k^buf_len L_{i, k} x_i 
            let mut sum = 0.0;
            let mut i   = k + 1;
            while i + 4 <= buf_len {
                sum += *column.add(i) * xbuffer[i];
                sum += *column.add(i + 1) * xbuffer[i + 1];
                sum += *column.add(i + 2) * xbuffer[i + 2];
                sum += *column.add(i + 3) * xbuffer[i + 3];
                i += 4;
            }
            while i < buf_len {
                sum += *column.add(i) * xbuffer[i];
                i += 1;
            }

            if unit_diag { 
                sum += xbuffer[k]; 
            } else { 
                sum += *column.add(k) * xbuffer[k];
            }

            *buffer.get_unchecked_mut(k) = sum;
        }

        core::ptr::copy_nonoverlapping(
            buffer.as_ptr(),
            y_block,
            buf_len
        );
    }
}

#[inline(always)]
fn a_ij(
    matrix  : *const f32, 
    i       : usize, 
    j       : usize, 
    inc_row : usize, 
    inc_col : usize, 
) -> *const f32 { 
    unsafe { 
        matrix.add(i * inc_row + j * inc_col) 
    }
}

#[inline(always)]
fn compute_lower_block_tail_notranspose( 
    n           : usize, 
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *mut f32, 
    incx        : usize, 
) { 
    if n == 0 { return; }

    unsafe { 
        let x0 = x_block; 

        for i in (0..n).rev() { 
            let ii = i; 
            let mut sum = if unit_diag { 
                *x0.add(ii * incx) 
            } else { 
                *a_ij(mat_block, ii, ii, 1, lda) * *x0.add(ii * incx)
            }; 

            for j in 0..i { 
                let jj = j; 
                sum += *a_ij(mat_block, ii, jj, 1, lda) * *x0.add(jj * incx); 
            }

            *x0.add(ii * incx) = sum;
        }
    }
}

#[inline(always)]
fn compute_lower_block_tail_transpose( 
    n           : usize, 
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *mut f32, 
    incx        : usize, 
) { 
    if n == 0 { return; } 
   
    unsafe { 
        let x0 = x_block; 

        for i in 0..n { 
            let ii = i; 
            let mut sum = if unit_diag { 
                *x0.add(ii * incx) 
            } else { 
                *a_ij(mat_block, ii, ii, 1, lda) * *x0.add(ii * incx) 
            }; 

            for j in (i + 1)..n { 
                let jj = j; 
                sum += *a_ij(mat_block, jj, ii, 1, lda) * *x0.add(jj * incx); 
            }

            *x0.add(ii * incx) = sum; 
        }
    }
}

#[inline]
fn strlmv_notranspose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "stride and leading dimension must be strictly positive"); 
    debug_assert!(required_len_ok(x.len(), n, incx), "x is not big enough for given n/incx"); 
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not big enough for given triangular nxn and leading dimension" 
    );

    // fast path 
    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut idx = n; 
            while idx >= nb { 
                idx -= nb; 

                // pointer to A[idx, idx] 
                let mat_block = matrix.as_ptr().add(idx + idx * lda); 

                // pointer to idx element in x
                let x_block = x.as_ptr().add(idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(idx); 

                // full NB x NB block 
                compute_lower_block_notranspose(
                    nb, 
                    unit_diag,
                    mat_block, 
                    lda, 
                    x_block,
                    x_block_mut
                );

                if idx > 0 { 
                    let cols_left = idx; 

                    // pointer to A[idx, 0] 
                    let mat_panel_ptr = matrix.as_ptr().add(idx); 

                    // matrix view from A[idx.., 0..] 
                    let mat_panel = slice::from_raw_parts(
                        mat_panel_ptr, 
                        (cols_left - 1) * lda + nb, 
                    );

                    let y_block = slice::from_raw_parts_mut(x_block_mut, nb); 

                    saxpyf(nb, cols_left, &x[..cols_left], 1, mat_panel, lda, y_block, 1);
                }
            }
            if nb_tail > 0 { 
                let blk_start = 0; 
                let blk_len   = nb_tail; 

                // pointer to A[blk_start, blk_start]
                let mat_block = matrix.as_ptr().add(blk_start + blk_start * lda); 

                let x_block = x.as_ptr().add(blk_start); 
                let y_block = x.as_mut_ptr().add(blk_start); 

                compute_lower_block_notranspose(
                    blk_len, 
                    unit_diag, 
                    mat_block,
                    lda, 
                    x_block,
                    y_block
                );
            }
        }
    } else { 
        compute_lower_block_tail_notranspose(
            n,
            unit_diag, 
            matrix.as_ptr(), 
            lda,
            x.as_mut_ptr(),
            incx
        );
    }
}

#[inline] 
fn strlmv_transpose( 
    n           : usize,
    unit_diag   : bool, 
    matrix      : &[f32],  
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "stride and leading dimension must be strictly positive"); 
    debug_assert!(required_len_ok(x.len(), n, incx), "x is not big enough for given n/incx"); 
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not big enough for given triangular nxn and leading dimension" 
    );

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut idx = 0; 
            while idx + nb <= n { 
                // pointer to A[idx, idx] 
                let mat_block   = matrix.as_ptr().add(idx + idx * lda); 

                // pointer to idx element in x
                let x_block     = x.as_ptr().add(idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(idx); 

                // full NB x NB block 
                compute_lower_block_transpose(
                    nb, 
                    unit_diag,
                    mat_block, 
                    lda, 
                    x_block,
                    x_block_mut
                );

                let row_tail = idx + nb; 
                if row_tail < n { 
                    let rows_left   = n - row_tail; 
                    let cols_left   = nb; 

                    // pointer to A[row_tail, idx] 
                    let mat_panel_ptr = matrix.as_ptr().add(row_tail + idx * lda);

                    // matrix view from A[row_tail.., idx..] 
                    let mat_panel     = slice::from_raw_parts(
                        mat_panel_ptr, 
                        (cols_left - 1) * lda + rows_left
                    ); 

                    let y_block = slice::from_raw_parts_mut(x_block_mut, nb); 

                    sdotf(rows_left, cols_left, mat_panel, lda, &x[row_tail..], 1, y_block);
                }

                idx += nb; 
            }

            if nb_tail > 0 { 
                let idx_left = n - nb_tail; 
                let blk_len  = nb_tail; 

                // pointer to A[idx_left, idx_left] 
                let mat_block = matrix.as_ptr().add(idx_left + idx_left * lda); 
                let x_block   = x.as_ptr().add(idx_left); 
                let y_block   = x.as_mut_ptr().add(idx_left); 

                compute_lower_block_transpose(
                    blk_len, 
                    unit_diag, 
                    mat_block, 
                    lda, 
                    x_block, 
                    y_block
                );
            }
        }
    } else { 
        compute_lower_block_tail_transpose( 
            n, 
            unit_diag, 
            matrix.as_ptr(), 
            lda, 
            x.as_mut_ptr(), 
            incx
        ); 
    }
}

#[inline]
#[cfg(target_arch = "aarch64")] 
pub(crate) fn strlmv( 
    n           : usize, 
    diagonal    : CoralDiag, 
    transpose   : CoralTranspose, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 

    let unit_diag = match diagonal { 
        CoralDiag::UnitDiag     => true, 
        CoralDiag::NonUnitDiag  => false, 
    };

    match transpose { 
        CoralTranspose::NoTranspose        => strlmv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => strlmv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => strlmv_transpose  (n, unit_diag, matrix, lda, x, incx),
    }
}

