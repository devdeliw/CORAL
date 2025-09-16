//! Performs a single precision triangular matrix–vector multiply (TRMV) with 
//! an upper triangular matrix.
//!
//! This function implements the BLAS [`strmv`] routine for **upper triangular** matrices,
//! computing the in-place product `x := op(A) * x`, where `op(A)` is either `A` or `A^T`
//!
//! [`strumv`] function is crate visible and is implemented via [`crate::level2::strmv`] routine. 
//!
//! # Arguments
//! - `n`          (usize)           : Order (dimension) of the square matrix `A`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `A` or `A^T`.
//! - `matrix`     (&[f32])          : Input slice containing the upper triangular matrix `A` in
//!                                  | column-major layout.
//! - `lda`        (usize)           : Leading dimension (stride between columns) of `A`.
//! - `x`          (&mut [f32])      : Input/output slice containing the vector `x`, updated in place.
//! - `incx`       (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place as `x := op(A) * x`.
//!
//! # Notes
//! - The implementation uses block decomposition with a block size of `NB = 64`.
//! - Fused level-1 routines ([`saxpyf`] and [`sdotf`]) are used for panel updates to improve 
//!   performance.
//! - The kernel is optimized for AArch64 NEON targets and assumes column-major memory layout.
//!
//! # Visibility 
//! - pub(crate)
//!
//! # Author
//! Deval Deliwala

use core::slice;
use crate::level2::enums::{CoralTranspose, CoralDiagonal};  

// fused level 1 
use crate::level1_special::{saxpyf::saxpyf, sdotf::sdotf};

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix; 

// mini kernels 
use crate::level2::trmv_kernels::single_add_and_scale; 

const NB: usize = 64; 

/// Computes the product of an upper `buf_len x buf_len` diagonal block (no transpose)
/// with a contiguous `x_block`, writing the result to `y_block`.
///
/// This handles the top-left square block `A[idx..idx+buf_len, idx..idx+buf_len]`
/// during the `NoTranspose` traversal.
///
/// # Arguments
/// - `buf_len` (usize)        : Size of the block to process.
/// - `unit_diag` (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block` (*const f32) : Pointer to the first element of the block.
/// - `lda` (usize)            : Leading dimension of the full matrix.
/// - `x_block` (*const f32)   : Pointer to the input `x` subvector.
/// - `y_block` (*mut f32)     : Pointer to the output subvector (overwrites `x_block` contents).
#[inline(always)]
fn compute_upper_block_notranspose( 
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

            single_add_and_scale(buffer.as_mut_ptr(), column, k, scale); 
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

/// Computes the product of an upper `buf_len x buf_len` diagonal block (transpose)
/// with a contiguous `x_block`, writing the result to `y_block`.
///
/// This handles the top-left square block `A[idx..idx+buf_len, idx..idx+buf_len]`
/// during the `Transpose` traversal.
///
/// # Arguments
/// - `buf_len` (usize)        : Size of the block to process.
/// - `unit_diag` (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block` (*const f32) : Pointer to the first element of the block.
/// - `lda` (usize)            : Leading dimension of the full matrix.
/// - `x_block` (*const f32)   : Pointer to the input `x` subvector.
/// - `y_block` (*mut f32)     : Pointer to the output subvector (overwrites `x_block` contents).
#[inline(always)]
fn compute_upper_block_transpose( 
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
            // pointer to U[..buf_len, k] 
            let column = mat_block.add(k * lda); 

            // accumulates sum_i^k U_{i, k} x_i 
            let mut sum = 0.0;
            let mut i   = 0;
            while i + 4 <= k {
                sum += *column.add(i) * xbuffer[i];
                sum += *column.add(i + 1) * xbuffer[i + 1];
                sum += *column.add(i + 2) * xbuffer[i + 2];
                sum += *column.add(i + 3) * xbuffer[i + 3];
                i += 4;
            }
            while i < k {
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

/// Returns raw pointer to `A[i, j]` 
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

/// Scalar fallback kernel for a small upper triangular tail block (no transpose).
///
/// Used when fewer than `NB` rows remain, or for non-unit stride.
#[inline(always)]
fn compute_upper_block_tail_notranspose( 
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
                sum += *a_ij(mat_block, ii, jj, 1, lda) * *x0.add(jj * incx); 
            }

            *x0.add(ii * incx) = sum;
        }
    }
}

/// Scalar fallback kernel for a small upper triangular tail block (transpose).
///
/// Used when fewer than `NB` rows remain, or for non-unit stride.
#[inline(always)]
fn compute_upper_block_tail_transpose( 
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
                sum += *a_ij(mat_block, jj, ii, 1, lda) * *x0.add(jj * incx); 
            }

            *x0.add(ii * incx) = sum; 
        }
    }
}

/// [`CoralTranspose::NoTranspose`] variant 
#[inline]
fn strumv_notranspose( 
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
            let mut idx = 0; 
            while idx + nb <= n { 
                // pointer to A[idx, idx] 
                let mat_block = matrix.as_ptr().add(idx + idx * lda); 

                // pointer to idx element in x
                let x_block = x.as_ptr().add(idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(idx); 

                // full NB x NB block 
                compute_upper_block_notranspose(
                    nb, 
                    unit_diag,
                    mat_block, 
                    lda, 
                    x_block,
                    x_block_mut
                );

                let col_tail = idx + nb; 
                if col_tail < n { 
                    let cols_left = n - col_tail; 

                    // pointer to A[idx, tail] 
                    let mat_panel_ptr = matrix.as_ptr().add(idx + col_tail * lda); 

                    // matrix view from A[idx.., tail..] 
                    let mat_panel = slice::from_raw_parts(
                        mat_panel_ptr, 
                        (cols_left - 1) * lda + nb, 
                    );

                    let y_block = slice::from_raw_parts_mut(x_block_mut, nb); 

                    saxpyf(nb, cols_left, &x[col_tail..], 1, mat_panel, lda, y_block, 1);
                }

                idx += nb; 
            }
            if nb_tail > 0 { 
                let idx_left = n - nb_tail; 

                // pointer to A[idx_left, idx_left]
                let mat_block = matrix.as_ptr().add(idx_left + idx_left * lda); 

                let x_block = x.as_ptr().add(idx_left); 
                let y_block = x.as_mut_ptr().add(idx_left); 

                compute_upper_block_notranspose(
                    nb_tail, 
                    unit_diag, 
                    mat_block,
                    lda, 
                    x_block,
                    y_block
                );
            }
        }
    } else { 
        compute_upper_block_tail_notranspose(
            n,
            unit_diag, 
            matrix.as_ptr(), 
            lda,
            x.as_mut_ptr(),
            incx
        );
    }
}

/// [`CoralTranspose::Transpose`] variant
#[inline] 
fn strumv_transpose( 
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
            let mut idx = n; 
            while idx >= nb { 
                idx -= nb; 

                // pointer to A[idx, idx] 
                let mat_block   = matrix.as_ptr().add(idx + idx * lda); 

                // pointer to idx element in x
                let x_block     = x.as_ptr().add(idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(idx); 

                // full NB x NB block 
                compute_upper_block_transpose(
                    nb, 
                    unit_diag,
                    mat_block, 
                    lda, 
                    x_block,
                    x_block_mut
                );

                if idx > 0 { 
                    let cols_left   = nb; 
                    let rows_left   = idx; 

                    // pointer to A[0, idx] 
                    let mat_panel_ptr = matrix.as_ptr().add(idx * lda);

                    // matrix view from A[0.., idx..] 
                    let mat_panel     = slice::from_raw_parts(
                        mat_panel_ptr, 
                        (cols_left - 1) * lda + rows_left
                    ); 

                    let y_block = slice::from_raw_parts_mut(x_block_mut, nb); 

                    sdotf(rows_left, cols_left, mat_panel, lda, &x[..rows_left], 1, y_block);
                }
            }

            if nb_tail > 0 { 
                let blk_start = 0; 
                let blk_len   = nb_tail; 

                // pointer to A[blk_start, blk_start] 
                let mat_block = matrix.as_ptr().add(blk_start + blk_start * lda); 
                let x_block   = x.as_ptr().add(blk_start); 
                let y_block   = x.as_mut_ptr().add(blk_start); 

                compute_upper_block_transpose(
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
        compute_upper_block_tail_transpose( 
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
pub(crate) fn strumv( 
    n           : usize, 
    diagonal    : CoralDiagonal, 
    transpose   : CoralTranspose, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 

    let unit_diag = match diagonal { 
        CoralDiagonal::UnitDiagonal     => true, 
        CoralDiagonal::NonUnitDiagonal  => false, 
    };

    match transpose { 
        CoralTranspose::NoTranspose        => strumv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => strumv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => strumv_transpose  (n, unit_diag, matrix, lda, x, incx),
    }
}

