//! Performs a single precision complex triangular matrix–vector multiply (CTRMV) with 
//! a lower triangular matrix.
//!
//! This function implements the BLAS [`crate::level2::ctrmv`] routine for **lower triangular** matrices,
//! computing the in-place product
//!
//! ```text
//! x := op(A) * x, where op(A) is either A, A^T, or A^H
//! ```
//!
//! Function is crate visible and is implemented via [`crate::level2::ctrmv`] routine. 
//!
//! # Arguments
//! - `n`          (usize)           : Order of the square matrix `A`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `A`, `A^T`, or `A^H`.
//! - `matrix`     (&[f32])          : Input slice of interleaved lower triangular complex matrix `A`
//! - `lda`        (usize)           : Leading dimension of `A` in complex elements.
//! - `x`          (&mut [f32])      : Input/output slice containing the complex vector `x`. 
//! - `incx`       (usize)           : Stride between consecutive complex elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place. 
//!
//! # Notes
//! - The implementation uses block decomposition.
//! - Fused level-1 routines ([`caxpyf`], [`cdotcf`], and [`cdotuf`]) are used for panel updates. 
//! - The kernel is optimized for AArch64 NEON targets. 
//! - Assumes column-major memory layout.
//!
//! # Visibility 
//! - pub(crate)
//!
//! # Author
//! Deval Deliwala

use core::slice;
use crate::enums::{CoralTranspose, CoralDiagonal};  

// fused level1
use crate::level1_special::{caxpyf::caxpyf, cdotcf::cdotcf, cdotuf::cdotuf};

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok_cplx; 
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx; 

// mini kernels 
use crate::level2::trmv_kernels::single_add_and_scale_c32;
use crate::level2::matrix_ij::a_ij_immutable_c32; 

const NB: usize = 64; 

/// Computes the product of a lower `buf_len x buf_len` diagonal block (no transpose)
/// with a contiguous `x_block`, writing the result to `y_block`.
///
/// This handles the bottom-right square block `A[idx..idx+buf_len, idx..idx+buf_len]`
/// during the `NoTranspose` traversal.
///
/// # Arguments
/// - `buf_len`     (usize)      : Size of the block to process.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the first element of the block.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*const f32) : Pointer to the input `x` subvector.
/// - `y_block`     (*mut f32)   : Pointer to the output subvector; overwrites `x_block`.
#[inline(always)]
fn compute_lower_block_notranspose( 
    buf_len     : usize,
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    y_block     : *mut f32, 
) { 
    let mut xbuffer: [f32; 2 * NB] = [0.0; 2 * NB];

    unsafe { 
        core::ptr::copy_nonoverlapping(
            x_block, 
            xbuffer.as_mut_ptr(), 
            2 * buf_len
        ); 
    } 

    let mut buffer: [f32; 2 * NB]  = [0.0; 2 * NB]; 

    unsafe { 
        for k in 0..buf_len { 
            let scale_re = *xbuffer.get_unchecked(2 * k);
            let scale_im = *xbuffer.get_unchecked(2 * k + 1);
            let column   = mat_block.add(2 * (k * lda)); 

            // strict lower part
            // rows (k+1..buf_len)
            let below = buf_len.saturating_sub(k + 1);
            single_add_and_scale_c32(
                buffer.as_mut_ptr().add(2 * (k + 1)),   
                column.add(2 * (k + 1)), 
                below, 
                [scale_re, scale_im]
            ); 

            if unit_diag { 
                *buffer.get_unchecked_mut(2 * k)     += scale_re; 
                *buffer.get_unchecked_mut(2 * k + 1) += scale_im; 
            } else { 
                let d_re = *column.add(2 * k);
                let d_im = *column.add(2 * k + 1);
                *buffer.get_unchecked_mut(2 * k)     += d_re * scale_re - d_im * scale_im;
                *buffer.get_unchecked_mut(2 * k + 1) += d_re * scale_im + d_im * scale_re;
            }
        }

        core::ptr::copy_nonoverlapping(              
            buffer.as_ptr(),
            y_block,
            2 * buf_len
        );
    }
}



/// Computes the product of a lower `buf_len x buf_len` diagonal block (transpose)
/// with a contiguous `x_block`, writing the result to `y_block`.
///
/// This handles the bottom-right square block `A[idx..idx+buf_len, idx..idx+buf_len]`
/// during the `Transpose` traversal.
///
/// # Arguments
/// - `buf_len`     (usize)      : Size of the block to process.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the first element of the block.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*const f32) : Pointer to the input `x` subvector.
/// - `y_block`     (*mut f32)   : Pointer to the output subvector; overwrites `x_block`.
#[inline(always)]
fn compute_lower_block_transpose( 
    buf_len     : usize,
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    y_block     : *mut f32, 
) { 
    let mut xbuffer: [f32; 2 * NB] = [0.0; 2 * NB];

    unsafe { 
        core::ptr::copy_nonoverlapping(
            x_block, 
            xbuffer.as_mut_ptr(), 
            2 * buf_len
        ); 
    } 

    let mut buffer: [f32; 2 * NB]  = [0.0; 2 * NB]; 

    unsafe { 
        for k in 0..buf_len {
            // pointer to L[..buf_len, k] 
            let column = mat_block.add(2 * (k * lda)); 

            // accumulates sum_k^buf_len L_{i, k} x_i 
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let mut i = k + 1;
            while i + 4 <= buf_len {
                let ar0 = *column.add(2 * i);     
                let ai0 = *column.add(2 * i + 1);
                let xr0 = xbuffer[2 * i];         
                let xi0 = xbuffer[2 * i + 1];
                sum_re += ar0 * xr0 - ai0 * xi0;  
                sum_im += ar0 * xi0 + ai0 * xr0;

                let ar1 = *column.add(2 * (i + 1)); 
                let ai1 = *column.add(2 * (i + 1) + 1);
                let xr1 = xbuffer[2 * (i + 1)];     
                let xi1 = xbuffer[2 * (i + 1) + 1];
                sum_re += ar1 * xr1 - ai1 * xi1;    
                sum_im += ar1 * xi1 + ai1 * xr1;

                let ar2 = *column.add(2 * (i + 2)); 
                let ai2 = *column.add(2 * (i + 2) + 1);
                let xr2 = xbuffer[2 * (i + 2)];     
                let xi2 = xbuffer[2 * (i + 2) + 1];
                sum_re += ar2 * xr2 - ai2 * xi2;    
                sum_im += ar2 * xi2 + ai2 * xr2;

                let ar3 = *column.add(2 * (i + 3)); 
                let ai3 = *column.add(2 * (i + 3) + 1);
                let xr3 = xbuffer[2 * (i + 3)];     
                let xi3 = xbuffer[2 * (i + 3) + 1];
                sum_re += ar3 * xr3 - ai3 * xi3;    
                sum_im += ar3 * xi3 + ai3 * xr3;

                i += 4;
            }
            while i < buf_len {
                let ar = *column.add(2 * i); let ai = *column.add(2 * i + 1);
                let xr = xbuffer[2 * i];     let xi = xbuffer[2 * i + 1];
                sum_re += ar * xr - ai * xi; sum_im += ar * xi + ai * xr;
                i += 1;
            }

            if unit_diag { 
                let xr = xbuffer[2 * k]; 
                let xi = xbuffer[2 * k + 1]; 
                sum_re += xr; 
                sum_im += xi; 
            } else { 
                let d_re = *column.add(2 * k);
                let d_im = *column.add(2 * k + 1);
                let xr   = xbuffer[2 * k]; 
                let xi   = xbuffer[2 * k + 1];
                sum_re += d_re * xr - d_im * xi;
                sum_im += d_re * xi + d_im * xr;
            }

            *buffer.get_unchecked_mut(2 * k)     = sum_re;
            *buffer.get_unchecked_mut(2 * k + 1) = sum_im;
        }

        core::ptr::copy_nonoverlapping(
            buffer.as_ptr(),
            y_block,
            2 * buf_len
        );
    }
}

/// Computes the product of a lower `buf_len x buf_len` diagonal block (conjugate transpose)
/// with a contiguous `x_block`, writing the result to `y_block`.
///
/// This handles the bottom-right square block `A[idx..idx+buf_len, idx..idx+buf_len]`
/// during the `ConjugateTranspose` traversal.
///
/// # Arguments
/// - `buf_len`     (usize)      : Size of the block to process.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the first element of the block.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*const f32) : Pointer to the input `x` subvector.
/// - `y_block`     (*mut f32)   : Pointer to the output subvector; overwrites `x_block`.
#[inline(always)]
fn compute_lower_block_conjugatetranspose( 
    buf_len     : usize,
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    y_block     : *mut f32, 
) { 
    let mut xbuffer: [f32; 2 * NB] = [0.0; 2 * NB];

    unsafe { 
        core::ptr::copy_nonoverlapping(
            x_block, 
            xbuffer.as_mut_ptr(), 
            2 * buf_len
        ); 
    } 

    let mut buffer: [f32; 2 * NB]  = [0.0; 2 * NB]; 

    unsafe { 
        for k in 0..buf_len {
            // pointer to L[..buf_len, k] 
            let column = mat_block.add(2 * (k * lda)); 

            // accumulates sum_k^buf_len conj(L_{i, k}) x_i 
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let mut i = k + 1;
            while i + 4 <= buf_len {
                let ar0 = *column.add(2 * i);     
                let ai0 = -*column.add(2 * i + 1);
                let xr0 = xbuffer[2 * i];         
                let xi0 = xbuffer[2 * i + 1];
                sum_re += ar0 * xr0 - ai0 * xi0;  
                sum_im += ar0 * xi0 + ai0 * xr0;

                let ar1 = *column.add(2 * (i + 1)); 
                let ai1 = -*column.add(2 * (i + 1) + 1);
                let xr1 = xbuffer[2 * (i + 1)];     
                let xi1 = xbuffer[2 * (i + 1) + 1];
                sum_re += ar1 * xr1 - ai1 * xi1;    
                sum_im += ar1 * xi1 + ai1 * xr1;

                let ar2 = *column.add(2 * (i + 2)); 
                let ai2 = -*column.add(2 * (i + 2) + 1);
                let xr2 = xbuffer[2 * (i + 2)];     
                let xi2 = xbuffer[2 * (i + 2) + 1];
                sum_re += ar2 * xr2 - ai2 * xi2;    
                sum_im += ar2 * xi2 + ai2 * xr2;

                let ar3 = *column.add(2 * (i + 3)); 
                let ai3 = -*column.add(2 * (i + 3) + 1);
                let xr3 = xbuffer[2 * (i + 3)];     
                let xi3 = xbuffer[2 * (i + 3) + 1];
                sum_re += ar3 * xr3 - ai3 * xi3;    
                sum_im += ar3 * xi3 + ai3 * xr3;

                i += 4;
            }
            while i < buf_len {
                let ar  = *column.add(2 * i); 
                let ai  = -*column.add(2 * i + 1);
                let xr  = xbuffer[2 * i];     
                let xi  = xbuffer[2 * i + 1];
                sum_re += ar * xr - ai * xi; 
                sum_im += ar * xi + ai * xr;
                i += 1;
            }

            if unit_diag { 
                let xr  = xbuffer[2 * k]; 
                let xi  = xbuffer[2 * k + 1]; 
                sum_re += xr; 
                sum_im += xi; 
            } else { 
                let d_re = *column.add(2 * k);
                let d_im = -*column.add(2 * k + 1);
                let xr   = xbuffer[2 * k]; 
                let xi   = xbuffer[2 * k + 1];
                sum_re  += d_re * xr - d_im * xi;
                sum_im  += d_re * xi + d_im * xr;
            }

            *buffer.get_unchecked_mut(2 * k)     = sum_re;
            *buffer.get_unchecked_mut(2 * k + 1) = sum_im;
        }

        core::ptr::copy_nonoverlapping(
            buffer.as_ptr(),
            y_block,
            2 * buf_len
        );
    }
}

/// Scalar fallback kernel for a small lower triangular tail block (no transpose).
///
/// Used when fewer than `NB` rows remain, or for non-unit stride.
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
            let (mut sum_re, mut sum_im) = if unit_diag { 
                (*x0.add(2 * ii * incx), *x0.add(2 * ii * incx + 1))
            } else { 
                let a  = a_ij_immutable_c32(mat_block, ii, ii, 1, lda);
                let ar = *a; let ai = *a.add(1);
                let xr = *x0.add(2 * ii * incx); 
                let xi = *x0.add(2 * ii * incx + 1);
                (ar * xr - ai * xi, ar * xi + ai * xr)
            }; 

            for j in 0..i { 
                let jj  = j; 
                let a   = a_ij_immutable_c32(mat_block, ii, jj, 1, lda);
                let ar  = *a; let ai = *a.add(1);
                let xr  = *x0.add(2 * jj * incx); 
                let xi  = *x0.add(2 * jj * incx + 1);
                sum_re += ar * xr - ai * xi; 
                sum_im += ar * xi + ai * xr; 
            }

            *x0.add(2 * ii * incx)     = sum_re;
            *x0.add(2 * ii * incx + 1) = sum_im;
        }
    }
}

/// Scalar fallback kernel for a small lower triangular tail block (transpose).
///
/// Used when fewer than `NB` rows remain, or for non-unit stride.
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
            let (mut sum_re, mut sum_im) = if unit_diag { 
                (*x0.add(2 * ii * incx), *x0.add(2 * ii * incx + 1))
            } else { 
                let a  = a_ij_immutable_c32(mat_block, ii, ii, 1, lda);
                let ar = *a; let ai = *a.add(1);
                let xr = *x0.add(2 * ii * incx); 
                let xi = *x0.add(2 * ii * incx + 1);
                (ar * xr - ai * xi, ar * xi + ai * xr)
            }; 

            for j in (i + 1)..n { 
                let jj  = j; 
                let a   = a_ij_immutable_c32(mat_block, jj, ii, 1, lda);
                let ar  = *a; let ai = *a.add(1);
                let xr  = *x0.add(2 * jj * incx); 
                let xi  = *x0.add(2 * jj * incx + 1);
                sum_re += ar * xr - ai * xi; 
                sum_im += ar * xi + ai * xr; 
            }

            *x0.add(2 * ii * incx)     = sum_re; 
            *x0.add(2 * ii * incx + 1) = sum_im; 
        }
    }
}

/// Scalar fallback kernel for a small lower triangular tail block (conjugate transpose).
///
/// Used when fewer than `NB` rows remain, or for non-unit stride.
#[inline(always)]
fn compute_lower_block_tail_conjugatetranspose( 
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
            let (mut sum_re, mut sum_im) = if unit_diag { 
                (*x0.add(2 * ii * incx), *x0.add(2 * ii * incx + 1))
            } else { 
                let a  = a_ij_immutable_c32(mat_block, ii, ii, 1, lda);
                let ar = *a; let ai = -*a.add(1);
                let xr = *x0.add(2 * ii * incx); 
                let xi = *x0.add(2 * ii * incx + 1);
                (ar * xr - ai * xi, ar * xi + ai * xr)
            }; 

            for j in (i + 1)..n { 
                let jj  = j; 
                let a   = a_ij_immutable_c32(mat_block, jj, ii, 1, lda);
                let ar  = *a; let ai = -*a.add(1);
                let xr  = *x0.add(2 * jj * incx); 
                let xi  = *x0.add(2 * jj * incx + 1);
                sum_re += ar * xr - ai * xi; 
                sum_im += ar * xi + ai * xr; 
            }

            *x0.add(2 * ii * incx)     = sum_re; 
            *x0.add(2 * ii * incx + 1) = sum_im; 
        }
    }
}

#[inline]
fn ctrlmv_notranspose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "stride and leading dimension must be strictly positive"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x is not big enough for given n/incx"); 
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
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
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda)); 

                // pointer to idx element in x
                let x_block = x.as_ptr().add(2 * idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(2 * idx); 

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
                    let mat_panel_ptr = matrix.as_ptr().add(2 * idx); 

                    // matrix view from A[idx.., 0..] 
                    let mat_panel = slice::from_raw_parts(
                        mat_panel_ptr, 
                        2 * ( (cols_left - 1) * lda + nb ),
                    );

                    let y_block = slice::from_raw_parts_mut(x_block_mut, 2 * nb); 

                    caxpyf(nb, cols_left, &x[..2 * cols_left], 1, mat_panel, lda, y_block, 1);
                }
            }
            if nb_tail > 0 { 
                let blk_start = 0; 
                let blk_len   = nb_tail; 

                // pointer to A[blk_start, blk_start]
                let mat_block = matrix.as_ptr().add(2 * (blk_start + blk_start * lda)); 

                let x_block = x.as_ptr().add(2 * blk_start); 
                let y_block = x.as_mut_ptr().add(2 * blk_start); 

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
fn ctrlmv_transpose( 
    n           : usize,
    unit_diag   : bool, 
    matrix      : &[f32],  
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "stride and leading dimension must be strictly positive"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x is not big enough for given n/incx"); 
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not big enough for given triangular nxn and leading dimension" 
    );

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut idx = 0; 
            while idx + nb <= n { 
                // pointer to A[idx, idx] 
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda)); 

                // pointer to idx element in x
                let x_block = x.as_ptr().add(2 * idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(2 * idx); 

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
                    let rows_left = n - row_tail; 
                    let cols_left = nb; 

                    // pointer to A[row_tail, idx] 
                    let mat_panel_ptr = matrix.as_ptr().add(2 * (row_tail + idx * lda));

                    // matrix view from A[row_tail.., idx..] 
                    let mat_panel     = slice::from_raw_parts(
                        mat_panel_ptr, 
                        2 * ( (cols_left - 1) * lda + rows_left )
                    ); 

                    let y_block = slice::from_raw_parts_mut(x_block_mut, 2 * nb); 

                    cdotuf(rows_left, cols_left, mat_panel, lda, &x[2 * row_tail..], 1, y_block);
                }

                idx += nb; 
            }

            if nb_tail > 0 { 
                let idx_left = n - nb_tail; 
                let blk_len  = nb_tail; 

                // pointer to A[idx_left, idx_left] 
                let mat_block = matrix.as_ptr().add(2 * (idx_left + idx_left * lda)); 
                let x_block   = x.as_ptr().add(2 * idx_left); 
                let y_block   = x.as_mut_ptr().add(2 * idx_left); 

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
fn ctrlmv_conjugatetranspose( 
    n           : usize,
    unit_diag   : bool, 
    matrix      : &[f32],  
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "stride and leading dimension must be strictly positive"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x is not big enough for given n/incx"); 
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not big enough for given triangular nxn and leading dimension" 
    );

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut idx = 0; 
            while idx + nb <= n { 
                // pointer to A[idx, idx] 
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda)); 

                // pointer to idx element in x
                let x_block = x.as_ptr().add(2 * idx);

                // mutable pointer to idx element in x 
                let x_block_mut = x.as_mut_ptr().add(2 * idx); 

                // full NB x NB block 
                compute_lower_block_conjugatetranspose(
                    nb, 
                    unit_diag,
                    mat_block, 
                    lda, 
                    x_block,
                    x_block_mut
                );

                let row_tail = idx + nb; 
                if row_tail < n { 
                    let rows_left = n - row_tail; 
                    let cols_left = nb; 

                    // pointer to A[row_tail, idx] 
                    let mat_panel_ptr = matrix.as_ptr().add(2 * (row_tail + idx * lda));

                    // matrix view from A[row_tail.., idx..] 
                    let mat_panel     = slice::from_raw_parts(
                        mat_panel_ptr, 
                        2 * ( (cols_left - 1) * lda + rows_left )
                    ); 

                    let y_block = slice::from_raw_parts_mut(x_block_mut, 2 * nb); 

                    cdotcf(rows_left, cols_left, mat_panel, lda, &x[2 * row_tail..], 1, y_block);
                }

                idx += nb; 
            }

            if nb_tail > 0 { 
                let idx_left = n - nb_tail; 
                let blk_len  = nb_tail; 

                // pointer to A[idx_left, idx_left] 
                let mat_block = matrix.as_ptr().add(2 * (idx_left + idx_left * lda)); 
                let x_block   = x.as_ptr().add(2 * idx_left); 
                let y_block   = x.as_mut_ptr().add(2 * idx_left); 

                compute_lower_block_conjugatetranspose(
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
        compute_lower_block_tail_conjugatetranspose( 
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
pub(crate) fn ctrlmv( 
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
        CoralTranspose::NoTranspose        => ctrlmv_notranspose        (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => ctrlmv_transpose          (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => ctrlmv_conjugatetranspose (n, unit_diag, matrix, lda, x, incx),
    }
}

