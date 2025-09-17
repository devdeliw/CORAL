//! Performs a single precision triangular solve (TRSV) with an upper triangular matrix.
//!
//! This function implements the BLAS [`strsv`] routine for **upper triangular** matrices,
//! solving the system `op(A) * x = b` in place for `x`, where `op(A)` is either `A` or `A^T`.
//!
//! The [`strusv`] function is crate-visible and is implemented via 
//! [`crate::level2::strsv`] using block back/forward substitution kernels.
//!
//! # Arguments
//! - `n`          (usize)           : Order (dimension) of the square matrix `A`.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `A` or `A^T`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `matrix`     (&[f32])          : Input slice containing the upper triangular matrix `A` in
//!                                  | column-major layout.
//! - `lda`        (usize)           : Leading dimension (stride between columns) of `A`.
//! - `x`          (&mut [f32])      : Input/output slice containing the right-hand side vector `x`,
//!                                  | which is overwritten with the solution.
//! - `incx`       (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place as the solution to `op(A) * x = b`.
//!
//! # Notes
//! - The implementation uses block decomposition with a block size of `NB = 8`.
//! - For the no-transpose case, diagonal blocks are solved using a **backward substitution** kernel,
//!   and previously solved elements are propagated with a fused [`saxpyf`] update.
//! - For the transpose case, diagonal blocks are solved using a **forward substitution** kernel,
//!   and remaining elements are updated via fused [`sdotf`] dot-product panels.
//! - The kernel is optimized for AArch64 NEON targets and assumes column-major memory layout.
//!
//! # Visibility
//! - pub(crate)
//!
//! # Author
//! Deval Deliwala

use core::slice; 
use crate::level2::enums::{CoralTranspose, CoralDiagonal}; 

// fused level1 
use crate::level1_special::{saxpyf::saxpyf, sdotf::sdotf}; 

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix; 

// TUNED
const NB: usize = 8; 

/// Solves a small `nb x nb` upper triangular diagonal block using
/// **backward substitution**; for no transpose only 
///
/// Used as the core kernel for the `NoTranspose` path.
///
/// # Arguments
/// - `nb`          (usize)      : Size of the block to solve.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the block `A[i.., i..]`.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*mut f32)   : Pointer to the subvector `x[i..]` to solve in place.
/// - `incx`        (usize)      : Stride between consecutive elements of `x_block`.
#[inline(always)] 
fn backward_substitution( 
    nb          : usize, 
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *mut f32,
    incx        : usize,
) {
    if nb == 0 { return; } 

    // nb x nb 
    // contiguous backwards substitution 
    unsafe { 
        if incx == 1 {
            // fast path; contiguous x
            for i in (0..nb).rev() { 
                let mut sum = 0.0;  
                
                // start of row i 
                let row_ptr = mat_block.add(i); 
                
                // columns right of diagonal 
                for k in (i + 1)..nb { 
                    // pointer to A[i, k] 
                    let matrix_ik = *row_ptr.add(k * lda);

                    // already solved 
                    let x_k = *x_block.add(k); 

                    sum += matrix_ik * x_k; 
                }

                let mut xi = *x_block.add(i) - sum; 

                if !unit_diag { 
                    // divide by diagonal elements 
                    let matrix_ii = *mat_block.add(i + i * lda); 
                    xi /= matrix_ii 
                } 

                *x_block.add(i) = xi; 
            }
        } else {
            // generic path; strided x
            for i in (0..nb).rev() { 
                let mut sum = 0.0;  
                
                // start of row i 
                let row_ptr = mat_block.add(i); 
                
                // columns right of diagonal 
                for k in (i + 1)..nb { 
                    // pointer to A[i, k] 
                    let matrix_ik = *row_ptr.add(k * lda);

                    // already solved 
                    let x_k = *x_block.add(k * incx); 

                    sum += matrix_ik * x_k; 
                }

                let mut xi = *x_block.add(i * incx) - sum; 

                if !unit_diag { 
                    // divide by diagonal element 
                    let matrix_ii = *mat_block.add(i + i * lda); 
                    xi /= matrix_ii 
                } 

                *x_block.add(i * incx) = xi; 
            }
        }
    }
}  

/// Solves a small `nb x nb` upper triangular diagonal block using 
/// **forward substitution**; for transpose only 
///
/// Used as the core kernel for the `Transpose` path.
///
/// # Arguments
/// - `nb`          (usize)      : Size of the block to solve.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the block `A[i.., i..]`.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*mut f32)   : Pointer to the subvector `x[i..]` to solve in place.
/// - `incx`        (usize)      : Stride between consecutive elements of `x_block`.
#[inline(always)] 
fn forward_substitution( 
    nb          : usize, 
    unit_diag   : bool,
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *mut f32, 
    incx        : usize,
) { 
    if nb == 0 { return; } 

    // nb x nb 
    // contiguous forward substitution 
    unsafe { 
        if incx == 1 {
            // fast path; contiguous x
            for i in 0..nb { 
                let mut sum = 0.0; 
                for k in 0..i { 
                    sum += *mat_block.add(k + i * lda) * *x_block.add(k); 
                }

                let mut xi = *x_block.add(i) - sum; 
                if !unit_diag { 
                    //divide by diagonal elements 
                    xi /= *mat_block.add(i + i * lda) 
                }

                *x_block.add(i) = xi; 
            }
        } else {
            // generic path; strided x
            let step = incx;
            for i in 0..nb { 
                let mut sum = 0.0; 
                for k in 0..i { 
                    sum += *mat_block.add(k + i * lda) * *x_block.add(k * step); 
                }

                let mut xi = *x_block.add(i * step) - sum; 
                if !unit_diag { 
                    // divide by diagonal element
                    xi /= *mat_block.add(i + i * lda) 
                }

                *x_block.add(i * step) = xi; 
            }
        }
    }
} 

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` below it; for transpose only 
///
/// Implements `x_tail := x_tail - A_view^T * x_block` using a fused dot kernel.
#[inline(always)] 
fn update_tail_transpose( 
    rows_below  : usize, 
    nb    	    : usize, 
    base        : *const f32, 
    lda         : usize, 
    x_block     : *const f32, 
    x_tail      : *mut f32, 
) { 
    if rows_below == 0 || nb == 0 { return; } 

    unsafe { 
        let mat_view_len = (rows_below - 1) * lda + nb; 
        let mat_view     = slice::from_raw_parts(base, mat_view_len); 

        // x_tail := x_tail - A_view^T * x_block 
        // implemented via fused sdotf given negative x_block 
        let mut x_block_neg = [0.0; NB]; 
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), nb);
        for k in 0..nb { x_block_neg[k] = -x_block_neg[k]; }

        let x_tail_slice = slice::from_raw_parts_mut(x_tail, rows_below); 

        // fused dot 
        sdotf(nb, rows_below, mat_view, lda, &x_block_neg[..nb], 1, x_tail_slice);
    }
}

#[inline] 
fn strusv_notranspose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "incx and lda stride must be nonzero"); 
    debug_assert!(required_len_ok(x.len(), n, incx), "x not large enough for given n, incx");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not large enough for given `nxn` shape and lda leading dimension"
    ); 

    // fast path 
    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut diag_idx = if n >= nb { n - nb } else { usize::MAX }; 
            while diag_idx != usize::MAX { 
                // pointer to A[diag_idx, diag_idx] 
                let mat_block = matrix.as_ptr().add(diag_idx + diag_idx * lda); 
            
                // mutable pointer to x[diag_idx..] 
                let x_block = x.as_mut_ptr().add(diag_idx); 

                backward_substitution(nb, unit_diag, mat_block, lda, x_block, 1);

                if diag_idx > 0 { 
                    let mat_panel_ptr = matrix.as_ptr().add(diag_idx * lda); 
                    let mat_panel_len = (nb - 1) * lda + diag_idx; 
                    let mat_panel     = slice::from_raw_parts(mat_panel_ptr, mat_panel_len); 

                    // only first nb values used; faster memory alloc 
                    // LLVM vectorizes 
                    let mut x_block_neg = [0.0; NB];
                    for k in 0..nb { 
                       x_block_neg[k] = -(*x_block.add(k));  
                    }

                    let y_head = slice::from_raw_parts_mut(x.as_mut_ptr(), diag_idx); 
                    saxpyf(diag_idx, nb, &x_block_neg, 1, mat_panel, lda, y_head, 1);
                }

                if diag_idx >= nb { diag_idx -= nb } else { break; } 
            } 

            if nb_tail > 0 { 
                let mat_block0 = matrix.as_ptr(); 
                let x_block0   = x.as_mut_ptr(); 

                backward_substitution(nb_tail, unit_diag, mat_block0, lda, x_block0, 1);
            }
        }
    } else { 
        backward_substitution(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline] 
fn strusv_transpose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "incx and lda stride must be nonzero"); 
    debug_assert!(required_len_ok(x.len(), n, incx), "x not large enough for given n, incx");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix not large enough for given `nxn` shape and lda leading dimension"
    ); 

    // fast path 
    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut diag_idx = 0; 
            while diag_idx + nb <= n { 
                // pointer to A[diag_idx, diag_idx] 
                let mat_block = matrix.as_ptr().add(diag_idx + diag_idx * lda); 

                // pointer to x[idx..]
                let x_block = x.as_mut_ptr().add(diag_idx); 

                forward_substitution(nb, unit_diag, mat_block, lda, x_block, 1);

                // starting idx to next block 
                let next_idx = diag_idx + nb; 
                if next_idx < n { 
                    let rows_below = n - next_idx; 
                    let col_base  = matrix.as_ptr().add(diag_idx + next_idx * lda); 
                    let x_tail    = x.as_mut_ptr().add(next_idx); 

                    // solve remaining 
                    update_tail_transpose(rows_below, nb, col_base, lda, x_block, x_tail); 
                } 

                diag_idx += nb; 
            }

            if nb_tail > 0 { 
                let idx = n - nb_tail; 
                let mat_block = matrix.as_ptr().add(idx + idx * lda); 
                let x_block   = x.as_mut_ptr().add(idx); 

                forward_substitution(nb_tail, unit_diag, mat_block, lda, x_block, 1);
            }
        } 
    } else { 
        forward_substitution(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
} 


#[inline] 
pub(crate) fn strusv(
    n           : usize, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    let unit_diag = match diagonal { 
        CoralDiagonal::UnitDiagonal    => true, 
        CoralDiagonal::NonUnitDiagonal => false, 
    }; 

    match transpose { 
        CoralTranspose::NoTranspose        => strusv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => strusv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => strusv_transpose  (n, unit_diag, matrix, lda, x, incx),
    }
}
