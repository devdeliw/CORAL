//! Performs a double precision complex triangular solve (TRSV) with an upper triangular matrix.
//!
//! This function implements the BLAS [`crate::level2::ztrsv`] routine for **upper triangular** matrices,
//! solving the system 
//!
//! ```text
//! op(A) * x = b in place for x, where op(A) is either A, A^T, or A^H.
//! ```
//!
//! The [`ztrusv`] function is crate-visible and is implemented via 
//! [`crate::level2::ztrsv`] using block back/forward substitution kernels.
//!
//! # Arguments
//! - `n`          (usize)           : Order of the square matrix `A`.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `A`, `A^T`, or `A^H`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `matrix`     (&[f64])          : Input slice containing the upper triangular matrix `A`
//! - `lda`        (usize)           : Leading dimension of `A`. 
//! - `x`          (&mut [f64])      : Input/output slice containing the right-hand side vector `x`
//!                                  | updated with solution. 
//! - `incx`       (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. `x` is updated in place with the solution. 
//!
//! # Notes
//! - The implementation uses block decomposition with a block size of `NB`.
//! - For the no-transpose case, diagonal blocks are solved using a **backward substitution** kernel,
//!   and previously solved elements are propagated with a fused [`zaxpyf`] update.
//! - For the transpose/conjugate-transpose case, diagonal blocks are solved using a **forward substitution** kernel,
//!   and remaining elements are updated via fused [`zdotuf`]/[`zdotcf`] dot-product panels.
//! - The kernel is optimized for AArch64 NEON targets 
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub(crate)
//!
//! # Author
//! Deval Deliwala

use core::slice; 
use crate::enums::{CoralTranspose, CoralDiagonal}; 

use crate::level1_special::{zaxpyf::zaxpyf, zdotcf::zdotcf, zdotuf::zdotuf}; 
use crate::level1::assert_length_helpers::required_len_ok_cplx; 
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx; 

const NB: usize = 8; 

/// Solves a small `nb x nb` upper triangular diagonal block using
/// **backward substitution**; for no transpose only 
///
/// Used as the core kernel for the `NoTranspose` path.
///
/// # Arguments
/// - `nb`          (usize)      : Size of the block to solve.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f64) : Pointer to the block `A[i.., i..]`.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*mut f64)   : Pointer to the subvector `x[i..]` to solve in place.
/// - `incx`        (usize)      : Stride between consecutive elements of `x_block`.
#[inline(always)] 
fn backward_substitution_z( 
    nb          : usize, 
    unit_diag   : bool, 
    mat_block   : *const f64, 
    lda         : usize, 
    x_block     : *mut f64,
    incx        : usize,
) {
    if nb == 0 { return; } 

    unsafe { 
        if incx == 1 {
            for i in (0..nb).rev() { 
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;

                let row_ptr = mat_block.add(2 * i); 
                
                for k in (i + 1)..nb { 
                    let a_ptr = row_ptr.add(2 * (k * lda));
                    let ar = *a_ptr;
                    let ai = *a_ptr.add(1);

                    let xk_ptr = x_block.add(2 * k);
                    let xr = *xk_ptr;
                    let xi = *xk_ptr.add(1);

                    sum_re += ar * xr - ai * xi; 
                    sum_im += ar * xi + ai * xr; 
                }

                let xi_ptr = x_block.add(2 * i);
                let mut xr = *xi_ptr - sum_re; 
                let mut xi = *xi_ptr.add(1) - sum_im; 

                if !unit_diag { 
                    let d_idx = 2 * (i + i * lda);
                    let dr = *mat_block.add(d_idx);
                    let di = *mat_block.add(d_idx + 1);
                    let den = dr * dr + di * di;
                    let nr =  xr * dr + xi * di;
                    let ni =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                } 

                *xi_ptr     = xr; 
                *xi_ptr.add(1) = xi; 
            }
        } else {
            for i in (0..nb).rev() { 
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;

                let row_ptr = mat_block.add(2 * i); 
                
                for k in (i + 1)..nb { 
                    let a_ptr = row_ptr.add(2 * (k * lda));
                    let ar = *a_ptr;
                    let ai = *a_ptr.add(1);

                    let xk_ptr = x_block.add(2 * (k * incx));
                    let xr = *xk_ptr;
                    let xi = *xk_ptr.add(1);

                    sum_re += ar * xr - ai * xi; 
                    sum_im += ar * xi + ai * xr; 
                }

                let xi_ptr = x_block.add(2 * (i * incx));
                let mut xr = *xi_ptr - sum_re; 
                let mut xi = *xi_ptr.add(1) - sum_im; 

                if !unit_diag { 
                    let d_idx = 2 * (i + i * lda);
                    let dr = *mat_block.add(d_idx);
                    let di = *mat_block.add(d_idx + 1);
                    let den = dr * dr + di * di;
                    let nr =  xr * dr + xi * di;
                    let ni =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                } 

                *xi_ptr     = xr; 
                *xi_ptr.add(1) = xi; 
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
/// - `conj`        (bool)       : Whether to conjugate `A`.
/// - `mat_block`   (*const f64) : Pointer to the block `A[i.., i..]`.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*mut f64)   : Pointer to the subvector `x[i..]` to solve in place.
/// - `incx`        (usize)      : Stride between consecutive elements of `x_block`.
#[inline(always)] 
fn forward_substitution_z( 
    nb          : usize, 
    unit_diag   : bool,
    conj        : bool,
    mat_block   : *const f64, 
    lda         : usize, 
    x_block     : *mut f64, 
    incx        : usize,
) { 
    if nb == 0 { return; } 

    unsafe { 
        if incx == 1 {
            for i in 0..nb { 
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;

                for k in 0..i { 
                    let a_idx = 2 * (k + i * lda);
                    let     ar = *mat_block.add(a_idx);
                    let mut ai = *mat_block.add(a_idx + 1);
                    if conj { ai = -ai; }

                    let xk_ptr = x_block.add(2 * k);
                    let xr = *xk_ptr;
                    let xi = *xk_ptr.add(1);

                    sum_re += ar * xr - ai * xi; 
                    sum_im += ar * xi + ai * xr; 
                }

                let xi_ptr = x_block.add(2 * i);
                let mut xr = *xi_ptr - sum_re; 
                let mut xi = *xi_ptr.add(1) - sum_im; 

                if !unit_diag { 
                    let d_idx = 2 * (i + i * lda);
                    let     dr = *mat_block.add(d_idx);
                    let mut di = *mat_block.add(d_idx + 1);
                    if conj { di = -di; }
                    let den = dr * dr + di * di;
                    let nr =  xr * dr + xi * di;
                    let ni =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *xi_ptr     = xr; 
                *xi_ptr.add(1) = xi; 
            }
        } else {
            let step = incx;
            for i in 0..nb { 
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;

                for k in 0..i { 
                    let a_idx = 2 * (k + i * lda);
                    let     ar = *mat_block.add(a_idx);
                    let mut ai = *mat_block.add(a_idx + 1);
                    if conj { ai = -ai; }

                    let xk_ptr = x_block.add(2 * (k * step));
                    let xr = *xk_ptr;
                    let xi = *xk_ptr.add(1);

                    sum_re += ar * xr - ai * xi; 
                    sum_im += ar * xi + ai * xr; 
                }

                let xi_ptr = x_block.add(2 * (i * step));
                let mut xr = *xi_ptr     - sum_re; 
                let mut xi = *xi_ptr.add(1) - sum_im; 

                if !unit_diag { 
                    let d_idx = 2 * (i + i * lda);
                    let     dr = *mat_block.add(d_idx);
                    let mut di = *mat_block.add(d_idx + 1);
                    if conj { di = -di; }
                    let den = dr * dr + di * di;
                    let nr =  xr * dr + xi * di;
                    let ni =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *xi_ptr     = xr; 
                *xi_ptr.add(1) = xi; 
            }
        }
    }
} 

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` below it; for transpose only 
///
/// Implements `x_tail := x_tail - A_view^T/H * x_block` using a fused dot kernel.
#[inline(always)] 
fn update_tail_transpose_z( 
    rows_below  : usize, 
    nb    	    : usize, 
    base        : *const f64, 
    lda         : usize, 
    x_block     : *const f64, 
    x_tail      : *mut f64, 
    conj        : bool,
) { 
    if rows_below == 0 || nb == 0 { return; } 

    unsafe { 
        let mat_view_len = 2 * ((rows_below - 1) * lda + nb); 
        let mat_view     = slice::from_raw_parts(base, mat_view_len); 

        let mut x_block_neg = [0.0f64; 2 * NB]; 
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), 2 * nb);
        for k in 0..(2 * nb) { x_block_neg[k] = -x_block_neg[k]; }

        let x_tail_slice = slice::from_raw_parts_mut(x_tail, 2 * rows_below); 

        if conj {
            zdotcf(nb, rows_below, mat_view, lda, &x_block_neg[..(2 * nb)], 1, x_tail_slice);
        } else {
            zdotuf(nb, rows_below, mat_view, lda, &x_block_neg[..(2 * nb)], 1, x_tail_slice);
        }
    }
}

#[inline] 
fn ztrusv_notranspose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f64], 
    lda         : usize, 
    x           : &mut [f64], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "incx and lda stride must be nonzero"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x not large enough for given n, incx");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not large enough for given `nxn` shape and lda leading dimension"
    ); 

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut diag_idx = if n >= nb { n - nb } else { usize::MAX }; 
            while diag_idx != usize::MAX { 
                let mat_block = matrix.as_ptr().add(2 * (diag_idx + diag_idx * lda)); 
                let x_block   = x.as_mut_ptr().add(2 * diag_idx); 

                backward_substitution_z(nb, unit_diag, mat_block, lda, x_block, 1);

                if diag_idx > 0 { 
                    let mat_panel_ptr = matrix.as_ptr().add(2 * (diag_idx * lda)); 
                    let mat_panel_len = 2 * ((nb - 1) * lda + diag_idx); 
                    let mat_panel     = slice::from_raw_parts(mat_panel_ptr, mat_panel_len); 

                    let mut x_block_neg = [0.0f64; 2 * NB];
                    core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), 2 * nb);
                    for k in 0..(2 * nb) { x_block_neg[k] = -x_block_neg[k]; }

                    let y_head = slice::from_raw_parts_mut(x.as_mut_ptr(), 2 * diag_idx); 
                    zaxpyf(diag_idx, nb, &x_block_neg[..(2 * nb)], 1, mat_panel, lda, y_head, 1);
                }

                if diag_idx >= nb { diag_idx -= nb } else { break; } 
            } 

            if nb_tail > 0 { 
                let mat_block0 = matrix.as_ptr(); 
                let x_block0   = x.as_mut_ptr(); 

                backward_substitution_z(nb_tail, unit_diag, mat_block0, lda, x_block0, 1);
            }
        }
    } else { 
        backward_substitution_z(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline] 
fn ztrusv_transpose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f64], 
    lda         : usize, 
    x           : &mut [f64], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "incx and lda stride must be nonzero"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x not large enough for given n, incx");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not large enough for given `nxn` shape and lda leading dimension"
    ); 

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut diag_idx = 0; 
            while diag_idx + nb <= n { 
                let mat_block = matrix.as_ptr().add(2 * (diag_idx + diag_idx * lda)); 
                let x_block   = x.as_mut_ptr().add(2 * diag_idx); 

                forward_substitution_z(nb, unit_diag, false, mat_block, lda, x_block, 1);

                let next_idx = diag_idx + nb; 
                if next_idx < n { 
                    let rows_below = n - next_idx; 
                    let col_base   = matrix.as_ptr().add(2 * (diag_idx + next_idx * lda)); 
                    let x_tail     = x.as_mut_ptr().add(2 * next_idx); 

                    update_tail_transpose_z(rows_below, nb, col_base, lda, x_block, x_tail, false); 
                } 

                diag_idx += nb; 
            }

            if nb_tail > 0 { 
                let idx = n - nb_tail; 
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda)); 
                let x_block   = x.as_mut_ptr().add(2 * idx); 

                forward_substitution_z(nb_tail, unit_diag, false, mat_block, lda, x_block, 1);
            }
        } 
    } else { 
        forward_substitution_z(n, unit_diag, false, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
} 

#[inline] 
fn ztrusv_conjtranspose( 
    n           : usize, 
    unit_diag   : bool, 
    matrix      : &[f64], 
    lda         : usize, 
    x           : &mut [f64], 
    incx        : usize, 
) { 
    if n == 0 { return; } 

    debug_assert!(incx > 0 && lda > 0, "incx and lda stride must be nonzero"); 
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x not large enough for given n, incx");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix not large enough for given `nxn` shape and lda leading dimension"
    ); 

    if incx == 1 { 
        let nb = NB; 
        let nb_tail = n % nb; 

        unsafe { 
            let mut diag_idx = 0; 
            while diag_idx + nb <= n { 
                let mat_block = matrix.as_ptr().add(2 * (diag_idx + diag_idx * lda)); 
                let x_block   = x.as_mut_ptr().add(2 * diag_idx); 

                forward_substitution_z(nb, unit_diag, true, mat_block, lda, x_block, 1);

                let next_idx = diag_idx + nb; 
                if next_idx < n { 
                    let rows_below = n - next_idx; 
                    let col_base  = matrix.as_ptr().add(2 * (diag_idx + next_idx * lda)); 
                    let x_tail    = x.as_mut_ptr().add(2 * next_idx); 

                    update_tail_transpose_z(rows_below, nb, col_base, lda, x_block, x_tail, true); 
                } 

                diag_idx += nb; 
            }

            if nb_tail > 0 { 
                let idx = n - nb_tail; 
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda)); 
                let x_block   = x.as_mut_ptr().add(2 * idx); 

                forward_substitution_z(nb_tail, unit_diag, true, mat_block, lda, x_block, 1);
            }
        } 
    } else { 
        forward_substitution_z(n, unit_diag, true, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
} 

#[inline] 
pub(crate) fn ztrusv(
    n           : usize, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    matrix      : &[f64], 
    lda         : usize, 
    x           : &mut [f64], 
    incx        : usize, 
) { 
    let unit_diag = match diagonal { 
        CoralDiagonal::UnitDiagonal    => true, 
        CoralDiagonal::NonUnitDiagonal => false, 
    }; 

    match transpose { 
        CoralTranspose::NoTranspose        => ztrusv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => ztrusv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => ztrusv_conjtranspose(n, unit_diag, matrix, lda, x, incx),
    }
}

