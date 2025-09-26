//! Performs a double precision triangular solve (TRSV) with a lower triangular matrix.
//!
//! This function implements the BLAS [`crate::level2::dtrsv`] routine for **lower triangular** matrices,
//! solving the system `op(L) * x = b` in place for `x`, where `op(L)` is either `L` or `L^T`.
//!
//! The [`dtrlsv`] function is crate-visible and is implemented via 
//! [`crate::level2::dtrsv`] using block forward/back substitution kernels.
//!
//! # Arguments
//! - `n`          (usize)           : Order (dimension) of the square matrix `L`.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `L` or `L^T`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `matrix`     (&[f64])          : Input slice containing the lower triangular matrix `L`.
//! - `lda`        (usize)           : Leading dimension of `L`.
//! - `x`          (&mut [f64])      : Input/output slice containing the right-hand side vector `x`
//!                                  | updated in place. 
//! - `incx`       (usize)           : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place. 
//!
//! # Notes
//! - The implementation uses block decomposition with a block size of `NB = 8`.
//! - For the no-transpose case, diagonal blocks are solved using a **forward substitution** kernel,
//!   and remaining elements are updated with a fused [`daxpyf`] panel update.
//! - For the transpose case, diagonal blocks are solved using a **backward substitution** kernel,
//!   and previously solved elements are propagated with a fused [`ddotf`] update.
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
use crate::level1_special::{daxpyf::daxpyf, ddotf::ddotf};

// assert length helpers
use crate::level1::assert_length_helpers::required_len_ok;
use crate::level2::assert_length_helpers::required_len_ok_matrix;

const NB: usize = 8;

/// Solves a small `nb x nb` lower triangular diagonal block using
/// **forward substitution**; for no transpose only 
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
fn forward_substitution(
    nb:       usize,
    unit_diag: bool,
    mat_block: *const f64,
    lda:      usize,
    x_block:  *mut f64,
    incx:     usize,
) {
    if nb == 0 { return; }

    // nb x nb
    // contiguous forward substitution
    unsafe {
        if incx == 1 {
            // fast path; contiguous x
            for i in 0..nb {
                let mut sum = 0.0;

                // start of row i
                let row_ptr = mat_block.add(i);

                // columns left of diagonal
                for k in 0..i {
                    // pointer to A[i, k]
                    let matrix_ik = *row_ptr.add(k * lda);

                    // already solved
                    let x_k = *x_block.add(k);

                    sum += matrix_ik * x_k;
                }

                let mut xi = *x_block.add(i) - sum;

                if !unit_diag {
                    // divide by diagonal element
                    let matrix_ii = *mat_block.add(i + i * lda);
                    xi /= matrix_ii;
                }

                *x_block.add(i) = xi;
            }
        } else {
            // generic path; strided x
            let step = incx;
            for i in 0..nb {
                let mut sum = 0.0;

                // start of row i
                let row_ptr = mat_block.add(i);

                // columns left of diagonal
                for k in 0..i {
                    // pointer to A[i, k]
                    let matrix_ik = *row_ptr.add(k * lda);

                    // already solved
                    let x_k = *x_block.add(k * step);

                    sum += matrix_ik * x_k;
                }

                let mut xi = *x_block.add(i * step) - sum;

                if !unit_diag {
                    // divide by diagonal element
                    let matrix_ii = *mat_block.add(i + i * lda);
                    xi /= matrix_ii;
                }

                *x_block.add(i * step) = xi;
            }
        }
    }
}

/// Solves a small `nb x nb` lower triangular diagonal block using
/// **backward substitution**; for transpose only 
///
/// Used as the core kernel for the `Transpose` path.
///
/// # Arguments
/// - `nb`          (usize)      : Size of the block to solve.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f64) : Pointer to the block `A[i.., i..]`.
/// - `lda`         (usize)      : Leading dimension of the full matrix.
/// - `x_block`     (*mut f64)   : Pointer to the subvector `x[i..]` to solve in place.
/// - `incx`        (usize)      : Stride between consecutive elements of `x_block`.
#[inline(always)]
fn backward_substitution(
    nb:       usize,
    unit_diag: bool,
    mat_block: *const f64,
    lda:      usize,
    x_block:  *mut f64,
    incx:     usize,
) {
    if nb == 0 { return; }

    // nb x nb
    // contiguous backward substitution 
    unsafe {
        if incx == 1 {
            // fast path; contiguous x
            for i in (0..nb).rev() {
                let mut sum = 0.0;

                // columns right of diagonal in L^T; 
                // below diagonal in L
                for k in (i + 1)..nb {
                    // pointer to (L^T)[i, k] = L[k, i]
                    let matrix_ik = *mat_block.add(k + i * lda);

                    // already solved
                    let x_k = *x_block.add(k);

                    sum += matrix_ik * x_k;
                }

                let mut xi = *x_block.add(i) - sum;

                if !unit_diag {
                    // divide by diagonal element (same in L and L^T)
                    let matrix_ii = *mat_block.add(i + i * lda);
                    xi /= matrix_ii;
                }

                *x_block.add(i) = xi;
            }
        } else {
            // generic path; strided x
            let step = incx;
            for i in (0..nb).rev() {
                let mut sum = 0.0;

                // columns right of diagonal in L^T  <=>  rows below diagonal in L
                for k in (i + 1)..nb {
                    // pointer to (L^T)[i, k] = L[k, i]
                    let matrix_ik = *mat_block.add(k + i * lda);

                    // already solved
                    let x_k = *x_block.add(k * step);

                    sum += matrix_ik * x_k;
                }

                let mut xi = *x_block.add(i * step) - sum;

                if !unit_diag {
                    // divide by diagonal element
                    let matrix_ii = *mat_block.add(i + i * lda);
                    xi /= matrix_ii;
                }

                *x_block.add(i * step) = xi;
            }
        }
    }
}

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` below it; no transpose only
///
/// Implements `x_tail := x_tail - A_view * x_block` using a fused axpy kernel.
#[inline(always)]
fn update_tail_notranspose(
    rows_below: usize,
    nb:         usize,
    base:       *const f64,
    lda:        usize,
    x_block:    *const f64,
    x_tail:     *mut f64,
) {
    if rows_below == 0 || nb == 0 { return; }

    unsafe {
        // view of the panel directly below the diagonal block:
        // shape = rows_below x nb (column-major)
        let mat_view_len = (nb - 1) * lda + rows_below;
        let mat_view     = slice::from_raw_parts(base, mat_view_len);

        // x_tail := x_tail - A_view * x_block
        // implemented via fused daxpyf given negative x_block
        let mut x_block_neg = [0.0f64; NB];
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), nb);
        for k in 0..nb { x_block_neg[k] = -x_block_neg[k]; }

        let x_tail_slice = slice::from_raw_parts_mut(x_tail, rows_below);

        // fused axpyf
        daxpyf(rows_below, nb, &x_block_neg, 1, mat_view, lda, x_tail_slice, 1);
    }
}

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` above it; transpose only 
///
/// Implements `x_head := x_head - A_view^T * x_block` using a fused dot kernel.
#[inline(always)]
fn update_head_transpose(
    head_len: usize,
    nb:       usize,
    base:     *const f64,
    lda:      usize,
    x_block:  *const f64,
    x_head:   *mut f64,
) {
    if head_len == 0 || nb == 0 { return; }

    unsafe {
        let mat_view_len = (head_len - 1) * lda + nb;
        let mat_view     = slice::from_raw_parts(base, mat_view_len);

        // x_head := x_head - A_left^T * x_block
        // implemented via fused ddotf given negative x_block
        let mut x_block_neg = [0.0f64; NB];
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), nb);
        for k in 0..nb { x_block_neg[k] = -x_block_neg[k]; }

        let x_head_slice = slice::from_raw_parts_mut(x_head, head_len);

        // fused dot
        ddotf(nb, head_len, mat_view, lda, &x_block_neg[..nb], 1, x_head_slice);
    }
}

#[inline]
fn dtrlsv_notranspose(
    n:         usize,
    unit_diag: bool,
    matrix:    &[f64],
    lda:       usize,
    x:         &mut [f64],
    incx:      usize,
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

                // pointer to x[diag_idx..]
                let x_block = x.as_mut_ptr().add(diag_idx);

                forward_substitution(nb, unit_diag, mat_block, lda, x_block, 1);

                // starting idx to next block
                let next_idx = diag_idx + nb;
                if next_idx < n {
                    let rows_below = n - next_idx;

                    // panel directly below current diagonal block; A[next_idx, diag_idx]
                    let below_base = matrix.as_ptr().add(next_idx + diag_idx * lda);
                    let x_tail     = x.as_mut_ptr().add(next_idx);

                    // update remaining
                    update_tail_notranspose(rows_below, nb, below_base, lda, x_block, x_tail);
                }

                diag_idx += nb;
            }

            if nb_tail > 0 {
                let idx       = n - nb_tail;
                let mat_block = matrix.as_ptr().add(idx + idx * lda);
                let x_block   = x.as_mut_ptr().add(idx);

                forward_substitution(nb_tail, unit_diag, mat_block, lda, x_block, 1);
            }
        }
    } else {
        // generic path; strided x
        forward_substitution(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline]
fn dtrlsv_transpose(
    n:         usize,
    unit_diag: bool,
    matrix:    &[f64],
    lda:       usize,
    x:         &mut [f64],
    incx:      usize,
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

                // pointer to x[diag_idx..]
                let x_block = x.as_mut_ptr().add(diag_idx);

                backward_substitution(nb, unit_diag, mat_block, lda, x_block, 1);

                if diag_idx > 0 {
                    let head_len = diag_idx;

                    // panel to the left of the current diagonal block; A[diag_idx, 0]
                    let left_base = matrix.as_ptr().add(diag_idx);
                    let x_head    = x.as_mut_ptr();

                    // update the head 
                    update_head_transpose(head_len, nb, left_base, lda, x_block, x_head);
                }

                if diag_idx >= nb { diag_idx -= nb } else { break; }
            }

            if nb_tail > 0 {
                let mat_block = matrix.as_ptr();       // A[0, 0]
                let x_block   = x.as_mut_ptr();        // x[0..]
                backward_substitution(nb_tail, unit_diag, mat_block, lda, x_block, 1);
            }
        }
    } else {
        // generic path; strided x
        backward_substitution(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline]
pub(crate) fn dtrlsv(
    n:         usize,
    transpose: CoralTranspose,
    diagonal:  CoralDiagonal,
    matrix:    &[f64],
    lda:       usize,
    x:         &mut [f64],
    incx:      usize,
) {
    let unit_diag = match diagonal {
        CoralDiagonal::UnitDiagonal    => true,
        CoralDiagonal::NonUnitDiagonal => false,
    };

    match transpose {
        CoralTranspose::NoTranspose        => dtrlsv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => dtrlsv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => dtrlsv_transpose  (n, unit_diag, matrix, lda, x, incx),
    }
}

