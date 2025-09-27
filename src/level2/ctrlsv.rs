//! Performs a single precision complex triangular solve (TRSV) with a lower triangular matrix.
//!
//! This function implements the BLAS [`crate::level2::ctrsv`] routine for **lower triangular** matrices,
//! solving the system 
//!
//! ```text
//! `op(L) * x = b` in place for `x`, where `op(L)` is `L`, `L^T`, or `L^H`
//! ```
//!
//! The [`ctrlsv`] function is crate-visible and is implemented via
//! [`crate::level2::ctrsv`] using block forward/back substitution kernels.
//!
//! # Arguments
//! - `n`          (usize)           : Order of the square matrix `L`.
//! - `transpose`  (CoralTranspose)  : Specifies whether to use `L`, `L^T`, or `L^H`.
//! - `diagonal`   (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `matrix`     (&[f32])          : Input slice containing the interleaved lower triangular matrix `L`.
//! - `lda`        (usize)           : Leading dimension of `L`.
//! - `x`          (&mut [f32])      : Input/output slice containing the right-hand side vector `x`.
//!                                  | updated in place.
//! - `incx`       (usize)           : Stride between consecutive complex elements of `x`.
//!
//! # Returns
//! - Nothing. The contents of `x` are updated in place.
//!
//! # Notes
//! - The implementation uses block decomposition. 
//! - For the no-transpose case, diagonal blocks are solved using a **forward substitution** kernel,
//!   and remaining elements are updated with a fused [`caxpyf`] panel update.
//! - For the transpose/conjugate-transpose case, diagonal blocks are solved using a **backward substitution** kernel,
//!   and previously solved elements are propagated with a fused [`cdotuf`]/[`cdotcf`] update.
//! - The kernel is optimized for AArch64 NEON targets 
//! - Assumes column-major memory layout.
//!
//! # Visibility
//! - pub(crate)
//!
//! # Author
//! Deval Deliwala

use core::slice;
use crate::level2::enums::{CoralTranspose, CoralDiagonal};

// fused level1 (complex)
use crate::level1_special::{caxpyf::caxpyf, cdotcf::cdotcf, cdotuf::cdotuf};

// assert length helpers (complex)
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;

const NB: usize = 8;

/// Solves a small `nb x nb` lower triangular diagonal block using
/// **forward substitution**; for no transpose only
///
/// Used as the core kernel for the `NoTranspose` path.
///
/// # Arguments
/// - `nb`          (usize)      : Size of the block to solve.
/// - `unit_diag`   (bool)       : Whether to assume implicit 1s on the diagonal.
/// - `mat_block`   (*const f32) : Pointer to the block `A[i.., i..]` (interleaved).
/// - `lda`         (usize)      : Leading dimension of the full matrix (in complex elements).
/// - `x_block`     (*mut f32)   : Pointer to the subvector `x[i..]` (interleaved) to solve in place.
/// - `incx`        (usize)      : Stride between consecutive complex elements of `x_block`.
#[inline(always)]
fn forward_substitution_c(
    nb:       usize,
    unit_diag: bool,
    mat_block: *const f32,
    lda:      usize,
    x_block:  *mut f32,
    incx:     usize,
) {
    if nb == 0 { return; }

    unsafe {
        if incx == 1 {
            for i in 0..nb {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for k in 0..i {
                    let a_idx = 2 * (i + k * lda);
                    let a_re  = *mat_block.add(a_idx);
                    let a_im  = *mat_block.add(a_idx + 1);

                    let x_idx = 2 * k;
                    let xr    = *x_block.add(x_idx);
                    let xi    = *x_block.add(x_idx + 1);

                    sum_re += a_re * xr - a_im * xi;
                    sum_im += a_re * xi + a_im * xr;
                }

                let xi_idx = 2 * i;
                let mut xr = *x_block.add(xi_idx)     - sum_re;
                let mut xi = *x_block.add(xi_idx + 1) - sum_im;

                if !unit_diag {
                    let d_idx = 2 * (i + i * lda);
                    let dr    = *mat_block.add(d_idx);
                    let di    = *mat_block.add(d_idx + 1);
                    let den   = dr * dr + di * di;
                    let nr    =  xr * dr + xi * di;
                    let ni    =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *x_block.add(xi_idx)     = xr;
                *x_block.add(xi_idx + 1) = xi;
            }
        } else {
            let step = incx;
            for i in 0..nb {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for k in 0..i {
                    let a_idx = 2 * (i + k * lda);
                    let a_re  = *mat_block.add(a_idx);
                    let a_im  = *mat_block.add(a_idx + 1);

                    let x_idx = 2 * (k * step);
                    let xr    = *x_block.add(x_idx);
                    let xi    = *x_block.add(x_idx + 1);

                    sum_re += a_re * xr - a_im * xi;
                    sum_im += a_re * xi + a_im * xr;
                }

                let xi_idx = 2 * (i * step);
                let mut xr = *x_block.add(xi_idx)     - sum_re;
                let mut xi = *x_block.add(xi_idx + 1) - sum_im;

                if !unit_diag {
                    let d_idx = 2 * (i + i * lda);
                    let dr    = *mat_block.add(d_idx);
                    let di    = *mat_block.add(d_idx + 1);
                    let den   = dr * dr + di * di;
                    let nr    =  xr * dr + xi * di;
                    let ni    =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *x_block.add(xi_idx)     = xr;
                *x_block.add(xi_idx + 1) = xi;
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
/// - `conj`        (bool)       : Whether to conjugate `L` (for conjugate-transpose).
/// - `mat_block`   (*const f32) : Pointer to the block `A[i.., i..]` (interleaved).
/// - `lda`         (usize)      : Leading dimension of the full matrix (in complex elements).
/// - `x_block`     (*mut f32)   : Pointer to the subvector `x[i..]` (interleaved) to solve in place.
/// - `incx`        (usize)      : Stride between consecutive complex elements of `x_block`.
#[inline(always)]
fn backward_substitution_c(
    nb:       usize,
    unit_diag: bool,
    conj:     bool,
    mat_block: *const f32,
    lda:      usize,
    x_block:  *mut f32,
    incx:     usize,
) {
    if nb == 0 { return; }

    unsafe {
        if incx == 1 {
            for i in (0..nb).rev() {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for k in (i + 1)..nb {
                    // (L^T)[i,k] = L[k,i] ; for L^H also conjugate
                    let a_idx = 2 * (k + i * lda);
                    let ar     = *mat_block.add(a_idx);
                    let mut ai = *mat_block.add(a_idx + 1);
                    if conj { ai = -ai; }

                    let x_idx = 2 * k;
                    let xr    = *x_block.add(x_idx);
                    let xi    = *x_block.add(x_idx + 1);

                    sum_re += ar * xr - ai * xi;
                    sum_im += ar * xi + ai * xr;
                }

                let xi_idx = 2 * i;
                let mut xr = *x_block.add(xi_idx)     - sum_re;
                let mut xi = *x_block.add(xi_idx + 1) - sum_im;

                if !unit_diag {
                    // divide by diagonal (conjugate if conj)
                    let d_idx = 2 * (i + i * lda);
                    let dr     = *mat_block.add(d_idx);
                    let mut di = *mat_block.add(d_idx + 1);
                    if conj { di = -di; }
                    let den = dr * dr + di * di;
                    let nr  =  xr * dr + xi * di;
                    let ni  =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *x_block.add(xi_idx)     = xr;
                *x_block.add(xi_idx + 1) = xi;
            }
        } else {
            let step = incx;
            for i in (0..nb).rev() {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for k in (i + 1)..nb {
                    let a_idx = 2 * (k + i * lda);
                    let ar     = *mat_block.add(a_idx);
                    let mut ai = *mat_block.add(a_idx + 1);
                    if conj { ai = -ai; }

                    let x_idx = 2 * (k * step);
                    let xr    = *x_block.add(x_idx);
                    let xi    = *x_block.add(x_idx + 1);

                    sum_re += ar * xr - ai * xi;
                    sum_im += ar * xi + ai * xr;
                }

                let xi_idx = 2 * (i * step);
                let mut xr = *x_block.add(xi_idx)     - sum_re;
                let mut xi = *x_block.add(xi_idx + 1) - sum_im;

                if !unit_diag {
                    let d_idx = 2 * (i + i * lda);
                    let dr     = *mat_block.add(d_idx);
                    let mut di = *mat_block.add(d_idx + 1);

                    if conj { di = -di; }

                    let den = dr * dr + di * di;
                    let nr  =  xr * dr + xi * di;
                    let ni  =  xi * dr - xr * di;
                    xr = nr / den;
                    xi = ni / den;
                }

                *x_block.add(xi_idx)     = xr;
                *x_block.add(xi_idx + 1) = xi;
            }
        }
    }
}

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` below it; no transpose only
///
/// Implements `x_tail := x_tail - A_view * x_block` using a fused axpy kernel.
#[inline(always)]
fn update_tail_notranspose_c(
    rows_below: usize,
    nb:         usize,
    base:       *const f32,
    lda:        usize,
    x_block:    *const f32,
    x_tail:     *mut f32,
) {
    if rows_below == 0 || nb == 0 { return; }

    unsafe {
        let mat_view_len = 2 * ((nb - 1) * lda + rows_below);
        let mat_view     = slice::from_raw_parts(base, mat_view_len);

        let mut x_block_neg = [0.0; 2 * NB];
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), 2 * nb);
        for k in 0..(2 * nb) { x_block_neg[k] = -x_block_neg[k]; }

        let x_tail_slice = slice::from_raw_parts_mut(x_tail, 2 * rows_below);

        caxpyf(rows_below, nb, &x_block_neg[..(2 * nb)], 1, mat_view, lda, x_tail_slice, 1);
    }
}

/// Applies the contribution of a solved diagonal block to the remaining
/// entries of `x` above it; transpose only
///
/// Implements `x_head := x_head - A_left^T/H * x_block` using a fused dot kernel.
#[inline(always)]
fn update_head_transpose_c(
    head_len: usize,
    nb:       usize,
    base:     *const f32,
    lda:      usize,
    x_block:  *const f32,
    x_head:   *mut f32,
    conj:     bool,
) {
    if head_len == 0 || nb == 0 { return; }

    unsafe {
        let mat_view_len = 2 * ((head_len - 1) * lda + nb);
        let mat_view     = slice::from_raw_parts(base, mat_view_len);

        let mut x_block_neg = [0.0; 2 * NB];
        core::ptr::copy_nonoverlapping(x_block, x_block_neg.as_mut_ptr(), 2 * nb);
        for k in 0..(2 * nb) { x_block_neg[k] = -x_block_neg[k]; }

        let x_head_slice = slice::from_raw_parts_mut(x_head, 2 * head_len);

        if conj {
            cdotcf(nb, head_len, mat_view, lda, &x_block_neg[..(2 * nb)], 1, x_head_slice);
        } else {
            cdotuf(nb, head_len, mat_view, lda, &x_block_neg[..(2 * nb)], 1, x_head_slice);
        }
    }
}

#[inline]
fn ctrlsv_notranspose(
    n:         usize,
    unit_diag: bool,
    matrix:    &[f32],
    lda:       usize,
    x:         &mut [f32],
    incx:      usize,
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

                forward_substitution_c(nb, unit_diag, mat_block, lda, x_block, 1);

                let next_idx = diag_idx + nb;
                if next_idx < n {
                    let rows_below = n - next_idx;

                    let below_base = matrix.as_ptr().add(2 * (next_idx + diag_idx * lda));
                    let x_tail     = x.as_mut_ptr().add(2 * next_idx);

                    update_tail_notranspose_c(rows_below, nb, below_base, lda, x_block, x_tail);
                }

                diag_idx += nb;
            }

            if nb_tail > 0 {
                let idx       = n - nb_tail;
                let mat_block = matrix.as_ptr().add(2 * (idx + idx * lda));
                let x_block   = x.as_mut_ptr().add(2 * idx);

                forward_substitution_c(nb_tail, unit_diag, mat_block, lda, x_block, 1);
            }
        }
    } else {
        forward_substitution_c(n, unit_diag, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline]
fn ctrlsv_transpose(
    n:         usize,
    unit_diag: bool,
    matrix:    &[f32],
    lda:       usize,
    x:         &mut [f32],
    incx:      usize,
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

                backward_substitution_c(nb, unit_diag, false, mat_block, lda, x_block, 1);

                if diag_idx > 0 {
                    let head_len = diag_idx;

                    let left_base = matrix.as_ptr().add(2 * diag_idx);
                    let x_head    = x.as_mut_ptr();

                    update_head_transpose_c(head_len, nb, left_base, lda, x_block, x_head, false);
                }

                if diag_idx >= nb { diag_idx -= nb } else { break; }
            }

            if nb_tail > 0 {
                let mat_block = matrix.as_ptr();
                let x_block   = x.as_mut_ptr();
                backward_substitution_c(nb_tail, unit_diag, false, mat_block, lda, x_block, 1);
            }
        }
    } else {
        backward_substitution_c(n, unit_diag, false, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline]
fn ctrlsv_conjtranspose(
    n:         usize,
    unit_diag: bool,
    matrix:    &[f32],
    lda:       usize,
    x:         &mut [f32],
    incx:      usize,
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

                backward_substitution_c(nb, unit_diag, true, mat_block, lda, x_block, 1);

                if diag_idx > 0 {
                    let head_len = diag_idx;

                    let left_base = matrix.as_ptr().add(2 * diag_idx);
                    let x_head    = x.as_mut_ptr();

                    update_head_transpose_c(head_len, nb, left_base, lda, x_block, x_head, true);
                }

                if diag_idx >= nb { diag_idx -= nb } else { break; }
            }

            if nb_tail > 0 {
                let mat_block = matrix.as_ptr();
                let x_block   = x.as_mut_ptr();
                backward_substitution_c(nb_tail, unit_diag, true, mat_block, lda, x_block, 1);
            }
        }
    } else {
        backward_substitution_c(n, unit_diag, true, matrix.as_ptr(), lda, x.as_mut_ptr(), incx);
    }
}

#[inline]
pub(crate) fn ctrlsv(
    n:         usize,
    transpose: CoralTranspose,
    diagonal:  CoralDiagonal,
    matrix:    &[f32],
    lda:       usize,
    x:         &mut [f32],
    incx:      usize,
) {
    let unit_diag = match diagonal {
        CoralDiagonal::UnitDiagonal    => true,
        CoralDiagonal::NonUnitDiagonal => false,
    };

    match transpose {
        CoralTranspose::NoTranspose        => ctrlsv_notranspose(n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::Transpose          => ctrlsv_transpose  (n, unit_diag, matrix, lda, x, incx),
        CoralTranspose::ConjugateTranspose => ctrlsv_conjtranspose(n, unit_diag, matrix, lda, x, incx),
    }
}

