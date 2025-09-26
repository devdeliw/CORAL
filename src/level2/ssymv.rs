//! Performs a single precision symmetric matrix–vector multiply (SYMV) in the form:
//!
//! ```text
//!     y := alpha * A * x + beta * y
//! ```
//!
//! where `A` is an `n` x `n` **symmetric** column-major matrix. Only the triangle
//! indicated by `uplo` is referenced. `x` is a vector of length `n`, and `y` is a
//! vector of length `n`.
//!
//! This function implements the BLAS [`crate::level2::ssymv`] routine, optimized for
//! AArch64 NEON architectures with blocking and panel packing. For off-diagonal work
//! it fuses a column-wise [`saxpyf`] stream with column-dots so each `A` element is read
//! exactly once while producing both the `A x` and `A^T x` contributions implied by
//! symmetry.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Dimension of the matrix `A`.
//! - `alpha`  (f32)             : Scalar multiplier applied to `A * x`.
//! - `matrix` (&[f32])          : Input slice containing the matrix `A`, stored in column-major
//!                              | Only the specified triangle referenced; other triangle ignored. 
//! - `lda`    (usize)           : Leading dimension of `A`. 
//! - `x`      (&[f32])          : Input vector of length `n`.
//! - `incx`   (usize)           : Stride between consecutive elements of `x`.
//! - `beta`   (f32)             : Scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f32])      : Input/output vector of length `n`.
//! - `incy`   (usize)           : Stride between consecutive elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place. 
//!
//! # Notes
//! - If `n == 0`,                      the function returns immediately.
//! - If `alpha == 0.0 && beta == 1.0`, the function returns immediately.
//! - A **fast path** is taken when `lda == n`, using an in-place triangular microkernel
//!   that touches each stored `A` element once without packing.
//! - Otherwise, a **blocked algorithm** iterates over row panels of height `MC` and
//!   column panels of width `NC`. Off-diagonal panels are handled with a fused
//!   [`saxpyf`]/[`sdotf`] kernel on packed rectangles that lie entirely within the stored
//!   triangle; diagonal blocks are handled by a triangular microkernel.
//!
//! # Author
//! Deval Deliwala

use core::slice;

use crate::level1::sscal::sscal;
use crate::level1_special::saxpyf::saxpyf;
use crate::level1_special::sdotf::sdotf;

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
    panel_packing::pack_panel_f32,
    enums::CoralTriangular,
};

const MC: usize = 128;
const NC: usize = 128;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn ssymv(
    uplo    : CoralTriangular,
    n       : usize,
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
    if n == 0 { return; }
    if alpha == 0.0 && beta == 1.0 { return; }

    debug_assert!(incx > 0 && incy > 0, "vector increments must be nonzero");
    debug_assert!(lda >= n, "matrix leading dimension must be >= n");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");
    debug_assert!(
        required_len_ok_matrix(matrix.len(), n, n, lda),
        "matrix too short for given n and lda"
    );

    // pack x into contiguous buff and scale by alpha 
    let mut xbuffer: Vec<f32> = Vec::new(); 
    pack_and_scale_f32(n, alpha, x, incx, &mut xbuffer); 

    // pack y into contiguous buffer iff incy != 1 
    let (mut ybuffer, mut packed_y): (Vec<f32>, bool) = (Vec::new(), false); 
    let y_slice: &mut [f32] = if incy == 1 { y } else { 
        packed_y = true; 
        pack_f32(n, y, incy, &mut ybuffer);
        ybuffer.as_mut_slice()
    }; 

    // y := beta * y 
    if beta == 0.0 {
        y_slice.fill(0.0);
    } else if beta != 1.0 {
        sscal(n, beta, y_slice, 1);
    }

    // fast path 
    if lda == n { 
        unsafe { 
            match uplo { 
                CoralTriangular::UpperTriangular => { 
                    // for each col j 
                    // update y[0..j] via a saxpy on upper elements
                    // accumulate dot for y[j] from same elements 
                    // finally add diagonal 
                    for j in 0..n { 
                        let col_ptr = matrix.as_ptr().add(j * lda); 
                        let xj      = xbuffer[j]; 
                        let mut acc = 0.0; 

                        // strictly upper part col j 
                        // rows [0, j) 
                        let mut i = 0; 
                        while i < j { 
                            let a_ij   = *col_ptr.add(i); 

                            // a saxpy 
                            y_slice[i] = a_ij.mul_add(xj, y_slice[i]); 

                            acc        = a_ij.mul_add(xbuffer[i], acc); 
                            i += 1; 
                        }

                        // diagonal 
                        let a_jj    = *col_ptr.add(j); 
                        y_slice[j]  = a_jj.mul_add(xj, y_slice[j]); 
                        y_slice[j] += acc; 
                    }
                } 

                CoralTriangular::LowerTriangular => { 
                    // for each col j 
                    // update y[(j+1)..n) via a saxpy on lower elements 
                    // accumulate dot for y[j] from same elements 
                    // finally add diagonal 
                    for j in 0..n { 
                        let col_ptr = matrix.as_ptr().add(j * lda); 
                        let xj      = xbuffer[j]; 
                        let mut acc = 0.0; 

                        // strictly lower part 
                        // rows (j, n)
                        let mut i = j + 1; 
                        while i < n { 
                            let a_ij   = *col_ptr.add(i); 

                            // a saxpy 
                            y_slice[i] = a_ij.mul_add(xj, y_slice[i]); 

                            acc        = a_ij.mul_add(xbuffer[i], acc); 
                            i += 1; 
                        }

                        // diagonal 
                        let a_jj    = *col_ptr.add(j); 
                        y_slice[j]  = a_jj.mul_add(xj, y_slice[j]); 
                        y_slice[j] += acc;
                    }
                }
            }
        }

        if packed_y { write_back_f32(n, &ybuffer, y, incy); }
        return; 
    } 

    // general case: blocked via panel packing 
    let mut apack: Vec<f32> = Vec::new(); 
    
    let mut row_idx = 0; 
    while row_idx < n { 
        let mb_eff = core::cmp::min(MC, n - row_idx); 

        // view starting from (row_idx, 0)
        let a_row_base = unsafe { 
            slice::from_raw_parts(
                matrix.as_ptr().add(row_idx), 
                (n - 1) * lda + (n - row_idx), 
            )
        }; 

        match uplo { 
            CoralTriangular::UpperTriangular => { 
                // for each col j inside block, read rows [0..j] relative to row_idx, 
                // fuse y_head[0..j] saxpy with doc accumulation for y_head[j]. 
                { 
                    let x_head = &xbuffer[row_idx..row_idx + mb_eff];
                    let y_head: &mut [f32] = &mut y_slice[row_idx..row_idx + mb_eff]; 
                    for j in 0..mb_eff { 
                        let col_abs = row_idx + j; 
                        let col_ptr = unsafe { a_row_base.as_ptr().add(col_abs * lda) }; 
                        let xj      = x_head[j]; 
                        let mut acc = 0.0; 

                        // rows [0, j] 
                        let mut i = 0; 
                        while i < j { 
                            let a_ij  = unsafe { *col_ptr.add(i) }; 
                            y_head[i] = a_ij.mul_add(xj, y_head[i]); 
                            acc       = a_ij.mul_add(x_head[i], acc); 

                            i += 1; 
                        }

                        // diagobal 
                        let a_jj = unsafe { *col_ptr.add(j) }; 
                        y_head[j] = a_jj.mul_add(xj, y_head[j]); 
                        y_head[j] += acc; 
                    }
                }
                let mut col_idx = row_idx + mb_eff;
                while col_idx < n {
                    let nb_eff = core::cmp::min(NC, n - col_idx);

                    // pack A[row_idx..row_idx+mb_eff, col_idx..col_idx+nb_eff]
                    pack_panel_f32(
                        &mut apack,
                        a_row_base,
                        mb_eff,
                        col_idx,
                        nb_eff,
                        1,
                        lda
                    );

                    // disjoint borrows; [0..row_idx+mb_eff) | [row_idx+mb_eff..n)
                    let (y_pre, y_post)     = y_slice.split_at_mut(row_idx + mb_eff);
                    let y_head: &mut [f32]  = &mut y_pre[row_idx..]; 
                    let start_tail          = col_idx - (row_idx + mb_eff);
                    let y_tail: &mut [f32]  = &mut y_post[start_tail .. start_tail + nb_eff];

                    // y_head += apack * x_tail
                    let x_tail = &xbuffer[col_idx .. col_idx + nb_eff];
                    saxpyf(
                        mb_eff,
                        nb_eff,
                        x_tail,
                        1,
                        &apack,
                        mb_eff,
                        y_head,
                        1
                    );

                    // y_tail += apack^T * x_head
                    let x_head = &xbuffer[row_idx .. row_idx + mb_eff];
                    sdotf(
                        mb_eff,
                        nb_eff,
                        &apack,
                        mb_eff,
                        x_head,
                        1,
                        y_tail
                    );

                    col_idx += nb_eff;
                }
            }

            CoralTriangular::LowerTriangular => { 
                // off diagonal strictly below
                let mut col_idx = 0;
                while col_idx < row_idx {
                    let nb_eff = core::cmp::min(NC, row_idx - col_idx);

                    // pack A[row_idx..row_idx+mb_eff, col_idx..col_idx+nb_eff]
                    pack_panel_f32(
                        &mut apack,
                        a_row_base,
                        mb_eff,
                        col_idx,
                        nb_eff,
                        1,
                        lda
                    );

                    // disjoint borrows; [0..row_idx) | [row_idx..n)
                    let (y_left_region, y_head_region) = y_slice.split_at_mut(row_idx);
                    let y_left: &mut [f32] = &mut y_left_region[col_idx .. col_idx + nb_eff];
                    let y_head: &mut [f32] = &mut y_head_region[..mb_eff];

                    // y_head += apack * x_left
                    let x_left = &xbuffer[col_idx .. col_idx + nb_eff];
                    saxpyf(
                        mb_eff,
                        nb_eff,
                        x_left,
                        1,
                        &apack,
                        mb_eff,
                        y_head,
                        1
                    );

                    // y_left += apack^T * x_head
                    let x_head = &xbuffer[row_idx .. row_idx + mb_eff];
                    sdotf(
                        mb_eff,
                        nb_eff,
                        &apack,
                        mb_eff,
                        x_head,
                        1,
                        y_left
                    );

                    col_idx += nb_eff;
                }

                // diagonal block; triangular microkernel 
                { 
                    let x_head = &xbuffer[row_idx..row_idx + mb_eff];
                    let y_head: &mut [f32] = &mut y_slice[row_idx..row_idx + mb_eff]; 
                    for j in 0..mb_eff { 
                        let col_abs = row_idx + j; 
                        let col_ptr = unsafe { a_row_base.as_ptr().add(col_abs * lda) }; 
                        let xj      = x_head[j]; 
                        let mut acc = 0.0; 

                        // rows (j, mb_eff) 
                        let mut i = j + 1; 
                        while i < mb_eff { 
                            let a_ij  = unsafe { *col_ptr.add(i) };

                            // a saxpy 
                            y_head[i] = a_ij.mul_add(xj, y_head[i]); 

                            acc = a_ij.mul_add(x_head[i], acc); 
                            i += 1;
                        }

                        // diagonal 
                        let a_jj   = unsafe { *col_ptr.add(j) }; 
                        y_head[j]  = a_jj.mul_add(xj, y_head[j]); 
                        y_head[j] += acc; 
                    }
                }
            }
        }

        row_idx += mb_eff; 
    }

    if packed_y { 
        write_back_f32(n, &ybuffer, y, incy); 
    }
}
