//! Performs a single precision complex Hermitian matrix–vector multiply (HEMV) in the form:
//!
//! ```text
//!     y := alpha * A * x + beta * y
//! ```
//!
//! where `A` is an `n` x `n` **Hermitian** interleaved column-major matrix `[re, im, ...]`. 
//! Only the triangle indicated by `uplo` is referenced. `x` is a complex vector of length `n`, 
//! and `y` is a complex vector of length `n`.
//!
//! This function implements the BLAS [`crate::level2::chemv`] routine, optimized for
//! AArch64 NEON architectures with blocking and panel packing. For off-diagonal work
//! it fuses a column-wise [`caxpyf`] stream with column-dots via [`cdotcf`] so each `A` element
//! is read exactly once while producing both the `A x` and the conjugate-transposed
//! contributions implied by Hermitian structure.
//!
//! # Arguments
//! - `uplo`   (CoralTriangular) : Which triangle of `A` is stored.
//! - `n`      (usize)           : Dimension of the matrix `A`.
//! - `alpha`  ([f32; 2])        : Scalar multiplier applied to `A * x` (`[re, im]`).
//! - `matrix` (&[f32])          : Input slice containing the matrix `A`; interleaved complex.
//! - `lda`    (usize)           : Leading dimension of `A`. 
//! - `x`      (&[f32])          : Input complex vector of length `n`. 
//! - `incx`   (usize)           : Stride between consecutive complex elements of `x`.
//! - `beta`   ([f32; 2])        : Scalar multiplier applied to `y` prior to accumulation.
//! - `y`      (&mut [f32])      : Input/output complex vector of complex length `n`
//! - `incy`   (usize)           : Stride between consecutive complex elements of `y`.
//!
//! # Returns
//! - Nothing. The contents of `y` are updated in place. 
//!
//! # Notes
//! - If `n == 0`, the function returns immediately.
//! - If `alpha == (0,0) && beta == (1,0)`, the function returns immediately (no change).
//! - A **fast path** is taken when `lda == n`, using an in-place triangular microkernel
//!   that touches each stored `A` element once without packing.
//! - Otherwise, a **blocked algorithm** iterates over row panels of height `MC` and
//!   column panels of width `NC`. Off-diagonal panels are handled with a fused
//!   [`caxpyf`]/[`cdotcf`] kernel on packed rectangles that lie entirely within the stored
//!   triangle; diagonal blocks are handled by a triangular microkernel.
//!
//! # Author
//! Deval Deliwala

use core::slice;

use crate::enums::CoralTriangular;
use crate::level1::cscal::cscal;
use crate::level1_special::caxpyf::caxpyf;
use crate::level1_special::cdotcf::cdotcf;

// assert length helpers
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level2::assert_length_helpers::required_len_ok_matrix_cplx;

// contiguous packing helpers
use crate::level2::{
    vector_packing::{
        pack_c32,
        pack_and_scale_c32,
        write_back_c32,
    },
    panel_packing::pack_panel_c32,
};

const MC: usize = 128;
const NC: usize = 128;

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn chemv(
    uplo    : CoralTriangular,
    n       : usize,
    alpha   : [f32; 2],
    matrix  : &[f32],
    lda     : usize,
    x       : &[f32],
    incx    : usize,
    beta    : [f32; 2],
    y       : &mut [f32],
    incy    : usize,
) {
    // quick return
    if n == 0 { return; }
    if alpha == [0.0, 0.0] && beta == [1.0, 0.0] { return; }


    debug_assert!(incx > 0 && incy > 0, "vector increments must be nonzero");
    debug_assert!(lda >= n, "matrix leading dimension must be >= n");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy");
    debug_assert!(
        required_len_ok_matrix_cplx(matrix.len(), n, n, lda),
        "matrix too short for given n and lda"
    );

    // pack x into contiguous buff and scale by alpha 
    let mut xbuffer: Vec<f32> = Vec::new(); 
    pack_and_scale_c32(n, alpha, x, incx, &mut xbuffer); 

    // pack y into contiguous buffer iff incy != 1 
    let (mut ybuffer, mut packed_y): (Vec<f32>, bool) = (Vec::new(), false); 
    let y_slice: &mut [f32] = if incy == 1 { y } else { 
        packed_y = true; 
        pack_c32(n, y, incy, &mut ybuffer);
        ybuffer.as_mut_slice()
    }; 

    // y := beta * y 
    if beta == [0.0, 0.0] {
        y_slice.fill(0.0);
    } else if beta != [1.0, 0.0] {
        cscal(n, beta, y_slice, 1);
    }

    // fast path 
    if lda == n { 
        unsafe { 
            match uplo { 
                CoralTriangular::UpperTriangular => { 
                    // for each col j
                    // update y[0..j] via complex axpy on the stored upper part
                    // accumulate y[j] via a conjugated dot (A[0..j-1,j])^H * x[0..j-1]
                    // finally add the real diagonal contribution
                    for j in 0..n { 
                        let col_ptr = matrix.as_ptr().add(j * 2 * lda); 
                        let xj_re   = xbuffer[2 * j]; 
                        let xj_im   = xbuffer[2 * j + 1]; 
                        let mut acc_re = 0.0f32; 
                        let mut acc_im = 0.0f32; 

                        // strictly upper part col j 
                        // rows [0, j) 
                        let mut i = 0; 
                        while i < j { 
                            let a_re = *col_ptr.add(2 * i); 
                            let a_im = *col_ptr.add(2 * i + 1); 

                            // a caxpy
                            let yptr = y_slice.as_mut_ptr().add(2 * i);
                            *yptr        += a_re * xj_re - a_im * xj_im;
                            *yptr.add(1) += a_re * xj_im + a_im * xj_re;

                            // acc += conj(a_ij) * x_i
                            let xi_re = xbuffer[2 * i];
                            let xi_im = xbuffer[2 * i + 1];
                            acc_re += a_re * xi_re + a_im * xi_im;
                            acc_im += a_re * xi_im - a_im * xi_re;

                            i += 1; 
                        }

                        // diagonal 
                        let a_jj_re = *col_ptr.add(2 * j); 
                        let yj_ptr  = y_slice.as_mut_ptr().add(2 * j);
                        *yj_ptr        += a_jj_re * xj_re + acc_re; 
                        *yj_ptr.add(1) += a_jj_re * xj_im + acc_im; 
                    }
                } 

                CoralTriangular::LowerTriangular => { 
                    // for each col j
                    // update y[(j+1)..n) via complex axpy on the stored lower part
                    // accumulate y[j] via a conjugated dot (A[(j+1)..,j])^H * x[(j+1)..]
                    // finally add the real diagonal contribution
                    for j in 0..n { 
                        let col_ptr = matrix.as_ptr().add(j * 2 * lda); 
                        let xj_re   = xbuffer[2 * j]; 
                        let xj_im   = xbuffer[2 * j + 1]; 
                        let mut acc_re = 0.0f32; 
                        let mut acc_im = 0.0f32; 

                        // strictly lower part 
                        // rows (j, n)
                        let mut i = j + 1; 
                        while i < n { 
                            let a_re = *col_ptr.add(2 * i); 
                            let a_im = *col_ptr.add(2 * i + 1); 

                            // a caxpy
                            let yptr = y_slice.as_mut_ptr().add(2 * i);
                            *yptr        += a_re * xj_re - a_im * xj_im;
                            *yptr.add(1) += a_re * xj_im + a_im * xj_re;

                            // acc += conj(a_ij) * x_i
                            let xi_re = xbuffer[2 * i];
                            let xi_im = xbuffer[2 * i + 1];
                            acc_re += a_re * xi_re + a_im * xi_im;
                            acc_im += a_re * xi_im - a_im * xi_re;

                            i += 1;
                        }

                        // diagonal 
                        let a_jj_re = *col_ptr.add(2 * j); 
                        let yj_ptr  = y_slice.as_mut_ptr().add(2 * j);
                        *yj_ptr        += a_jj_re * xj_re + acc_re; 
                        *yj_ptr.add(1) += a_jj_re * xj_im + acc_im; 
                    }
                }
            }
        }

        if packed_y { write_back_c32(n, &ybuffer, y, incy); }
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
                matrix.as_ptr().add(2 * row_idx), 
                (n - 1) * 2 * lda + 2 * (n - row_idx), 
            )
        }; 

        match uplo { 
            CoralTriangular::UpperTriangular => { 
                // for each col j inside block, read rows [0..j] relative to row_idx, 
                // fuse y_head[0..j] complex axpy with conjugated dot accumulation for y_head[j]. 
                { 
                    let x_head = &xbuffer[2 * row_idx .. 2 * (row_idx + mb_eff)];
                    let y_head: &mut [f32] = &mut y_slice[2 * row_idx .. 2 * (row_idx + mb_eff)]; 
                    for j in 0..mb_eff { 
                        let col_abs = row_idx + j; 
                        let col_ptr = unsafe { a_row_base.as_ptr().add(col_abs * 2 * lda) }; 
                        let xj_re   = x_head[2 * j]; 
                        let xj_im   = x_head[2 * j + 1]; 
                        let mut acc_re = 0.0f32; 
                        let mut acc_im = 0.0f32; 

                        // rows [0, j] 
                        let mut i = 0; 
                        while i < j { 
                            let a_re  = unsafe { *col_ptr.add(2 * i) }; 
                            let a_im  = unsafe { *col_ptr.add(2 * i + 1) }; 

                            // a caxpy
                            let yi = 2 * i;
                            unsafe { 
                                let yptr = y_head.as_mut_ptr().add(yi);
                                *yptr        += a_re * xj_re - a_im * xj_im; 
                                *yptr.add(1) += a_re * xj_im + a_im * xj_re; 
                            }

                            // acc += conj(a_ij) * x_head[i]
                            let xi_re = x_head[2 * i];
                            let xi_im = x_head[2 * i + 1];
                            acc_re += a_re * xi_re + a_im * xi_im; 
                            acc_im += a_re * xi_im - a_im * xi_re; 

                            i += 1; 
                        }

                        // diagonal 
                        let a_jj_re = unsafe { *col_ptr.add(2 * j) }; 
                        let yj = 2 * j;
                        unsafe { 
                            let yptr = y_head.as_mut_ptr().add(yj);
                            *yptr        += a_jj_re * xj_re + acc_re; 
                            *yptr.add(1) += a_jj_re * xj_im + acc_im; 
                        }
                    }
                }
                let mut col_idx = row_idx + mb_eff;
                while col_idx < n {
                    let nb_eff = core::cmp::min(NC, n - col_idx);

                    // pack A[row_idx..row_idx+mb_eff, col_idx..col_idx+nb_eff] (complex interleaved; lda in complex units)
                    pack_panel_c32(
                        &mut apack,
                        a_row_base,
                        mb_eff,
                        col_idx,
                        nb_eff,
                        1,
                        lda
                    );

                    // disjoint borrows; [0..row_idx+mb_eff) | [row_idx+mb_eff..n)
                    let (y_pre, y_post)     = y_slice.split_at_mut(2 * (row_idx + mb_eff));
                    let y_head: &mut [f32]  = &mut y_pre[2 * row_idx .. 2 * (row_idx + mb_eff)]; 
                    let start_tail          = 2 * (col_idx - (row_idx + mb_eff));
                    let y_tail: &mut [f32]  = &mut y_post[start_tail .. start_tail + 2 * nb_eff];

                    // y_head += apack * x_tail
                    let x_tail = &xbuffer[2 * col_idx .. 2 * (col_idx + nb_eff)];
                    caxpyf(
                        mb_eff,
                        nb_eff,
                        x_tail,
                        1,
                        &apack,
                        mb_eff,
                        y_head,
                        1
                    );

                    // y_tail += apack^H * x_head
                    let x_head = &xbuffer[2 * row_idx .. 2 * (row_idx + mb_eff)];
                    cdotcf(
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

                    // pack A[row_idx..row_idx+mb_eff, col_idx..col_idx+nb_eff] (complex interleaved)
                    pack_panel_c32(
                        &mut apack,
                        a_row_base,
                        mb_eff,
                        col_idx,
                        nb_eff,
                        1,
                        lda
                    );

                    // disjoint borrows; [0..row_idx) | [row_idx..n)
                    let (y_left_region, y_head_region) = y_slice.split_at_mut(2 * row_idx);
                    let y_left: &mut [f32] = &mut y_left_region[2 * col_idx .. 2 * (col_idx + nb_eff)];
                    let y_head: &mut [f32] = &mut y_head_region[.. 2 * mb_eff];

                    // y_head += apack * x_left
                    let x_left = &xbuffer[2 * col_idx .. 2 * (col_idx + nb_eff)];
                    caxpyf(
                        mb_eff,
                        nb_eff,
                        x_left,
                        1,
                        &apack,
                        mb_eff,
                        y_head,
                        1
                    );

                    // y_left += apack^H * x_head
                    let x_head = &xbuffer[2 * row_idx .. 2 * (row_idx + mb_eff)];
                    cdotcf(
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
                    let x_head = &xbuffer[2 * row_idx .. 2 * (row_idx + mb_eff)];
                    let y_head: &mut [f32] = &mut y_slice[2 * row_idx .. 2 * (row_idx + mb_eff)]; 
                    for j in 0..mb_eff { 
                        let col_abs = row_idx + j; 
                        let col_ptr = unsafe { a_row_base.as_ptr().add(col_abs * 2 * lda) }; 
                        let xj_re   = x_head[2 * j]; 
                        let xj_im   = x_head[2 * j + 1]; 
                        let mut acc_re = 0.0f32; 
                        let mut acc_im = 0.0f32; 

                        // rows (j, mb_eff) 
                        let mut i = j + 1; 
                        while i < mb_eff { 
                            let a_re  = unsafe { *col_ptr.add(2 * i) };
                            let a_im  = unsafe { *col_ptr.add(2 * i + 1) };

                            // a caxpy
                            unsafe { 
                                let yptr = y_head.as_mut_ptr().add(2 * i);
                                *yptr        += a_re * xj_re - a_im * xj_im; 
                                *yptr.add(1) += a_re * xj_im + a_im * xj_re; 
                            }

                            // acc += conj(a_ij) * x_head[i]
                            let xi_re = x_head[2 * i];
                            let xi_im = x_head[2 * i + 1];
                            acc_re += a_re * xi_re + a_im * xi_im; 
                            acc_im += a_re * xi_im - a_im * xi_re; 
                            i += 1;
                        }

                        // diagonal 
                        let a_jj_re = unsafe { *col_ptr.add(2 * j) }; 
                        unsafe { 
                            let yptr = y_head.as_mut_ptr().add(2 * j);
                            *yptr        += a_jj_re * xj_re + acc_re; 
                            *yptr.add(1) += a_jj_re * xj_im + acc_im; 
                        }
                    }
                }
            }
        }

        row_idx += mb_eff; 
    }

    if packed_y { 
        write_back_c32(n, &ybuffer, y, incy); 
    }
}

