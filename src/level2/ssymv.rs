use core::slice;
use core::arch::aarch64::{
    vdupq_n_f32, vld1q_f32, vst1q_f32, vmulq_f32, vmlaq_f32, vaddq_f32, vfmaq_f32, vaddvq_f32
};
use crate::level1::sscal::sscal;
use crate::level2::{
    enums::UpLo,
    buf_helpers::{
        copy_back_y_from_unit_f32, 
        pack_y_to_unit_f32, 
        pack_and_scale_x_f32
    },
    assert_length_helpers::{
        required_len_ok_vec,
        required_len_ok_mat,
    },
};

#[inline]
pub fn ssymv(
    uplo        : UpLo,
    n           : usize,
    alpha       : f32,
    a           : &[f32],
    inc_row_a   : isize,   // col-major = 1
    inc_col_a   : isize,   // col-major > 0
    x           : &[f32],
    incx        : isize,
    beta        : f32,
    y           : &mut [f32],
    incy        : isize,
) {
    if n == 0 { return; }
    if alpha == 0.0 && beta == 1.0 { return; }

    debug_assert!(incx  != 0 && incy != 0, "vector increments must be non-zero");
    debug_assert!(inc_row_a != 0 && inc_col_a != 0, "matrix strides must be non-zero");
    debug_assert!(required_len_ok_vec(x.len(), n, incx), "x too short for n/incx (SYMV)");
    debug_assert!(required_len_ok_vec(y.len(), n, incy), "y too short for n/incy (SYMV)");
    debug_assert!(
        required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a),
        "a too short for n x n/strides (SYMV)"
    );

    if beta != 1.0 {
        sscal(n, beta, y, incy);
    }
    if alpha == 0.0 { return; }

    let mut xbuf: Vec<f32> = Vec::new();
    unsafe { pack_and_scale_x_f32(n, alpha, x, incx, &mut xbuf); }

    let mut ybuf: Option<Vec<f32>> = None;
    let y_unit: *mut f32 = if incy == 1 {
        y.as_mut_ptr()
    } else {
        let mut tmp = Vec::<f32>::new();
        unsafe { pack_y_to_unit_f32(n, y, incy, &mut tmp); }
        let p = tmp.as_mut_ptr();
        ybuf = Some(tmp);
        p
    };

    let col_major_fast = inc_row_a == 1 && inc_col_a > 0;
    unsafe {
        match (uplo, col_major_fast) {
            (UpLo::UpperTriangular, true) => {
                let lda = inc_col_a as usize;
                let yv  = slice::from_raw_parts_mut(y_unit, n);

                for j in 0..n {
                    let xj   = *xbuf.get_unchecked(j);
                    let colp = a.as_ptr().add(j * lda);
                    let mut temp2 = 0.0f32;

                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);

                    let mut i = 0usize;
                    let vxj = vdupq_n_f32(xj);

                    while i + 16 <= j {
                        let a0 = vld1q_f32(colp.add(i + 0));
                        let a1 = vld1q_f32(colp.add(i + 4));
                        let a2 = vld1q_f32(colp.add(i + 8));
                        let a3 = vld1q_f32(colp.add(i + 12));

                        // y += xj * A
                        let y0 = vld1q_f32(yv.as_ptr().add(i + 0));
                        let y1 = vld1q_f32(yv.as_ptr().add(i + 4));
                        let y2 = vld1q_f32(yv.as_ptr().add(i + 8));
                        let y3 = vld1q_f32(yv.as_ptr().add(i + 12));

                        let y0 = vmlaq_f32(y0, a0, vxj);
                        let y1 = vmlaq_f32(y1, a1, vxj);
                        let y2 = vmlaq_f32(y2, a2, vxj);
                        let y3 = vmlaq_f32(y3, a3, vxj);

                        vst1q_f32(yv.as_mut_ptr().add(i + 0),  y0);
                        vst1q_f32(yv.as_mut_ptr().add(i + 4),  y1);
                        vst1q_f32(yv.as_mut_ptr().add(i + 8),  y2);
                        vst1q_f32(yv.as_mut_ptr().add(i + 12), y3);

                        // temp2 += dot(A, xbuf)
                        let x0 = vld1q_f32(xbuf.as_ptr().add(i + 0));
                        let x1 = vld1q_f32(xbuf.as_ptr().add(i + 4));
                        let x2 = vld1q_f32(xbuf.as_ptr().add(i + 8));
                        let x3 = vld1q_f32(xbuf.as_ptr().add(i + 12));

                        acc0    = vfmaq_f32(acc0, a0, x0);
                        acc1    = vfmaq_f32(acc1, a1, x1);
                        acc2    = vfmaq_f32(acc2, a2, x2);
                        acc3    = vfmaq_f32(acc3, a3, x3);

                        i += 16;
                    }

                    let acc01 = vaddq_f32(acc0, acc1);
                    let acc23 = vaddq_f32(acc2, acc3);
                    temp2 += vaddvq_f32(vaddq_f32(acc01, acc23));

                    while i + 4 <= j {
                        let a0 = vld1q_f32(colp.add(i));
                        let y0 = vld1q_f32(yv.as_ptr().add(i));
                        let y0 = vmlaq_f32(y0, a0, vxj);
                        vst1q_f32(yv.as_mut_ptr().add(i), y0);

                        let x0 = vld1q_f32(xbuf.as_ptr().add(i));
                        temp2 += vaddvq_f32(vmulq_f32(a0, x0));

                        i += 4;
                    }
                    while i < j {
                        let aij = *colp.add(i);
                        *yv.get_unchecked_mut(i) += xj * aij;
                        temp2 += aij * *xbuf.get_unchecked(i);

                        i += 1;
                    }

                    *yv.get_unchecked_mut(j) += *colp.add(j) * xj + temp2;
                }
            }

            (UpLo::LowerTriangular, true) => {
                let lda = inc_col_a as usize;
                let yv  = slice::from_raw_parts_mut(y_unit, n);

                for j in 0..n {
                    let xj   = *xbuf.get_unchecked(j);
                    let colp = a.as_ptr().add(j * lda);
                    let mut temp2 = 0.0f32;

                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);

                    *yv.get_unchecked_mut(j) += *colp.add(j) * xj;

                    let start = j + 1;
                    let len   = n.saturating_sub(start);

                    let mut i = 0usize;
                    let vxj = vdupq_n_f32(xj);

                    while i + 16 <= len {
                        let off = start + i;

                        let a0 = vld1q_f32(colp.add(off + 0));
                        let a1 = vld1q_f32(colp.add(off + 4));
                        let a2 = vld1q_f32(colp.add(off + 8));
                        let a3 = vld1q_f32(colp.add(off + 12));

                        let y0 = vld1q_f32(yv.as_ptr().add(off + 0));
                        let y1 = vld1q_f32(yv.as_ptr().add(off + 4));
                        let y2 = vld1q_f32(yv.as_ptr().add(off + 8));
                        let y3 = vld1q_f32(yv.as_ptr().add(off + 12));

                        let y0 = vmlaq_f32(y0, a0, vxj);
                        let y1 = vmlaq_f32(y1, a1, vxj);
                        let y2 = vmlaq_f32(y2, a2, vxj);
                        let y3 = vmlaq_f32(y3, a3, vxj);

                        vst1q_f32(yv.as_mut_ptr().add(off + 0),  y0);
                        vst1q_f32(yv.as_mut_ptr().add(off + 4),  y1);
                        vst1q_f32(yv.as_mut_ptr().add(off + 8),  y2);
                        vst1q_f32(yv.as_mut_ptr().add(off + 12), y3);

                        let x0 = vld1q_f32(xbuf.as_ptr().add(off + 0));
                        let x1 = vld1q_f32(xbuf.as_ptr().add(off + 4));
                        let x2 = vld1q_f32(xbuf.as_ptr().add(off + 8));
                        let x3 = vld1q_f32(xbuf.as_ptr().add(off + 12));

                        acc0    = vfmaq_f32(acc0, a0, x0);
                        acc1    = vfmaq_f32(acc1, a1, x1);
                        acc2    = vfmaq_f32(acc2, a2, x2);
                        acc3    = vfmaq_f32(acc3, a3, x3);

                        i += 16;
                    }

                    let acc01 = vaddq_f32(acc0, acc1);
                    let acc23 = vaddq_f32(acc2, acc3);
                    temp2 += vaddvq_f32(vaddq_f32(acc01, acc23));

                    while i + 4 <= len {
                        let off = start + i;

                        let a0 = vld1q_f32(colp.add(off));
                        let y0 = vld1q_f32(yv.as_ptr().add(off));
                        let y0 = vmlaq_f32(y0, a0, vxj);
                        vst1q_f32(yv.as_mut_ptr().add(off), y0);

                        let x0 = vld1q_f32(xbuf.as_ptr().add(off));
                        temp2 += vaddvq_f32(vmulq_f32(a0, x0));

                        i += 4;
                    }
                    while i < len {
                        let off = start + i;
                        let aij = *colp.add(off);

                        *yv.get_unchecked_mut(off) += xj * aij;
                        temp2 += aij * *xbuf.get_unchecked(off);

                        i += 1;
                    }

                    *yv.get_unchecked_mut(j) += temp2;
                }
            }

            // non unit strides 
            (UpLo::UpperTriangular, false) => {
                let yv = slice::from_raw_parts_mut(y_unit, n);
                let rs = inc_row_a;
                let cs = inc_col_a;

                for j in 0..n {
                    let xj   = *xbuf.get_unchecked(j);
                    let colb = a.as_ptr().offset((j as isize) * cs);
                    let mut temp2 = 0.0f32;

                    let mut i = 0usize;
                    while i < j {
                        let aij = *colb.offset((i as isize) * rs);
                        *yv.get_unchecked_mut(i) += xj * aij;
                        temp2 += aij * *xbuf.get_unchecked(i);

                        i += 1;
                    }
                    *yv.get_unchecked_mut(j) += *colb.offset((j as isize) * rs) * xj + temp2;
                }
            }
            (UpLo::LowerTriangular, false) => {
                let yv = slice::from_raw_parts_mut(y_unit, n);
                let rs = inc_row_a;
                let cs = inc_col_a;

                for j in 0..n {
                    let xj   = *xbuf.get_unchecked(j);
                    let colb = a.as_ptr().offset((j as isize) * cs);
                    let mut temp2 = 0.0f32;

                    *yv.get_unchecked_mut(j) += *colb.offset((j as isize) * rs) * xj;

                    let mut i = j + 1;
                    while i < n {
                        let aij = *colb.offset((i as isize) * rs);
                        *yv.get_unchecked_mut(i) += xj * aij;
                        temp2 += aij * *xbuf.get_unchecked(i);

                        i += 1;
                    }
                    *yv.get_unchecked_mut(j) += temp2;
                }
            }
        }
    }

    if let Some(ytmp) = ybuf {
        unsafe { copy_back_y_from_unit_f32(n, &ytmp, y, incy); }
    }
}

