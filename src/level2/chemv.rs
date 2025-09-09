use core::slice;
use core::arch::aarch64::{
    vdupq_n_f32, 
    vld1q_f32, 
    vst1q_f32, 
    vmulq_f32, 
    vmlsq_f32, 
    vfmaq_f32, 
    vaddq_f32,
    vaddvq_f32,
    vuzp1q_f32, 
    vuzp2q_f32, 
    vzip1q_f32,
    vzip2q_f32,
};
use crate::level1::cscal::cscal;
use crate::level2::{
    enums::UpLo,
    buf_helpers::{
        pack_x_unit_c32, 
        pack_y_to_unit_c32, 
        copy_back_y_from_unit_c32,
    },
    assert_length_helpers::{ 
        required_len_ok_vec, 
        required_len_ok_mat 
    },
};

#[inline]
pub fn chemv(
    uplo        : UpLo,
    n           : usize,
    alpha       : [f32; 2],
    a           : &[f32],     
    inc_row_a   : isize,      
    inc_col_a   : isize,      
    x           : &[f32],
    incx        : isize,      
    beta        : [f32; 2],
    y           : &mut [f32], 
    incy        : isize,    
) {
    if n == 0 { return; }
    if alpha == [0.0, 0.0] && beta == [1.0, 0.0] { return; }

    debug_assert!(incx  != 0 && incy != 0, "vector increments must be non-zero");
    debug_assert!(inc_row_a != 0 && inc_col_a != 0, "matrix strides must be non-zero");
    debug_assert!(required_len_ok_vec(x.len(), n, incx * 2), "x too short for n/incx (HEMV)");
    debug_assert!(required_len_ok_vec(y.len(), n, incy * 2), "y too short for n/incy (HEMV)");
    debug_assert!(
        required_len_ok_mat(a.len(), n, n, inc_row_a * 2, inc_col_a * 2),
        "a too short for n x n/strides (HEMV)"
    );

    if beta != [1.0, 0.0] {
        cscal(n, beta, y, incy);
    }
    if alpha == [0.0, 0.0] { return; }

    // pack x to unit stride and scale by alpha
    let mut xbuf: Vec<f32> = Vec::new();
    unsafe { pack_x_unit_c32(n, x, incx, &mut xbuf); }
    cscal(n, alpha, &mut xbuf, 1);

    // y buffer 
    let mut ybuf: Option<Vec<f32>> = None;
    let y_unit: *mut f32 = if incy == 1 {
        y.as_mut_ptr()
    } else {
        let mut tmp = Vec::<f32>::new();
        unsafe { pack_y_to_unit_c32(n, y, incy, &mut tmp); }
        let p = tmp.as_mut_ptr();
        ybuf = Some(tmp);
        p
    };

    let col_major_fast = inc_row_a == 1 && inc_col_a > 0;

    unsafe {
        match (uplo, col_major_fast) {
            (UpLo::UpperTriangular, true) => {
                let lda = (inc_col_a as usize) * 2;
                let yv  = slice::from_raw_parts_mut(y_unit, 2 * n);

                for j in 0..n {
                    let xr = *xbuf.get_unchecked(2*j + 0);
                    let xi = *xbuf.get_unchecked(2*j + 1);
                    let colp = a.as_ptr().add(j * lda);

                    let mut acc_re0 = vdupq_n_f32(0.0);
                    let mut acc_im0 = vdupq_n_f32(0.0);

                    let mut i = 0usize;
                    let v_xr = vdupq_n_f32(xr);
                    let v_xi = vdupq_n_f32(xi);

                    while i + 4 <= j {
                        let off = 2 * i;

                        // A block (4 complex), Y block
                        let a0 = vld1q_f32(colp.add(off + 0));
                        let a1 = vld1q_f32(colp.add(off + 4));
                        let y0 = vld1q_f32(yv.as_ptr().add(off + 0));
                        let y1 = vld1q_f32(yv.as_ptr().add(off + 4));

                        // deinterleave A
                        let ar = vuzp1q_f32(a0, a1);
                        let ai = vuzp2q_f32(a0, a1);

                        // y += A * xj
                        let t_re = vmlsq_f32(vmulq_f32(ar, v_xr), ai, v_xi);
                        let t_im = vfmaq_f32(vmulq_f32(ar, v_xi), ai, v_xr);
                        let add_lo = vzip1q_f32(t_re, t_im);
                        let add_hi = vzip2q_f32(t_re, t_im);
                        let y0 = vaddq_f32(y0, add_lo);
                        let y1 = vaddq_f32(y1, add_hi);
                        vst1q_f32(yv.as_mut_ptr().add(off + 0), y0);
                        vst1q_f32(yv.as_mut_ptr().add(off + 4), y1);

                        // temp2 += dot(conj(A), xbuf)
                        let x0 = vld1q_f32(xbuf.as_ptr().add(off + 0));
                        let x1 = vld1q_f32(xbuf.as_ptr().add(off + 4));
                        let xr_v = vuzp1q_f32(x0, x1);
                        let xi_v = vuzp2q_f32(x0, x1);

                        acc_re0 = vfmaq_f32(acc_re0, ar, xr_v);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi_v);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi_v);
                        acc_im0 = vmlsq_f32(acc_im0, ai, xr_v);

                        i += 4;
                    }

                    let mut temp2_re = vaddvq_f32(acc_re0);
                    let mut temp2_im = vaddvq_f32(acc_im0);

                    while i < j {
                        let off = 2 * i;
                        let ar = *colp.add(off + 0);
                        let ai = *colp.add(off + 1);

                        // y[i] += A[i,j] * xj
                        let yre = yv.as_mut_ptr().add(off + 0);
                        let yim = yv.as_mut_ptr().add(off + 1);
                        *yre += ar * xr - ai * xi;
                        *yim += ar * xi + ai * xr;

                        // temp2 += conj(A[i,j]) * x[i]
                        let xr_i = *xbuf.get_unchecked(off + 0);
                        let xi_i = *xbuf.get_unchecked(off + 1);
                        temp2_re += ar * xr_i + ai * xi_i;
                        temp2_im += ar * xi_i - ai * xr_i;

                        i += 1;
                    }

                    // diagonal 
                    let ajj_re = *colp.add(2*j + 0);
                    let yjr = yv.as_mut_ptr().add(2*j + 0);
                    let yji = yv.as_mut_ptr().add(2*j + 1);
                    *yjr += ajj_re * xr + temp2_re;
                    *yji += ajj_re * xi + temp2_im;
                }
            }

            (UpLo::LowerTriangular, true) => {
                let lda = (inc_col_a as usize) * 2;
                let yv  = slice::from_raw_parts_mut(y_unit, 2 * n);

                for j in 0..n {
                    let xr = *xbuf.get_unchecked(2*j + 0);
                    let xi = *xbuf.get_unchecked(2*j + 1);
                    let colp = a.as_ptr().add(j * lda);

                    // y[j] += A[j,j].real * xj
                    let ajj_re = *colp.add(2*j + 0);
                    let yjr = yv.as_mut_ptr().add(2*j + 0);
                    let yji = yv.as_mut_ptr().add(2*j + 1);
                    *yjr += ajj_re * xr;
                    *yji += ajj_re * xi;

                    let start = j + 1;
                    let len   = n.saturating_sub(start);

                    let mut acc_re0 = vdupq_n_f32(0.0);
                    let mut acc_im0 = vdupq_n_f32(0.0);

                    let mut i = 0usize;
                    let v_xr = vdupq_n_f32(xr);
                    let v_xi = vdupq_n_f32(xi);

                    while i + 4 <= len {
                        let off = 2 * (start + i);

                        let a0 = vld1q_f32(colp.add(off + 0));
                        let a1 = vld1q_f32(colp.add(off + 4));
                        let y0 = vld1q_f32(yv.as_ptr().add(off + 0));
                        let y1 = vld1q_f32(yv.as_ptr().add(off + 4));

                        let ar = vuzp1q_f32(a0, a1);
                        let ai = vuzp2q_f32(a0, a1);

                        // y += A * xj
                        let t_re = vmlsq_f32(vmulq_f32(ar, v_xr), ai, v_xi);
                        let t_im = vfmaq_f32(vmulq_f32(ar, v_xi), ai, v_xr);
                        let add_lo = vzip1q_f32(t_re, t_im);
                        let add_hi = vzip2q_f32(t_re, t_im);
                        let y0 = vaddq_f32(y0, add_lo);
                        let y1 = vaddq_f32(y1, add_hi);
                        vst1q_f32(yv.as_mut_ptr().add(off + 0), y0);
                        vst1q_f32(yv.as_mut_ptr().add(off + 4), y1);

                        // temp2 += dot(conj(A), xbuf[start..])
                        let x0 = vld1q_f32(xbuf.as_ptr().add(off + 0));
                        let x1 = vld1q_f32(xbuf.as_ptr().add(off + 4));
                        let xr_v = vuzp1q_f32(x0, x1);
                        let xi_v = vuzp2q_f32(x0, x1);

                        acc_re0 = vfmaq_f32(acc_re0, ar, xr_v);
                        acc_re0 = vfmaq_f32(acc_re0, ai, xi_v);
                        acc_im0 = vfmaq_f32(acc_im0, ar, xi_v);
                        acc_im0 = vmlsq_f32(acc_im0, ai, xr_v);

                        i += 4;
                    }

                    let mut temp2_re = vaddvq_f32(acc_re0);
                    let mut temp2_im = vaddvq_f32(acc_im0);

                    while i < len {
                        let off = 2 * (start + i);
                        let ar = *colp.add(off + 0);
                        let ai = *colp.add(off + 1);

                        let yre = yv.as_mut_ptr().add(off + 0);
                        let yim = yv.as_mut_ptr().add(off + 1);
                        *yre += ar * xr - ai * xi;
                        *yim += ar * xi + ai * xr;

                        let xr_i = *xbuf.get_unchecked(off + 0);
                        let xi_i = *xbuf.get_unchecked(off + 1);
                        temp2_re += ar * xr_i + ai * xi_i;
                        temp2_im += ar * xi_i - ai * xr_i;

                        i += 1;
                    }

                    *yjr += temp2_re;
                    *yji += temp2_im;
                }
            }

            // generic strides 
            (UpLo::UpperTriangular, false) => {
                let yv = slice::from_raw_parts_mut(y_unit, 2 * n);
                let rs = inc_row_a * 2;
                let cs = inc_col_a * 2;

                for j in 0..n {
                    let xr = *xbuf.get_unchecked(2*j + 0);
                    let xi = *xbuf.get_unchecked(2*j + 1);
                    let colb = a.as_ptr().offset((j as isize) * cs);

                    let mut tr = 0.0f32;
                    let mut ti = 0.0f32;

                    let mut i = 0usize;
                    while i < j {
                        let p = colb.offset((i as isize) * rs);
                        let ar = *p.add(0);
                        let ai = *p.add(1);

                        let yre = yv.as_mut_ptr().add(2*i + 0);
                        let yim = yv.as_mut_ptr().add(2*i + 1);
                        *yre += ar * xr - ai * xi;
                        *yim += ar * xi + ai * xr;

                        let xr_i = *xbuf.get_unchecked(2*i + 0);
                        let xi_i = *xbuf.get_unchecked(2*i + 1);

                        // temp2 += dot(A*, x)
                        tr += ar * xr_i + ai * xi_i;    
                        ti += ar * xi_i - ai * xr_i;

                        i += 1;
                    }

                    let ajj_re = *colb.offset((j as isize) * rs).add(0);
                    let yjr = yv.as_mut_ptr().add(2*j + 0);
                    let yji = yv.as_mut_ptr().add(2*j + 1);
                    *yjr += ajj_re * xr + tr;
                    *yji += ajj_re * xi + ti;
                }
            }

            (UpLo::LowerTriangular, false) => {
                let yv = slice::from_raw_parts_mut(y_unit, 2 * n);
                let rs = inc_row_a * 2;
                let cs = inc_col_a * 2;

                for j in 0..n {
                    let xr = *xbuf.get_unchecked(2*j + 0);
                    let xi = *xbuf.get_unchecked(2*j + 1);
                    let colb = a.as_ptr().offset((j as isize) * cs);

                    let ajj_re = *colb.offset((j as isize) * rs).add(0);
                    let yjr = yv.as_mut_ptr().add(2*j + 0);
                    let yji = yv.as_mut_ptr().add(2*j + 1);
                    *yjr += ajj_re * xr;
                    *yji += ajj_re * xi;

                    let mut tr = 0.0f32;
                    let mut ti = 0.0f32;

                    let mut i = j + 1;
                    while i < n {
                        let p = colb.offset((i as isize) * rs);
                        let ar = *p.add(0);
                        let ai = *p.add(1);

                        let yre = yv.as_mut_ptr().add(2*i + 0);
                        let yim = yv.as_mut_ptr().add(2*i + 1);
                        *yre += ar * xr - ai * xi;
                        *yim += ar * xi + ai * xr;

                        let xr_i = *xbuf.get_unchecked(2*i + 0);
                        let xi_i = *xbuf.get_unchecked(2*i + 1);

                        // temp2 += dot(A*, x)
                        tr += ar * xr_i + ai * xi_i;    
                        ti += ar * xi_i - ai * xr_i;

                        i += 1;
                    }

                    *yjr += tr;
                    *yji += ti;
                }
            }
        }
    }

    if let Some(ytmp) = ybuf {
        unsafe { copy_back_y_from_unit_c32(n, &ytmp, y, incy); }
    }
}

