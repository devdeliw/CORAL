use core::slice;
use crate::level1::sscal::sscal;
use crate::level1::extensions::{ 
    saxpyf::saxpyf, 
    sdotf::sdotf, 
}; 
use crate::level2::{
    trans::Trans, 
    assert_length_helpers::{ 
        required_len_ok_vec,
        required_len_ok_mat, 
    }, 
}; 

#[inline]
fn ensure_len(buf: &mut Vec<f32>, n: usize) {
    if buf.len() != n {
        buf.resize(n, 0.0);
    }
}

#[inline]
pub unsafe fn pack_and_scale_x(
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: isize,        
    dst: &mut Vec<f32>, // unit-stride, scaled x
) { unsafe { 
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    if n == 0 { return; }
    ensure_len(dst, n);

    if alpha == 0.0 {
        core::ptr::write_bytes(dst.as_mut_ptr(), 0, n * core::mem::size_of::<f32>());
        return;
    }
    if incx == 1 && alpha == 1.0 {
        core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), n);
        return;
    }

    let pd = dst.as_mut_ptr();

    // fast path 
    if incx == 1 {
        let a = vdupq_n_f32(alpha);
        let mut i = 0usize;
        while i + 16 <= n {
            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p.add(0));
            let x1 = vld1q_f32(p.add(4));
            let x2 = vld1q_f32(p.add(8));
            let x3 = vld1q_f32(p.add(12));

            vst1q_f32(pd.add(i + 0),  vmulq_f32(x0, a));
            vst1q_f32(pd.add(i + 4),  vmulq_f32(x1, a));
            vst1q_f32(pd.add(i + 8),  vmulq_f32(x2, a));
            vst1q_f32(pd.add(i + 12), vmulq_f32(x3, a));

            i += 16;
        }
        while i + 8 <= n {

            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p.add(0));
            let x1 = vld1q_f32(p.add(4));

            vst1q_f32(pd.add(i + 0), vmulq_f32(x0, a));
            vst1q_f32(pd.add(i + 4), vmulq_f32(x1, a));

            i += 8;
        }
        while i + 4 <= n {
            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p);

            vst1q_f32(pd.add(i), vmulq_f32(x0, a));

            i += 4;
        }
        while i < n {
            *pd.add(i) = alpha * *x.as_ptr().add(i);
            i += 1;
        }
    } else {
        // non-unit stride 
        let step  = incx.unsigned_abs() as usize;
        let mut i = 0usize;

        let mut idx  = if incx > 0 { 0usize as isize } else { ((n - 1) * step) as isize };
        let delta    = if incx > 0 { step as isize } else { -(step as isize) };

        while i + 4 <= n {
            let p0 = x.as_ptr().offset(idx + 0 * delta);
            let p1 = x.as_ptr().offset(idx + 1 * delta);
            let p2 = x.as_ptr().offset(idx + 2 * delta);
            let p3 = x.as_ptr().offset(idx + 3 * delta);

            *pd.add(i + 0) = alpha * *p0;
            *pd.add(i + 1) = alpha * *p1;
            *pd.add(i + 2) = alpha * *p2;
            *pd.add(i + 3) = alpha * *p3;

            idx += 4 * delta;
            i   += 4;
        }
        while i < n {
            let p = x.as_ptr().offset(idx);
            *pd.add(i) = alpha * *p;

            idx += delta;
            i   += 1;
        }
    }
}}

#[inline]
pub unsafe fn pack_y_to_unit(
    m: usize,
    y: &[f32],
    incy: isize,        
    ybuf: &mut Vec<f32>,
) { unsafe {
    if m == 0 { return; }
    ensure_len(ybuf, m);

    let pd = ybuf.as_mut_ptr();

    if incy == 1 {
        core::ptr::copy_nonoverlapping(y.as_ptr(), pd, m);
        return;
    }

    let step  = incy.unsigned_abs() as usize;
    let mut i = 0usize;

    let mut idx  = if incy > 0 { 0usize as isize } else { ((m - 1) * step) as isize };
    let delta    = if incy > 0 { step as isize } else { -(step as isize) };

    while i + 4 <= m {
        let p0 = y.as_ptr().offset(idx + 0 * delta);
        let p1 = y.as_ptr().offset(idx + 1 * delta);
        let p2 = y.as_ptr().offset(idx + 2 * delta);
        let p3 = y.as_ptr().offset(idx + 3 * delta);

        *pd.add(i + 0) = *p0;
        *pd.add(i + 1) = *p1;
        *pd.add(i + 2) = *p2;
        *pd.add(i + 3) = *p3;

        idx += 4 * delta;
        i   += 4;
    }

    while i < m {
        *pd.add(i) = *y.as_ptr().offset(idx);
        idx += delta;
        i   += 1;
    }
}}

#[inline(always)]
unsafe fn copy_back_y_from_unit(
    m   : usize,
    ybuf: &[f32],
    y   : &mut [f32],
    incy: isize,
) { unsafe {
    if m == 0 { return; }

    if incy > 0 {
        let step   = incy as usize;
        let mut py = y.as_mut_ptr();
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.add(step);
        }
    } else {
        let step   = (-incy) as usize;
        let mut py = y.as_mut_ptr().add((m - 1) * step);
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.sub(step);
        }
    }
}}

#[inline(always)]
unsafe fn pack_a_panel_into(
    apack: &mut Vec<f32>,
    a  : &[f32],
    m  : usize,
    j  : usize,
    nc : usize,
    rs : isize,
    cs : isize,
) { unsafe {
    apack.clear();
    apack.reserve_exact(m * nc);
    apack.set_len(m * nc);

    for c in 0..nc {
        let col_base = a.as_ptr().offset(((j + c) as isize) * cs);
        let mut rp = if rs >= 0 {
            col_base
        } else {
            col_base.offset(((m - 1) as isize) * rs)
        };
        let dst_col = &mut apack[(c * m)..((c + 1) * m)];
        for r in 0..m {
            *dst_col.get_unchecked_mut(r) = *rp;
            rp = rp.offset(rs);
        }
    }
}}


#[inline(always)]
fn sgemv_notrans(
    m           : usize,
    n           : usize,
    alpha       : f32,
    a           : &[f32],
    inc_row_a   : isize,   // col major; should be 1
    inc_col_a   : isize,   // col major; should be lda
    x           : &[f32],
    incx        : isize,
    beta        : f32,
    y           : &mut [f32],
    incy        : isize,
) {
    // quick return
    if m == 0 || n == 0 {
        return;
    }

    if alpha == 0.0 && beta == 1.0 {
        return;
    }

    debug_assert!(incx != 0 && incy != 0, "vector increments must be non-zero");
    debug_assert!(inc_row_a != 0 && inc_col_a != 0, "matrix strides must be non-zero");
    debug_assert!(required_len_ok_vec(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_vec(y.len(), m, incy), "y too short for m/incy");
    debug_assert!(
        required_len_ok_mat(a.len(), m, n, inc_row_a, inc_col_a),
        "a too short for m x n/strides"
    );

    // y := beta * y
    if beta != 1.0 {
        sscal(m, beta, y, incy);
    }

    if alpha == 0.0 {
        return;
    }

    let mut xbuf: Vec<f32> = Vec::new();
    unsafe { pack_and_scale_x(n, alpha, x, incx, &mut xbuf); }

    // if y non unit stride, pack into unit and write back
    let mut ybuf: Option<Vec<f32>> = None;
    let y_unit: *mut f32;

    if incy == 1 {
        y_unit = y.as_mut_ptr();
    } else {
        let mut tmp = Vec::<f32>::new();
        unsafe { pack_y_to_unit(m, y, incy, &mut tmp); }

        y_unit = tmp.as_mut_ptr();
        ybuf   = Some(tmp);
    }

    // fast path; one shot
    if inc_row_a == 1 && inc_col_a > 0 {
        let lda = inc_col_a as usize;
        assert!(lda >= m, "lda must be >= m when inc_row_a==1");

        unsafe {
            // full view of A, all n cols
            let a_len  = (n - 1).saturating_mul(lda) + m;
            let a_view = slice::from_raw_parts(a.as_ptr(), a_len);

            // fused saxpyf only takes unit stride bufs
            if let Some(ref mut ytmp) = ybuf {
                saxpyf(m, n, &xbuf, 1, a_view, lda, ytmp, 1);
            } else{
                let y_slice = slice::from_raw_parts_mut(y_unit, m);
                saxpyf(m, n, &xbuf, 1, a_view, lda, y_slice, 1);
            }
        }
    } else {
        // general path; packing by NC=128 cols
        const NC: usize = 128;

        // buf for packing
        let mut apack: Vec<f32> = Vec::new();

        let mut j = 0usize;
        while j < n {
            let nc = core::cmp::min(NC, n - j);

            if inc_row_a == 1 && inc_col_a > 0 {
                let lda = inc_col_a as usize;

                // A subview for cols [j, ..., j + nc]
                unsafe {
                    let a_ptr  = a.as_ptr().add(j * lda);
                    let a_len  = (nc - 1).saturating_mul(lda) + m;
                    let a_view = slice::from_raw_parts(a_ptr, a_len);

                    if let Some(ref mut ytmp) = ybuf {
                        saxpyf(m, nc, &xbuf[j..j + nc], 1, a_view, lda, ytmp, 1);
                    } else {
                        let y_slice = slice::from_raw_parts_mut(y_unit, m);
                        saxpyf(m, nc, &xbuf[j..j + nc], 1, a_view, lda, y_slice, 1);
                    }
                }
            } else {
                // non unit row stride
                unsafe {
                    pack_a_panel_into(&mut apack, a, m, j, nc, inc_row_a, inc_col_a);

                    if let Some(ref mut ytmp) = ybuf {
                        saxpyf(m, nc, &xbuf[j..j + nc], 1, &apack, m, ytmp, 1);
                    } else {
                        let y_slice = slice::from_raw_parts_mut(y_unit, m);
                        saxpyf(m, nc, &xbuf[j..j + nc], 1, &apack, m, y_slice, 1);
                    }
                }
            }

            j += nc;
        }
    }

    // copy back to y
    if let Some(ytmp) = ybuf {
        unsafe { copy_back_y_from_unit(m, &ytmp, y, incy); }
    }
}

#[inline(always)]
fn sgemv_trans(
    m           : usize,
    n           : usize,
    alpha       : f32,
    a           : &[f32],
    inc_row_a   : isize,   
    inc_col_a   : isize,   
    x           : &[f32],  
    incx        : isize,
    beta        : f32,
    y           : &mut [f32], 
    incy        : isize,
) {
    if n == 0 { return; }
    if alpha == 0.0 && beta == 1.0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be non-zero");
    debug_assert!(inc_row_a != 0 && inc_col_a != 0, "matrix strides must be non-zero");
    debug_assert!(required_len_ok_vec(x.len(), m, incx), "x too short for m/incx (TRANS)");
    debug_assert!(required_len_ok_vec(y.len(), n, incy), "y too short for n/incy (TRANS)");
    debug_assert!(
        required_len_ok_mat(a.len(), m, n, inc_row_a, inc_col_a),
        "a too short for m x n/strides (TRANS)"
    );

    if beta != 1.0 {
        sscal(n, beta, y, incy);
    }
    if alpha == 0.0 { return; }

    let mut xbuf: Vec<f32> = Vec::new();
    unsafe { pack_and_scale_x(m, alpha, x, incx, &mut xbuf); }

    const BF: usize = 4;
    let mut apack: Vec<f32> = Vec::new();

    unsafe {
        let col_major_fast = inc_row_a == 1 && inc_col_a > 0;
        let lda = inc_col_a as usize;
        let mut j = 0usize;

        while j < n {
            let bf = core::cmp::min(BF, n - j);

            if col_major_fast {
                let a_ptr  = a.as_ptr().add(j * lda);
                let a_len  = (bf - 1).saturating_mul(lda) + m;
                let a_view = slice::from_raw_parts(a_ptr, a_len);

                if incy == 1 {
                    sdotf(m, bf, a_view, lda, &xbuf, 1, &mut y[j..j + bf]);
                } else {
                    let mut tmp = [0.0f32; BF];
                    sdotf(m, bf, a_view, lda, &xbuf, 1, &mut tmp[..bf]);
                    let sy = incy as usize;
                    for k in 0..bf {
                        *y.get_unchecked_mut((j + k) * sy) += tmp[k];
                    }
                }
            } else {
                pack_a_panel_into(&mut apack, a, m, j, bf, inc_row_a, inc_col_a);
                let a_view = &apack[..(m * bf)];
                let lda_tile = m;

                if incy == 1 {
                    sdotf(m, bf, a_view, lda_tile, &xbuf, 1, &mut y[j..j + bf]);
                } else {
                    let mut tmp = [0.0f32; BF];
                    sdotf(m, bf, a_view, lda_tile, &xbuf, 1, &mut tmp[..bf]);
                    let sy = incy as usize;
                    for k in 0..bf {
                        *y.get_unchecked_mut((j + k) * sy) += tmp[k];
                    }
                }
            }

            j += bf;
        }
    }
}


#[inline]
pub fn sgemv(
    trans       : Trans,
    m           : usize,
    n           : usize,
    alpha       : f32,
    a           : &[f32],
    inc_row_a   : isize,
    inc_col_a   : isize,
    x           : &[f32],
    incx        : isize,
    beta        : f32,
    y           : &mut [f32],
    incy        : isize,
) {
    match trans {
        Trans::NoTrans => sgemv_notrans(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
        Trans::Trans   => sgemv_trans  (m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
    }
}

