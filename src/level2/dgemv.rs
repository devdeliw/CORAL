use core::slice;
use crate::level1::dscal::dscal;
use crate::level1::extensions::{
    daxpyf::daxpyf,
    ddotf::ddotf,
};
use crate::level2::trans::Trans;

#[inline(always)]
fn required_len_ok_vec(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = inc.unsigned_abs() as usize;
    1 + (n - 1) * step <= len
}

#[inline(always)]
fn required_len_ok_mat(len: usize, m: usize, n: usize, rs: isize, cs: isize) -> bool {
    let (rmin, rmax) = if rs >= 0 { (0, (m - 1) as isize * rs) } else { ((m - 1) as isize * rs, 0) };
    let (cmin, cmax) = if cs >= 0 { (0, (n - 1) as isize * cs) } else { ((n - 1) as isize * cs, 0) };
    let min = rmin + cmin;
    let max = rmax + cmax;
    min >= 0 && (max as usize) < len
}

#[inline(always)]
unsafe fn pack_and_scale_x(
    n    : usize,
    alpha: f64,
    x    : &[f64],
    incx : isize,
    dst  : &mut Vec<f64>, // updated scaled x
) { unsafe {
    dst.clear();
    dst.reserve_exact(n);
    dst.set_len(n);

    if n == 0 { return; }

    if incx > 0 {
        let step = incx as usize;
        let mut px = x.as_ptr();
        for i in 0..n {
            *dst.get_unchecked_mut(i) = alpha * *px;
            px = px.add(step);
        }
    } else {
        let step = (-incx) as usize;
        let mut px = x.as_ptr().add((n - 1) * step);
        for i in 0..n {
            *dst.get_unchecked_mut(i) = alpha * *px;
            px = px.sub(step);
        }
    }
}}

#[inline(always)]
unsafe fn pack_y_to_unit(
    m   : usize,
    y   : &[f64],
    incy: isize,
    ybuf: &mut Vec<f64>, // updated unit stride y buf
) { unsafe {
    ybuf.clear();
    ybuf.reserve_exact(m);
    ybuf.set_len(m);

    if m == 0 { return; }

    if incy > 0 {
        let step = incy as usize;
        let mut py = y.as_ptr();
        for i in 0..m {
            *ybuf.get_unchecked_mut(i) = *py;
            py = py.add(step);
        }
    } else {
        let step = (-incy) as usize;
        let mut py = y.as_ptr().add((m - 1) * step);
        for i in 0..m {
            *ybuf.get_unchecked_mut(i) = *py;
            py = py.sub(step);
        }
    }
}}

#[inline(always)]
unsafe fn copy_back_y_from_unit(
    m   : usize,
    ybuf: &[f64],
    y   : &mut [f64],
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
    apack: &mut Vec<f64>,
    a  : &[f64],
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
fn dgemv_notrans(
    m           : usize,
    n           : usize,
    alpha       : f64,
    a           : &[f64],
    inc_row_a   : isize,   // col major; should be 1
    inc_col_a   : isize,   // col major; should be lda
    x           : &[f64],
    incx        : isize,
    beta        : f64,
    y           : &mut [f64],
    incy        : isize,
) {
    // quick return
    if m == 0 || n == 0 { return; }
    if alpha == 0.0 && beta == 1.0 { return; }

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
        dscal(m, beta, y, incy);
    }
    if alpha == 0.0 { return; }

    let mut xbuf: Vec<f64> = Vec::new();
    unsafe { pack_and_scale_x(n, alpha, x, incx, &mut xbuf); }

    // if y non unit stride, pack into unit and write back
    let mut ybuf: Option<Vec<f64>> = None;
    let y_unit: *mut f64;

    if incy == 1 {
        y_unit = y.as_mut_ptr();
    } else {
        let mut tmp = Vec::<f64>::new();
        unsafe { pack_y_to_unit(m, y, incy, &mut tmp); }

        y_unit = tmp.as_mut_ptr();
        ybuf   = Some(tmp);
    }

    // fast path; one shot
    if inc_row_a == 1 && inc_col_a > 0 {
        let lda = inc_col_a as usize;
        assert!(lda >= m, "lda must be >= m when inc_row_a==1");

        unsafe {
            let a_len  = (n - 1).saturating_mul(lda) + m;
            let a_view = slice::from_raw_parts(a.as_ptr(), a_len);

            if let Some(ref mut ytmp) = ybuf {
                daxpyf(m, n, &xbuf, 1, a_view, lda, ytmp, 1);
            } else {
                let y_slice = slice::from_raw_parts_mut(y_unit, m);
                daxpyf(m, n, &xbuf, 1, a_view, lda, y_slice, 1);
            }
        }
    } else {
        // general path; packing by NC=128 cols
        const NC: usize = 128;

        let mut apack: Vec<f64> = Vec::new();

        let mut j = 0usize;
        while j < n {
            let nc = core::cmp::min(NC, n - j);

            if inc_row_a == 1 && inc_col_a > 0 {
                let lda = inc_col_a as usize;

                unsafe {
                    let a_ptr  = a.as_ptr().add(j * lda);
                    let a_len  = (nc - 1).saturating_mul(lda) + m;
                    let a_view = slice::from_raw_parts(a_ptr, a_len);

                    if let Some(ref mut ytmp) = ybuf {
                        daxpyf(m, nc, &xbuf[j..j + nc], 1, a_view, lda, ytmp, 1);
                    } else {
                        let y_slice = slice::from_raw_parts_mut(y_unit, m);
                        daxpyf(m, nc, &xbuf[j..j + nc], 1, a_view, lda, y_slice, 1);
                    }
                }
            } else {
                unsafe {
                    pack_a_panel_into(&mut apack, a, m, j, nc, inc_row_a, inc_col_a);

                    if let Some(ref mut ytmp) = ybuf {
                        daxpyf(m, nc, &xbuf[j..j + nc], 1, &apack, m, ytmp, 1);
                    } else {
                        let y_slice = slice::from_raw_parts_mut(y_unit, m);
                        daxpyf(m, nc, &xbuf[j..j + nc], 1, &apack, m, y_slice, 1);
                    }
                }
            }

            j += nc;
        }
    }

    if let Some(ytmp) = ybuf {
        unsafe { copy_back_y_from_unit(m, &ytmp, y, incy); }
    }
}

#[inline(always)]
fn dgemv_trans(
    m           : usize,
    n           : usize,
    alpha       : f64,
    a           : &[f64],
    inc_row_a   : isize,   
    inc_col_a   : isize,   
    x           : &[f64],  
    incx        : isize,
    beta        : f64,
    y           : &mut [f64], 
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
        dscal(n, beta, y, incy);
    }
    if alpha == 0.0 { return; }

    // pack & scale x by alpha 
    let mut xbuf: Vec<f64> = Vec::new();
    unsafe { pack_and_scale_x(m, alpha, x, incx, &mut xbuf); }

    const BF: usize = 4; 
    let mut apack: Vec<f64> = Vec::new();

    unsafe {
        let col_major_fast = inc_row_a == 1 && inc_col_a > 0;
        let lda   = inc_col_a as usize;
        let mut j = 0usize;

        while j < n {
            let bf = core::cmp::min(BF, n - j);

            if col_major_fast {
                let a_ptr  = a.as_ptr().add(j * lda);
                let a_len  = (bf - 1).saturating_mul(lda) + m;
                let a_view = slice::from_raw_parts(a_ptr, a_len);

                if incy == 1 {
                    ddotf(m, bf, a_view, lda, &xbuf, 1, &mut y[j..j + bf]);
                } else {
                    let mut tmp = [0.0f64; BF];

                    ddotf(m, bf, a_view, lda, &xbuf, 1, &mut tmp[..bf]);

                    let sy = incy as usize;
                    for k in 0..bf {
                        *y.get_unchecked_mut((j + k) * sy) += tmp[k];
                    }
                }
            } else {
                // pack A[:, j..j+bf-1); contiguous (m x bf) 
                pack_a_panel_into(&mut apack, a, m, j, bf, inc_row_a, inc_col_a);
                let a_view   = &apack[..(m * bf)];
                let lda_tile = m;

                if incy == 1 {
                    ddotf(m, bf, a_view, lda_tile, &xbuf, 1, &mut y[j..j + bf]);
                } else {
                    let mut tmp = [0.0f64; BF];
                    ddotf(m, bf, a_view, lda_tile, &xbuf, 1, &mut tmp[..bf]);

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
pub fn dgemv(
    trans       : Trans,
    m           : usize,
    n           : usize,
    alpha       : f64,
    a           : &[f64],
    inc_row_a   : isize,
    inc_col_a   : isize,
    x           : &[f64],
    incx        : isize,
    beta        : f64,
    y           : &mut [f64],
    incy        : isize,
) {
    match trans {
        Trans::NoTrans => dgemv_notrans(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
        Trans::Trans   => dgemv_trans  (m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
    }
}

