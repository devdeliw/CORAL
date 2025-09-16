use core::slice;
use crate::level2::{
    enums::Trans,
    assert_length_helpers::{required_len_ok_vec, required_len_ok_mat},
    buf_helpers::{
        pack_x_unit_c64,
        pack_y_to_unit_c64,
        copy_back_y_from_unit_c64,
    },
};
use crate::level1::{
    zscal::zscal,
    zaxpy::zaxpy,
};
use crate::level1::extensions::{
    zaxpyf::zaxpyf,
    zdotuf::zdotuf,
    zdotcf::zdotcf,
};

#[inline(always)]
fn z_is_zero(a: [f64; 2]) -> bool { a[0] == 0.0 && a[1] == 0.0 }
#[inline(always)]
fn z_is_one (a: [f64; 2]) -> bool { a[0] == 1.0 && a[1] == 0.0 }

#[inline(always)]
unsafe fn pack_a_panel_into_c64(
    apack  : &mut Vec<f64>,
    a      : &[f64],
    m      : usize,
    j      : usize,
    nc     : usize,
    rs_f64 : isize,
    cs_f64 : isize,
) { unsafe {
    apack.clear();
    apack.reserve_exact(m * nc * 2);
    apack.set_len(m * nc * 2);

    for c in 0..nc {
        let col_base = a.as_ptr().offset(((j + c) as isize) * cs_f64);

        let mut rp = if rs_f64 >= 0 { 
            col_base 
        } else {
            col_base.offset(((m - 1) as isize) * rs_f64) 
        };

        let dst = &mut apack[(c * m * 2)..((c + 1) * m * 2)];
        for r in 0..m {
            *dst.get_unchecked_mut(2 * r)     = *rp;
            *dst.get_unchecked_mut(2 * r + 1) = *rp.add(1);
            rp = rp.offset(rs_f64);
        }
    }
}}

#[inline(always)]
fn zgemv_notrans(
    m          : usize,
    n          : usize,
    alpha      : [f64; 2],
    a          : &[f64],
    inc_row_a  : isize,   
    inc_col_a  : isize,   
    x          : &[f64],
    incx       : isize,   
    beta       : [f64; 2],
    y          : &mut [f64],
    incy       : isize,   
) {
    if m == 0 || n == 0 { return; }
    if z_is_zero(alpha) && z_is_one(beta) { return; }

    let rs_f64 = inc_row_a * 2;
    let cs_f64 = inc_col_a * 2;
    let incx_f = incx * 2;
    let incy_f = incy * 2;

    debug_assert!(incx != 0 && incy != 0);
    debug_assert!(inc_row_a != 0 && inc_col_a != 0);
    debug_assert!(required_len_ok_vec(x.len().saturating_sub(1), n, incx_f));
    debug_assert!(required_len_ok_vec(y.len().saturating_sub(1), m, incy_f));
    debug_assert!(required_len_ok_mat(a.len().saturating_sub(1), m, n, rs_f64, cs_f64));

    unsafe {
        let mut ybuf: Vec<f64> = Vec::new();
        pack_y_to_unit_c64(m, y, incy, &mut ybuf);
        zscal(m, beta, &mut ybuf, 1);

        if z_is_zero(alpha) { 
            copy_back_y_from_unit_c64(m, &ybuf, y, incy); 
            return;
        }

        let mut xbuf: Vec<f64> = Vec::new();
        pack_x_unit_c64(n, x, incx, &mut xbuf);
        zscal(n, alpha, &mut xbuf, 1);

        if inc_row_a == 1 && inc_col_a > 0 {
            let lda_c  = inc_col_a as usize;
            let a_len  = (n - 1).saturating_mul(lda_c * 2) + m * 2;
            let a_view = slice::from_raw_parts(a.as_ptr(), a_len);
            zaxpyf(m, n, &xbuf, 1, a_view, lda_c, &mut ybuf, 1);
        } else {
            const NC: usize = 64;
            let mut apack: Vec<f64> = Vec::new();
            let mut j = 0usize;
            while j < n {
                let nc = core::cmp::min(NC, n - j);
                pack_a_panel_into_c64(&mut apack, a, m, j, nc, rs_f64, cs_f64);

                let a_view = &apack[..(m * nc * 2)];
                zaxpyf(m, nc, &xbuf[(2 * j)..(2 * (j + nc))], 1, a_view, m, &mut ybuf, 1);
                j += nc;
            }
        }

        copy_back_y_from_unit_c64(m, &ybuf, y, incy);
    }
}

#[inline(always)]
fn zgemv_trans_like(
    m          : usize,
    n          : usize,
    alpha      : [f64; 2],
    a          : &[f64],
    inc_row_a  : isize, 
    inc_col_a  : isize,   
    x          : &[f64],
    incx       : isize,   
    beta       : [f64; 2],
    y          : &mut [f64],
    incy       : isize,   
    conj       : bool,
) {
    if n == 0 { return; }
    if z_is_zero(alpha) && z_is_one(beta) { return; }

    let rs_f64 = inc_row_a * 2;
    let cs_f64 = inc_col_a * 2;
    let incx_f = incx * 2;
    let incy_f = incy * 2;

    debug_assert!(incx != 0 && incy != 0);
    debug_assert!(inc_row_a != 0 && inc_col_a != 0);
    debug_assert!(required_len_ok_vec(x.len().saturating_sub(1), m, incx_f));
    debug_assert!(required_len_ok_vec(y.len().saturating_sub(1), n, incy_f));
    debug_assert!(required_len_ok_mat(a.len().saturating_sub(1), m, n, rs_f64, cs_f64));

    unsafe {
        let mut ybuf: Vec<f64> = Vec::new();
        pack_y_to_unit_c64(n, y, incy, &mut ybuf);
        zscal(n, beta, &mut ybuf, 1);

        if z_is_zero(alpha) {
            copy_back_y_from_unit_c64(n, &ybuf, y, incy); 
            return; 
        }

        let mut xbuf: Vec<f64> = Vec::new();
        pack_x_unit_c64(m, x, incx, &mut xbuf);

        const BF: usize = 4;
        let mut apack: Vec<f64> = Vec::new();
        let lda_c = inc_col_a as usize;

        let mut j = 0usize;
        while j < n {
            let bf = core::cmp::min(BF, n - j);
            let mut tmp: [f64; BF * 2] = [0.0; BF * 2];
            let out = &mut tmp[..(bf * 2)];

            if inc_row_a == 1 && inc_col_a > 0 {
                let a_ptr  = a.as_ptr().add(j * lda_c * 2);
                let a_len  = (bf - 1).saturating_mul(lda_c * 2) + m * 2;
                let a_view = slice::from_raw_parts(a_ptr, a_len);
                if conj { zdotcf(m, bf, a_view, lda_c, &xbuf, 1, out); }
                else    { zdotuf(m, bf, a_view, lda_c, &xbuf, 1, out); }
            } else {
                pack_a_panel_into_c64(&mut apack, a, m, j, bf, rs_f64, cs_f64);
                let a_view   = &apack[..(m * bf * 2)];
                let lda_tile = m;
                if conj { zdotcf(m, bf, a_view, lda_tile, &xbuf, 1, out); }
                else    { zdotuf(m, bf, a_view, lda_tile, &xbuf, 1, out); }
            }

            zaxpy(bf, alpha, out, 1, &mut ybuf[(j * 2)..], 1);
            j += bf;
        }

        copy_back_y_from_unit_c64(n, &ybuf, y, incy);
    }
}

#[inline]
pub fn zgemv(
    trans      : Trans,
    m          : usize,
    n          : usize,
    alpha      : [f64; 2],
    a          : &[f64],
    inc_row_a  : isize,   
    inc_col_a  : isize,   
    x          : &[f64],
    incx       : isize,   
    beta       : [f64; 2],
    y          : &mut [f64],
    incy       : isize,   
) {
    match trans {
        Trans::NoTrans   => zgemv_notrans   (m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
        Trans::Trans     => zgemv_trans_like(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy, false),
        Trans::ConjTrans => zgemv_trans_like(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy, true),
    }
}

