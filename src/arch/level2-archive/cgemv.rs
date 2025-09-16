use core::slice;
use crate::level2::{
    enums::Trans,
    assert_length_helpers::{
        required_len_ok_vec, 
        required_len_ok_mat
    },
    buf_helpers::{
        pack_x_unit_c32,
        pack_y_to_unit_c32, 
        copy_back_y_from_unit_c32
    },
};
use crate::level1::{
    cscal::cscal,
    caxpy::caxpy,
};
use crate::level1::extensions::{
    caxpyf::caxpyf,
    cdotuf::cdotuf,
    cdotcf::cdotcf,
};

#[inline(always)]
fn c_is_zero(a: [f32; 2]) -> bool { a[0] == 0.0 && a[1] == 0.0 }
#[inline(always)]
fn c_is_one (a: [f32; 2]) -> bool { a[0] == 1.0 && a[1] == 0.0 }

#[inline(always)]
unsafe fn pack_a_panel_into_c32(
    apack  : &mut Vec<f32>,
    a      : &[f32],
    m      : usize,
    j      : usize,
    nc     : usize,
    rs_f32 : isize,
    cs_f32 : isize,
) { unsafe {
    apack.clear();
    apack.reserve_exact(m * nc * 2);
    apack.set_len(m * nc * 2);

    for c in 0..nc {
        let col_base = a.as_ptr().offset(((j + c) as isize) * cs_f32);

        let mut rp = if rs_f32 >= 0 {
            col_base 
        } else { 
            col_base.offset(((m - 1) as isize) * rs_f32)
        };

        let dst = &mut apack[(c * m * 2)..((c + 1) * m * 2)];
        for r in 0..m {
            *dst.get_unchecked_mut(2 * r)     = *rp;
            *dst.get_unchecked_mut(2 * r + 1) = *rp.add(1);
            rp = rp.offset(rs_f32);
        }
    }
}}

#[inline(always)]
fn cgemv_notrans(
    m          : usize,
    n          : usize,
    alpha      : [f32; 2],
    a          : &[f32],
    inc_row_a  : isize,
    inc_col_a  : isize,   
    x          : &[f32],
    incx       : isize,   
    beta       : [f32; 2],
    y          : &mut [f32],
    incy       : isize,   
) {
    if m == 0 || n == 0 { return; }
    if c_is_zero(alpha) && c_is_one(beta) { return; }

    let rs_f32  = inc_row_a * 2;
    let cs_f32  = inc_col_a * 2;
    let incx_f  = incx * 2;
    let incy_f  = incy * 2;

    debug_assert!(incx != 0 && incy != 0);
    debug_assert!(inc_row_a != 0 && inc_col_a != 0);
    debug_assert!(required_len_ok_vec(x.len().saturating_sub(1), n, incx_f));
    debug_assert!(required_len_ok_vec(y.len().saturating_sub(1), m, incy_f));
    debug_assert!(required_len_ok_mat(a.len().saturating_sub(1), m, n, rs_f32, cs_f32));

    unsafe {
        let mut ybuf: Vec<f32> = Vec::new();
        pack_y_to_unit_c32(m, y, incy, &mut ybuf);
        cscal(m, beta, &mut ybuf, 1);
        if c_is_zero(alpha) { copy_back_y_from_unit_c32(m, &ybuf, y, incy); return; }

        let mut xbuf: Vec<f32> = Vec::new();
        pack_x_unit_c32(n, x, incx, &mut xbuf);
        cscal(n, alpha, &mut xbuf, 1);

        if inc_row_a == 1 && inc_col_a > 0 {
            let lda_f32 = inc_col_a as usize;
            let a_len   = (n - 1).saturating_mul(lda_f32 * 2) + m * 2;
            let a_view  = slice::from_raw_parts(a.as_ptr(), a_len);
            caxpyf(m, n, &xbuf, 1, a_view, lda_f32, &mut ybuf, 1);
        } else {
            const NC: usize = 64;
            let mut apack: Vec<f32> = Vec::new();
            let mut j = 0usize;
            while j < n {
                let nc = core::cmp::min(NC, n - j);
                pack_a_panel_into_c32(&mut apack, a, m, j, nc, rs_f32, cs_f32);
                let a_view = &apack[..(m * nc * 2)];
                caxpyf(m, nc, &xbuf[(2 * j)..(2 * (j + nc))], 1, a_view, m, &mut ybuf, 1);
                j += nc;
            }
        }

        copy_back_y_from_unit_c32(m, &ybuf, y, incy);
    }
}

#[inline(always)]
fn cgemv_trans_like(
    m          : usize,
    n          : usize,
    alpha      : [f32; 2],
    a          : &[f32],
    inc_row_a  : isize,   
    inc_col_a  : isize,   
    x          : &[f32],
    incx       : isize,   
    beta       : [f32; 2],
    y          : &mut [f32],
    incy       : isize,   
    conj       : bool,
) {
    if n == 0 { return; }
    if c_is_zero(alpha) && c_is_one(beta) { return; }

    let rs_f32  = inc_row_a * 2;
    let cs_f32  = inc_col_a * 2;
    let incx_f  = incx * 2;
    let incy_f  = incy * 2;

    debug_assert!(incx != 0 && incy != 0);
    debug_assert!(inc_row_a != 0 && inc_col_a != 0);
    debug_assert!(required_len_ok_vec(x.len().saturating_sub(1), m, incx_f));
    debug_assert!(required_len_ok_vec(y.len().saturating_sub(1), n, incy_f));
    debug_assert!(required_len_ok_mat(a.len().saturating_sub(1), m, n, rs_f32, cs_f32));

    unsafe {
        let mut ybuf: Vec<f32> = Vec::new();

        pack_y_to_unit_c32(n, y, incy, &mut ybuf);
        cscal(n, beta, &mut ybuf, 1);

        if c_is_zero(alpha) { copy_back_y_from_unit_c32(n, &ybuf, y, incy); return; }

        let mut xbuf: Vec<f32> = Vec::new();
        pack_x_unit_c32(m, x, incx, &mut xbuf);

        const BF: usize = 4;
        let mut apack: Vec<f32> = Vec::new();
        let lda_f32 = inc_col_a as usize;

        let mut j = 0usize;
        while j < n {
            let bf = core::cmp::min(BF, n - j);
            let mut tmp: [f32; BF * 2] = [0.0; BF * 2];
            let out = &mut tmp[..(bf * 2)];

            if inc_row_a == 1 && inc_col_a > 0 {
                let a_ptr  = a.as_ptr().add(j * lda_f32 * 2);
                let a_len  = (bf - 1).saturating_mul(lda_f32 * 2) + m * 2;
                let a_view = slice::from_raw_parts(a_ptr, a_len);
                if conj { cdotcf(m, bf, a_view, lda_f32, &xbuf, 1, out); }
                else    { cdotuf(m, bf, a_view, lda_f32, &xbuf, 1, out); }
            } else {
                pack_a_panel_into_c32(&mut apack, a, m, j, bf, rs_f32, cs_f32);
                let a_view   = &apack[..(m * bf * 2)];
                let lda_tile = m;
                if conj { cdotcf(m, bf, a_view, lda_tile, &xbuf, 1, out); }
                else    { cdotuf(m, bf, a_view, lda_tile, &xbuf, 1, out); }
            }

            caxpy(bf, alpha, out, 1, &mut ybuf[(j * 2)..], 1);
            j += bf;
        }

        copy_back_y_from_unit_c32(n, &ybuf, y, incy);
    }
}

#[inline]
pub fn cgemv(
    trans      : Trans,
    m          : usize,
    n          : usize,
    alpha      : [f32; 2],
    a          : &[f32],
    inc_row_a  : isize,   
    inc_col_a  : isize,   
    x          : &[f32],
    incx       : isize,   
    beta       : [f32; 2],
    y          : &mut [f32],
    incy       : isize,   
) {
    match trans {
        Trans::NoTrans   => cgemv_notrans   (m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy),
        Trans::Trans     => cgemv_trans_like(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy, false),
        Trans::ConjTrans => cgemv_trans_like(m, n, alpha, a, inc_row_a, inc_col_a, x, incx, beta, y, incy, true),
    }
}

