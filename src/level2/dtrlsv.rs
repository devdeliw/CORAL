use core::slice;
use crate::level1::extensions::daxpyf::daxpyf;
use crate::level2::assert_length_helpers::{required_len_ok_mat, required_len_ok_vec};
use crate::level2::enums::{ 
    Diag, 
    Trans
}; 

const BF_F64: usize = 8;

#[inline(always)]
fn dtrlsv_unblk(
    bf       : usize,
    unit_diag: bool,
    ablk     : *const f64,  // pointer to A[j + j*lda]   
    lda      : usize,
    xblk     : *mut f64,    // pointer to x[j] 
) {
    if bf == 0 { return; }

    // contiguous forward substitution 
    // bf x bf lower triangular 
    unsafe {
        for ii in 0..bf {
            let mut sum = 0.0;
            for kk in 0..ii {
                sum += *ablk.add(ii + kk * lda) * *xblk.add(kk);
            }
            let mut xi = *xblk.add(ii) - sum;
            if !unit_diag {
                xi /= *ablk.add(ii + ii * lda);
            }
            *xblk.add(ii) = xi;
        }
    }
}

#[inline(always)]
fn dtrlsv_unblk_tdiag(
    bf: usize,
    unit_diag: bool,
    ablk: *const f64,
    lda: usize,
    xblk: *mut f64,
) {
    if bf == 0 { return; }
    unsafe {
        for ii in (0..bf).rev() {
            let mut sum = 0.0f64;
            for kk in (ii + 1)..bf {
                sum += *ablk.add(kk + ii * lda) * *xblk.add(kk);
            }
            let mut xi = *xblk.add(ii) - sum;
            if !unit_diag {
                xi /= *ablk.add(ii + ii * lda);
            }
            *xblk.add(ii) = xi;
        }
    }
}

#[inline(always)]
unsafe fn dtrlsv_unblk_scalar(
    n         : usize,
    unit_diag : bool,
    a_ptr     : *const f64,
    inc_row_a : isize,
    inc_col_a : isize,
    x_ptr     : *mut f64,
    incx      : isize,
) { unsafe { 
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn a_ij(
        a: *const f64, 
        i: isize,
        j: isize, 
        ir: isize, 
        ic: isize
    ) -> *const f64 { unsafe { 
        a.offset(i * ir + j * ic) 
    }}

    let mut x0 = x_ptr;
    if incx < 0 {
        x0 = x0.offset((n as isize - 1) * incx); 
    }

    for i in 0..n {
        let i_isz = i as isize;
        let mut sum = 0.0f64;

        for j in 0..i {
            let j_isz = j as isize;
            sum += *a_ij(
                a_ptr, 
                i_isz, 
                j_isz, 
                inc_row_a,
                inc_col_a
            ) * *x0.offset(j_isz * incx);
        }
        let xi_p = x0.offset(i_isz * incx);
        let mut xi = *xi_p - sum;

        if !unit_diag {
            xi /= *a_ij(
                a_ptr, 
                i_isz, 
                i_isz, 
                inc_row_a,
                inc_col_a
            );
        }
        *xi_p = xi;
    }
}}

#[inline(always)]
unsafe fn dtrlsv_unblk_scalar_trans(
    n         : usize,
    unit_diag : bool,
    a_ptr     : *const f64,
    inc_row_a : isize,
    inc_col_a : isize,
    x_ptr     : *mut f64,
    incx      : isize,
) { unsafe { 
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn a_ij(
        a: *const f64, 
        i: isize, 
        j: isize, 
        ir: isize, 
        ic: isize
    ) -> *const f64 { unsafe { 
        a.offset(i * ir + j * ic) 
    }}

    let mut x0 = x_ptr;
    if incx < 0 {
        x0 = x0.offset((n as isize - 1) * incx);
    }

    for i in (0..n).rev() {
        let i_isz = i as isize;
        let mut sum = 0.0f64;

        for j in (i + 1)..n {
            let j_isz = j as isize;
            sum += *a_ij(
                a_ptr, 
                j_isz,
                i_isz, 
                inc_row_a,
                inc_col_a
            ) * *x0.offset(j_isz * incx);
        }

        let xi_p = x0.offset(i_isz * incx);
        let mut xi = *xi_p - sum;
        if !unit_diag {
            xi /= *a_ij(a_ptr, i_isz, i_isz, inc_row_a, inc_col_a);
        }
        *xi_p = xi;
    }
}}

#[inline(always)]
unsafe fn dgemv_t_small_k<const BF: usize>(
    rows   : usize,      
    bf     : usize,      
    a_pan  : *const f64, 
    lda    : usize,
    x_tail : *const f64, 
    acc    : &mut [f64; BF],
) { unsafe { 
    for r in 0..rows {
        let xr = *x_tail.add(r);
        let row = a_pan.add(r);
        for k in 0..bf {
            *acc.get_unchecked_mut(k) += xr * *row.add(k * lda);
        }
    }
}}

#[inline]
pub fn dtrlsv_notrans(
    n          : usize,
    unit_diag  : bool,
    a          : &[f64],
    inc_row_a  : isize,
    inc_col_a  : isize,
    x          : &mut [f64],
    incx       : isize,
) {
    if n == 0 { return; }
    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    // fast path 
    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        let lda = inc_col_a as usize;
        let bf  = BF_F64;
        let nl  = n % bf;

        unsafe {
            let mut j = 0usize;
            while j + bf <= n {
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_mut_ptr().add(j);

                dtrlsv_unblk(bf, unit_diag, ablk, lda, xblk);

                // tail 
                let tail = j + bf;
                if tail < n {
                    let rows        = n - tail;
                    let a_panel_ptr = a.as_ptr().add(tail + j * lda);
                    let a_panel_len = (bf - 1).saturating_mul(lda) + rows;
                    let a_panel     = slice::from_raw_parts(a_panel_ptr, a_panel_len);

                    let mut neg_panel: [f64; BF_F64] = [0.0; BF_F64];

                    for k in 0..bf { neg_panel[k] = -(*xblk.add(k)); }

                    let y_tail = slice::from_raw_parts_mut(x.as_mut_ptr().add(tail), rows);

                    daxpyf(rows, bf, &neg_panel, 1, a_panel, lda, y_tail, 1);
                }
                j += bf;
            }

            if nl > 0 {
                let j0    = n - nl;
                let ablk0 = a.as_ptr().add(j0 + j0 * lda);
                let xblk0 = x.as_mut_ptr().add(j0);
                dtrlsv_unblk(nl, unit_diag, ablk0, lda, xblk0);
            }
        }
    } else {
        // non unit stride 
        unsafe {
            dtrlsv_unblk_scalar(
                n, unit_diag, a.as_ptr(), inc_row_a, inc_col_a, x.as_mut_ptr(), incx,
            );
        }
    }
}

#[inline]
pub fn dtrlsv_trans(
    n          : usize,
    unit_diag  : bool,
    a          : &[f64],
    inc_row_a  : isize,
    inc_col_a  : isize,
    x          : &mut [f64],
    incx       : isize,
) {
    if n == 0 { return; }
    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        let lda = inc_col_a as usize;
        let bf  = BF_F64;

        unsafe {
            let mut j = n;
            while j > 0 {
                let curr = if j >= bf { bf } else { j };
                j -= curr;

                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_mut_ptr().add(j);

                let tail = j + curr;
                let rows = n - tail;
                if rows > 0 {
                    let a_panel = a.as_ptr().add(tail + j * lda); 
                    let x_tail  = x.as_ptr().add(tail);

                    let mut acc: [f64; BF_F64] = [0.0; BF_F64];
                    dgemv_t_small_k::<BF_F64>(rows, curr, a_panel, lda, x_tail, &mut acc);

                    for k in 0..curr {
                        let p = xblk.add(k);
                        *p = *p - acc[k];
                    }
                }

                dtrlsv_unblk_tdiag(curr, unit_diag, ablk, lda, xblk);
            }
        }
    } else {
        unsafe {
            dtrlsv_unblk_scalar_trans(
                n, unit_diag, a.as_ptr(), inc_row_a, inc_col_a, x.as_mut_ptr(), incx,
            );
        }
    }
}

#[inline]
pub fn dtrlsv(
    n          : usize,
    diag       : Diag,
    trans      : Trans,
    a          : &[f64],
    inc_row_a  : isize,
    inc_col_a  : isize,
    x          : &mut [f64],
    incx       : isize,
) {
    let unit_diag = match diag { 
        Diag::UnitDiag      => true, 
        Diag::NonUnitDiag   => false, 
    };
    match trans {
        Trans::NoTrans   => dtrlsv_notrans(n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
        Trans::Trans     => dtrlsv_trans  (n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
        Trans::ConjTrans => dtrlsv_trans  (n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
    }
}

