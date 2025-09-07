use core::slice;
use crate::level1::extensions::daxpyf::daxpyf;
use crate::level2::assert_length_helpers::{required_len_ok_mat, required_len_ok_vec};
use crate::level2::enums::{ 
    Trans, 
    Diag, 
};

const BF_F64: usize = 8;

#[inline(always)]
fn dtrusv_unblk(
    bf        : usize,
    unit_diag : bool,
    ablk      : *const f64,  
    lda       : usize,
    xblk      : *mut f64,     
) {
    if bf == 0 { return; }

    // contiguous backward substitution
    // bf x bf upper triangular 
    unsafe {
        for ii in (0..bf).rev() {
            let mut sum = 0.0_f64;
            let row_i_ptr = ablk.add(ii); 

            for kk in (ii + 1)..bf {
                let a_ik  = *row_i_ptr.add(kk * lda); 
                let x_k   = *xblk.add(kk);
                sum      += a_ik * x_k;
            }

            let mut xi = *xblk.add(ii) - sum;
            if !unit_diag {
                let a_ii = *ablk.add(ii + ii * lda);
                xi      /= a_ii;
            }
            *xblk.add(ii) = xi;
        }
    }
}

#[inline(always)]
fn dtrusv_unblk_tdiag(
    bf: usize,
    unit_diag: bool,
    ablk: *const f64,
    lda: usize,
    xblk: *mut f64,
) {
    if bf == 0 { return; }
    unsafe {
        for ii in 0..bf {
            let mut sum = 0.0f64;
            for kk in 0..ii {
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
unsafe fn dtrusv_unblk_scalar(
    n         : usize,
    unit_diag : bool,
    a_ptr     : *const f64,   
    inc_row_a : isize,
    inc_col_a : isize,
    x_ptr     : *mut f64,     
    incx      : isize,
) { unsafe { 
    if n == 0 { return; }

    // non contiguous forward substitution 
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

    for i in (0..n).rev() {
        let i_isz = i as isize;

        let mut sum = 0.0f64;
        if i + 1 < n {
            for j in (i + 1)..n {
                let j_isz = j as isize;

                let a_ijp = a_ij(
                    a_ptr, 
                    i_isz,
                    j_isz,
                    inc_row_a, 
                    inc_col_a
                );

                let x_j   = *x_ptr.offset(j_isz * incx);
                sum      += *a_ijp * x_j;
            }
        }

        let xi_p = x_ptr.offset(i_isz * incx);
        let mut xi = *xi_p - sum;

        if !unit_diag {
            let a_ii = *a_ij(a_ptr, i_isz, i_isz, inc_row_a, inc_col_a);
            xi /= a_ii;
        }
        *xi_p = xi;
    }
}}

#[inline(always)]
unsafe fn dtrusv_unblk_scalar_trans(
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
unsafe fn ut_update_tail(
    rows   : usize,
    bf     : usize,
    base   : *const f64, // pointer to A[j + tail*lda]
    lda    : usize,
    xblk   : *const f64, // pointer to x[j]
    ytail  : *mut f64,   
) { unsafe { 
    for c in 0..rows {
        let mut t = 0.0f64;
        let col = base.add(c * lda);
        for k in 0..bf {
            t += *col.add(k) * *xblk.add(k);
        }
        *ytail.add(c) -= t;
    }
}} 

#[inline]
pub fn dtrusv_notrans(
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
        let bf  = BF_F64;   // block width 
        let nl  = n % bf;   // remainder 

        unsafe {
            let mut j = if n >= bf { n - bf } else { usize::MAX };
            while j != usize::MAX {
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_mut_ptr().add(j);

                dtrusv_unblk(bf, unit_diag, ablk, lda, xblk);

                // tail 
                if j > 0 {
                    let a_panel_ptr = a.as_ptr().add(j * lda);   
                    let a_panel_len = (bf - 1).saturating_mul(lda) + j;
                    let a_panel     = slice::from_raw_parts(a_panel_ptr, a_panel_len);

                    let mut neg_panel: [f64; BF_F64] = [0.0; BF_F64];

                    for k in 0..bf { 
                        neg_panel[k] = -(*xblk.add(k)); 
                    }

                    let y_head = slice::from_raw_parts_mut(x.as_mut_ptr(), j);

                    daxpyf(
                        j,
                        bf,
                        &neg_panel, 1,
                        a_panel, lda,
                        y_head, 1,
                    );
                }

                if j >= bf { j -= bf } else { break; }
            }

            if nl > 0 {
                let ablk0 = a.as_ptr();     
                let xblk0 = x.as_mut_ptr();  

                dtrusv_unblk(nl, unit_diag, ablk0, lda, xblk0);
            }
        }
    } else {
        // non unit stride 
        unsafe {
            dtrusv_unblk_scalar(
                n,
                unit_diag,
                a.as_ptr(),
                inc_row_a,
                inc_col_a,
                x.as_mut_ptr(),
                incx,
            );
        }
    }
}

#[inline]
pub fn dtrusv_trans(
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
        let nl  = n % bf;   

        unsafe {
            let mut j = 0usize;
            while j + bf <= n {
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_mut_ptr().add(j);

                dtrusv_unblk_tdiag(bf, unit_diag, ablk, lda, xblk);

                let tail = j + bf;
                if tail < n {
                    let rows = n - tail;
                    let base = a.as_ptr().add(j + tail * lda);
                    let ytl  = x.as_mut_ptr().add(tail);
                    ut_update_tail(rows, bf, base, lda, xblk, ytl);
                }

                j += bf;
            }

            if nl > 0 {
                let j0   = n - nl;
                let ablk = a.as_ptr().add(j0 + j0 * lda);
                let xblk = x.as_mut_ptr().add(j0);
                dtrusv_unblk_tdiag(nl, unit_diag, ablk, lda, xblk);
            }
        }
    } else {
        unsafe {
            dtrusv_unblk_scalar_trans(
                n,
                unit_diag,
                a.as_ptr(),
                inc_row_a,
                inc_col_a,
                x.as_mut_ptr(),
                incx,
            );
        }
    }
}

#[inline]
pub fn dtrusv(
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
        Diag::UnitDiag    => true, 
        Diag::NonUnitDiag => false, 
    };  
    match trans {
        Trans::NoTrans => dtrusv_notrans(n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
        Trans::Trans   => dtrusv_trans  (n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
    }
}

