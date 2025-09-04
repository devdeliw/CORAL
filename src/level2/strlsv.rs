use core::slice;
use crate::level1::extensions::saxpyf::saxpyf;
use crate::level2::assert_length_helpers::{required_len_ok_mat, required_len_ok_vec};

const BF_F32: usize = 8;

#[inline(always)]
fn strlsv_unblk(
    bf       : usize,
    unit_diag: bool,
    ablk     : *const f32,  // pointer to A[j + j * lda]
    lda      : usize,
    xblk     : *mut f32,    // pointer to to x[j]
) {
    if bf == 0 { return; }

    // contiguous forward substitution 
    // bf x bf lower triangular 
    unsafe {
        for ii in 0..bf {
            let mut sum = 0.0_f32;

            for kk in 0..ii {
                let a_ik = *ablk.add(ii + kk * lda);
                let x_k  = *xblk.add(kk);
                sum     += a_ik * x_k;
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
unsafe fn strlsv_unblk_scalar(
    n         : usize,
    unit_diag : bool,
    a_ptr     : *const f32,
    inc_row_a : isize,
    inc_col_a : isize,
    x_ptr     : *mut f32,
    incx      : isize,
) { unsafe {
    if n == 0 { return; }

    // not contiguous forward substitution 
    #[inline(always)]
    unsafe fn a_ij(
        a: *const f32,
        i: isize,
        j: isize,
        ir: isize,
        ic: isize,
    ) -> *const f32 { unsafe { 
        a.offset(i * ir + j * ic)
    }}

    let mut x0 = x_ptr;
    if incx < 0 {
        x0 = x0.offset((n as isize - 1) * incx);
    }

    for i in 0..n {
        let i_isz = i as isize;

        let mut sum = 0.0f32;
        if i > 0 {
            for j in 0..i {
                let j_isz = j as isize;
                let a_ijp = a_ij(a_ptr, i_isz, j_isz, inc_row_a, inc_col_a);
                let x_j   = *x0.offset(j_isz * incx);
                sum      += *a_ijp * x_j;
            }
        }

        let xi_p = x0.offset(i_isz * incx);
        let mut xi = *xi_p - sum;

        if !unit_diag {
            let a_ii = *a_ij(a_ptr, i_isz, i_isz, inc_row_a, inc_col_a);
            xi /= a_ii;
        }
        *xi_p = xi;
    }
}}

#[inline]
pub fn strlsv(
    n          : usize,
    unit_diag  : bool,
    a          : &[f32],
    inc_row_a  : isize,   
    inc_col_a  : isize,   
    x          : &mut [f32],
    incx       : isize,   
) {
    if n == 0 { return; }

    debug_assert!(
        inc_row_a != 0 && inc_col_a != 0 && incx != 0,
        "strides must be non-zero"
    );
    debug_assert!(
        required_len_ok_vec(x.len(), n, incx),
        "x too short for n/stride"
    );
    debug_assert!(
        required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a),
        "A too short for nxn/strides"
    );

    // fast path 
    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        let lda = inc_col_a as usize;
        let bf  = BF_F32;
        let nl  = n % bf;

        unsafe {
            let mut j = 0usize;
            while j + bf <= n {
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_mut_ptr().add(j);

                strlsv_unblk(bf, unit_diag, ablk, lda, xblk);

                // tail 
                let tail = j + bf;
                if tail < n {
                    let rows         = n - tail;
                    let a_panel_ptr  = a.as_ptr().add(tail + j * lda); 
                    let a_panel_len  = (bf - 1).saturating_mul(lda) + rows;
                    let a_panel      = slice::from_raw_parts(a_panel_ptr, a_panel_len);

                    let mut neg_panel: [f32; BF_F32] = [0.0; BF_F32];
                    for k in 0..bf { neg_panel[k] = -(*xblk.add(k)); }

                    let y_tail = slice::from_raw_parts_mut(x.as_mut_ptr().add(tail), rows);

                    saxpyf(
                        rows,
                        bf,
                        &neg_panel, 1,
                        a_panel,    lda,
                        y_tail,     1,
                    );
                }

                j += bf;
            }

            if nl > 0 {
                let j0    = n - nl;
                let ablk0 = a.as_ptr().add(j0 + j0 * lda);
                let xblk0 = x.as_mut_ptr().add(j0);
                strlsv_unblk(nl, unit_diag, ablk0, lda, xblk0);
            }
        }
    } else {
        // non unit stride 
        unsafe {
            strlsv_unblk_scalar(
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

