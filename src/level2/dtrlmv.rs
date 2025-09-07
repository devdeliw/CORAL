use core::slice;
use core::arch::aarch64::{vld1q_f64, vst1q_f64, vfmaq_f64, vdupq_n_f64};
use crate::level1::extensions::daxpyf::daxpyf;
use crate::level2::assert_length_helpers::{required_len_ok_mat, required_len_ok_vec};
use crate::level2::enums::{Trans, Diag};

const BF_F64: usize = 16;

#[inline(always)]
unsafe fn add_col_lower_f64_neon(
    acc  : *mut f64,
    col  : *const f64,
    start: usize,
    len  : usize,
    xk   : f64,
) { unsafe {
    if len == 0 { return; }

    let vx  = vdupq_n_f64(xk);
    let mut i = start;
    let end   = start + len;

    while i + 8 <= end {
        let a0 = vld1q_f64(col.add(i));
        let y0 = vld1q_f64(acc.add(i));
        let y0 = vfmaq_f64(y0, a0, vx);
        vst1q_f64(acc.add(i), y0);

        let a1 = vld1q_f64(col.add(i + 2));
        let y1 = vld1q_f64(acc.add(i + 2));
        let y1 = vfmaq_f64(y1, a1, vx);
        vst1q_f64(acc.add(i + 2), y1);

        let a2 = vld1q_f64(col.add(i + 4));
        let y2 = vld1q_f64(acc.add(i + 4));
        let y2 = vfmaq_f64(y2, a2, vx);
        vst1q_f64(acc.add(i + 4), y2);

        let a3 = vld1q_f64(col.add(i + 6));
        let y3 = vld1q_f64(acc.add(i + 6));
        let y3 = vfmaq_f64(y3, a3, vx);
        vst1q_f64(acc.add(i + 6), y3);

        i += 8;
    }

    while i + 2 <= end {
        let a = vld1q_f64(col.add(i));
        let y = vld1q_f64(acc.add(i));
        let y = vfmaq_f64(y, a, vx);
        vst1q_f64(acc.add(i), y);
        i += 2;
    }

    while i < end {
        *acc.add(i) += *col.add(i) * xk;
        i += 1;
    }
}}

#[inline(always)]
unsafe fn colmajor_panel<'a>(
    ptr : *const f64,
    lda : usize,
    rows: usize,
    cols: usize
) -> &'a [f64] { unsafe {
    slice::from_raw_parts(
        ptr,
        (cols.saturating_sub(1)).saturating_mul(lda) + rows
    )
}}

#[inline(always)]
unsafe fn dtrlmv_unblk_accum_cm<const BF: usize>(
    unit_diag: bool,
    ablk     : *const f64,
    lda      : usize,
    xblk     : *const f64,
    yblk     : *mut f64,
) { unsafe {
    let mut xbuf: [f64; BF] = [0.0; BF];
    for k in 0..BF { xbuf[k] = *xblk.add(k); }

    let mut acc: [f64; BF] = [0.0; BF];

    for r in 0..BF {
        let xk  = xbuf[r];
        let col = ablk.add(r * lda);

        if unit_diag {
            *acc.get_unchecked_mut(r) += xk;
        } else {
            *acc.get_unchecked_mut(r) += *col.add(r) * xk;
        }

        let start = r + 1;
        if start < BF {
            add_col_lower_f64_neon(
                acc.as_mut_ptr(),
                col,
                start,
                BF - start,
                xk
            );
        }
    }

    for i in 0..BF {
        *yblk.add(i) = acc[i];
    }
}}

#[inline(always)]
unsafe fn dtrlmv_unblk_accum_cm_var(
    bf       : usize,
    unit_diag: bool,
    ablk     : *const f64,
    lda      : usize,
    xblk     : *const f64,
    yblk     : *mut f64,
) { unsafe {
    debug_assert!(bf <= BF_F64);

    let mut xbuf: [f64; BF_F64] = [0.0; BF_F64];
    for k in 0..bf { xbuf[k] = *xblk.add(k); }

    let mut acc: [f64; BF_F64] = [0.0; BF_F64];

    for r in 0..bf {
        let xk  = xbuf[r];
        let col = ablk.add(r * lda);

        if unit_diag {
            acc[r] += xk;
        } else {
            acc[r] += *col.add(r) * xk;
        }

        let mut i = r + 1;
        while i + 4 <= bf {
            acc[i    ] += *col.add(i    ) * xk;
            acc[i + 1] += *col.add(i + 1) * xk;
            acc[i + 2] += *col.add(i + 2) * xk;
            acc[i + 3] += *col.add(i + 3) * xk;
            i += 4;
        }
        while i < bf {
            acc[i] += *col.add(i) * xk;
            i += 1;
        }
    }

    for i in 0..bf {
        *yblk.add(i) = acc[i];
    }
}}

#[inline(always)]
unsafe fn dtrlmv_unblk_scalar(
    n        : usize,
    unit_diag: bool,
    a_ptr    : *const f64,
    inc_row_a: isize,
    inc_col_a: isize,
    x_ptr    : *mut f64,
    incx     : isize,
) { unsafe {
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn a_ij(
        a : *const f64,
        i : isize,
        j : isize,
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
        let ii = i as isize;

        let mut sum = if unit_diag {
            *x0.offset(ii * incx)
        } else {
            *a_ij(
                a_ptr,
                ii,
                ii,
                inc_row_a,
                inc_col_a
            ) * *x0.offset(ii * incx)
        };

        for j in 0..i {
            let jj = j as isize;
            sum += *a_ij(
                a_ptr,
                ii,
                jj,
                inc_row_a,
                inc_col_a
            ) * *x0.offset(jj * incx);
        }

        *x0.offset(ii * incx) = sum;
    }
}}

#[inline(always)]
unsafe fn dtrlmv_unblk_scalar_trans(
    n        : usize,
    unit_diag: bool,
    a_ptr    : *const f64,
    inc_row_a: isize,
    inc_col_a: isize,
    x_ptr    : *mut f64,
    incx     : isize,
) { unsafe {
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn a_ij(
        a : *const f64,
        i : isize,
        j : isize,
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
        let ii = i as isize;

        let mut sum = if unit_diag {
            *x0.offset(ii * incx)
        } else {
            *a_ij(
                a_ptr,
                ii,
                ii,
                inc_row_a,
                inc_col_a
            ) * *x0.offset(ii * incx)
        };

        for j in (i + 1)..n {
            let jj = j as isize;
            sum += *a_ij(
                a_ptr,
                jj,
                ii,
                inc_row_a,
                inc_col_a
            ) * *x0.offset(jj * incx);
        }

        *x0.offset(ii * incx) = sum;
    }
}}

#[inline]
pub fn dtrlmv_notrans(
    n         : usize,
    unit_diag : bool,
    a         : &[f64],
    inc_row_a : isize,
    inc_col_a : isize,
    x         : &mut [f64],
    incx      : isize,
) {
    if n == 0 { return; }

    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        unsafe {
            let lda = inc_col_a as usize;
            let bf  = BF_F64;

            let mut j = n;
            while j >= bf {
                j -= bf;
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_ptr().add(j);
                let yblk = x.as_mut_ptr().add(j);

                dtrlmv_unblk_accum_cm::<BF_F64>(unit_diag, ablk, lda, xblk, yblk);

                if j > 0 {
                    let cols_left   = j;
                    let a_panel_ptr = a.as_ptr().add(j);
                    let a_panel     = colmajor_panel(a_panel_ptr, lda, bf, cols_left);
                    let y_block     = slice::from_raw_parts_mut(yblk, bf);
                    daxpyf(bf, cols_left, &x[..j], 1, a_panel, lda, y_block, 1);
                }
            }

            if j > 0 {
                let ablk = a.as_ptr().add(0);
                let xblk = x.as_ptr().add(0);
                let yblk = x.as_mut_ptr().add(0);
                dtrlmv_unblk_accum_cm_var(j, unit_diag, ablk, lda, xblk, yblk);
            }
        }
    } else {
        unsafe {
            dtrlmv_unblk_scalar(
                n,
                unit_diag,
                a.as_ptr(),
                inc_row_a,
                inc_col_a,
                x.as_mut_ptr(),
                incx
            );
        }
    }
}

#[inline]
pub fn dtrlmv_trans(
    n         : usize,
    unit_diag : bool,
    a         : &[f64],
    inc_row_a : isize,
    inc_col_a : isize,
    x         : &mut [f64],
    incx      : isize,
) {
    if n == 0 { return; }

    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        unsafe {
            let lda = inc_col_a as usize;
            let bf  = BF_F64;

            let mut j = 0usize;
            while j + bf <= n {
                let xblk_ptr = x.as_ptr().add(j);
                let mut xbuf: [f64; BF_F64] = [0.0; BF_F64];
                for r in 0..bf { xbuf[r] = *xblk_ptr.add(r); }

                let mut acc: [f64; BF_F64] = [0.0; BF_F64];

                for r in 0..bf {
                    let c   = j + r;
                    let col = a.as_ptr().add(c * lda);
                    let mut s = 0.0_f64;

                    if unit_diag {
                        s += xbuf[r];
                    } else {
                        s += *col.add(j + r) * xbuf[r];
                    }

                    let mut t = r + 1;
                    while t + 4 <= bf {
                        s += *col.add(j + t    ) * xbuf[t    ];
                        s += *col.add(j + t + 1) * xbuf[t + 1];
                        s += *col.add(j + t + 2) * xbuf[t + 2];
                        s += *col.add(j + t + 3) * xbuf[t + 3];
                        t += 4;
                    }
                    while t < bf {
                        s += *col.add(j + t) * xbuf[t];
                        t += 1;
                    }

                    if j + bf < n {
                        let tail = j + bf;
                        let mut u = tail;
                        while u + 4 <= n {
                            s += *col.add(u    ) * *x.as_ptr().add(u    );
                            s += *col.add(u + 1) * *x.as_ptr().add(u + 1);
                            s += *col.add(u + 2) * *x.as_ptr().add(u + 2);
                            s += *col.add(u + 3) * *x.as_ptr().add(u + 3);
                            u += 4;
                        }
                        while u < n {
                            s += *col.add(u) * *x.as_ptr().add(u);
                            u += 1;
                        }
                    }

                    acc[r] = s;
                }

                let yblk = x.as_mut_ptr().add(j);
                for r in 0..bf { *yblk.add(r) = acc[r]; }

                j += bf;
            }

            let rem = n - j;
            if rem > 0 {
                let mut xbuf: [f64; BF_F64] = [0.0; BF_F64];
                for r in 0..rem { xbuf[r] = *x.as_ptr().add(j + r); }
                let mut acc: [f64; BF_F64] = [0.0; BF_F64];

                for r in 0..rem {
                    let c   = j + r;
                    let col = a.as_ptr().add(c * lda);
                    let mut s = 0.0_f64;

                    if unit_diag {
                        s += xbuf[r];
                    } else {
                        s += *col.add(j + r) * xbuf[r];
                    }

                    let mut t = r + 1;
                    while t < rem {
                        s += *col.add(j + t) * xbuf[t];
                        t += 1;
                    }

                    if j + rem < n {
                        let tail = j + rem;
                        let mut u = tail;
                        while u < n {
                            s += *col.add(u) * *x.as_ptr().add(u);
                            u += 1;
                        }
                    }

                    acc[r] = s;
                }

                let yblk = x.as_mut_ptr().add(j);
                for r in 0..rem { *yblk.add(r) = acc[r]; }
            }
        }
    } else {
        unsafe {
            dtrlmv_unblk_scalar_trans(
                n,
                unit_diag,
                a.as_ptr(),
                inc_row_a,
                inc_col_a,
                x.as_mut_ptr(),
                incx
            );
        }
    }
}

#[inline]
pub fn dtrlmv(
    n          : usize,
    diag       : Diag,
    trans      : Trans,
    a          : &[f64],
    inc_row_a  : isize,
    inc_col_a  : isize,
    x          : &mut [f64],
    incx       : isize,
) {
    let unit_diag = matches!(diag, Diag::UnitDiag);
    match trans {
        Trans::NoTrans => dtrlmv_notrans(n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
        Trans::Trans   => dtrlmv_trans  (n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
    }
}

