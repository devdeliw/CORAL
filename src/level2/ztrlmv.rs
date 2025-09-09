use core::arch::aarch64::{
    vaddvq_f64,
    vdupq_n_f64,
    vld1q_f64,
    vfmaq_f64,
    vnegq_f64,
    vst1q_f64,
    vuzp1q_f64,
    vuzp2q_f64,
};

use crate::level2::assert_length_helpers::{
    required_len_ok_mat,
    required_len_ok_vec,
};
use crate::level2::enums::{
    Diag,
    Trans,
};

const BF_C64: usize = 16;

#[inline(always)]
unsafe fn load2(
    p: *const f64,
) -> (f64, f64) { unsafe {
    (*p, *p.add(1))
}}

#[inline(always)]
fn cmul(
    ar: f64,
    ai: f64,
    xr: f64,
    xi: f64,
) -> (f64, f64) {
    (ar * xr - ai * xi, ar * xi + ai * xr)
}

#[inline(always)]
fn cmul_conj_a(
    ar: f64,
    ai: f64,
    xr: f64,
    xi: f64,
) -> (f64, f64) {
    (ar * xr + ai * xi, ar * xi - ai * xr)
}

#[inline(always)]
unsafe fn add_col_c64_neon(
    acc_re : *mut f64,
    acc_im : *mut f64,
    col    : *const f64, // interleaved
    start  : usize,      // complex offset
    len    : usize,      // complex length
    xr     : f64,
    xi     : f64,
    conj   : bool,
) { unsafe {
    if len == 0 { return; }

    let vxr = vdupq_n_f64(xr);
    let vxi = vdupq_n_f64(xi);

    let mut i   = start;
    let     end = start + len;

    while i + 2 <= end {
        let a0 = vld1q_f64(col.add(2 * i));
        let a1 = vld1q_f64(col.add(2 * i + 2));
        let ar = vuzp1q_f64(a0, a1);
        let ai = vuzp2q_f64(a0, a1);

        let yr0 = vld1q_f64(acc_re.add(i));
        let yi0 = vld1q_f64(acc_im.add(i));

        let mut yr = vfmaq_f64(yr0, ar, vxr);
        let mut yi = vfmaq_f64(yi0, ar, vxi);

        let t1 = if conj { vxi } else { vnegq_f64(vxi) };
        let t2 = if conj { vnegq_f64(vxr) } else { vxr   };

        yr = vfmaq_f64(yr, ai, t1);
        yi = vfmaq_f64(yi, ai, t2);

        vst1q_f64(acc_re.add(i), yr);
        vst1q_f64(acc_im.add(i), yi);

        i += 2;
    }

    while i < end {
        let (ar, ai) = load2(col.add(2 * i));
        let (rr, ii) = if conj {
            cmul_conj_a(ar, ai, xr, xi)
        } else {
            cmul(ar, ai, xr, xi)
        };
        *acc_re.add(i) += rr;
        *acc_im.add(i) += ii;
        i += 1;
    }
}}

#[inline(always)]
unsafe fn caxpyf_panel_cm(
    bf      : usize,
    cols    : usize,
    x_head  : *const f64, // interleaved
    a_panel : *const f64, // interleaved, lda in complex
    lda_c   : usize,
    acc_re  : *mut f64,
    acc_im  : *mut f64,
) { unsafe {
    for t in 0..cols {
        let (xr, xi) = load2(x_head.add(2 * t));
        let col      = a_panel.add(2 * (t * lda_c));
        add_col_c64_neon(
            acc_re,
            acc_im,
            col,
            0,
            bf,
            xr,
            xi,
            false,
        );
    }
}}

#[inline(always)]
unsafe fn ztrlmv_unblk_accum_cm<const BF: usize>(
    unit_diag : bool,
    ablk      : *const f64, // A[j + j*lda], interleaved
    lda_c     : usize,      // complex lda
    xblk      : *const f64, // interleaved
    acc_re    : &mut [f64; BF],
    acc_im    : &mut [f64; BF],
) { unsafe {
    let mut xre: [f64; BF] = [0.0; BF];
    let mut xim: [f64; BF] = [0.0; BF];

    for k in 0..BF {
        let (r, i) = load2(xblk.add(2 * k));
        xre[k] = r;
        xim[k] = i;
    }

    for r in 0..BF {
        let (xr, xi) = (xre[r], xim[r]);
        let col      = ablk.add(2 * (r * lda_c));

        if unit_diag {
            acc_re[r] += xr;
            acc_im[r] += xi;
        } else {
            let (ar, ai) = load2(col.add(2 * r));
            let (rr, ii) = cmul(ar, ai, xr, xi);
            acc_re[r]   += rr;
            acc_im[r]   += ii;
        }

        let start = r + 1;
        if start < BF {
            add_col_c64_neon(
                acc_re.as_mut_ptr(),
                acc_im.as_mut_ptr(),
                col,
                start,
                BF - start,
                xr,
                xi,
                false,
            );
        }
    }
}}

#[inline(always)]
unsafe fn ztrlmv_unblk_accum_cm_var(
    bf        : usize,
    unit_diag : bool,
    ablk      : *const f64,
    lda_c     : usize,
    xblk      : *const f64,
    acc_re    : &mut [f64; BF_C64],
    acc_im    : &mut [f64; BF_C64],
) { unsafe {
    let mut xre: [f64; BF_C64] = [0.0; BF_C64];
    let mut xim: [f64; BF_C64] = [0.0; BF_C64];

    for k in 0..bf {
        let (r, i) = load2(xblk.add(2 * k));
        xre[k] = r;
        xim[k] = i;
    }

    for r in 0..bf {
        let (xr, xi) = (xre[r], xim[r]);
        let col      = ablk.add(2 * (r * lda_c));

        if unit_diag {
            acc_re[r] += xr;
            acc_im[r] += xi;
        } else {
            let (ar, ai) = load2(col.add(2 * r));
            let (rr, ii) = cmul(ar, ai, xr, xi);
            acc_re[r]   += rr;
            acc_im[r]   += ii;
        }

        let start = r + 1;
        if start < bf {
            add_col_c64_neon(
                acc_re.as_mut_ptr(),
                acc_im.as_mut_ptr(),
                col,
                start,
                bf - start,
                xr,
                xi,
                false,
            );
        }
    }
}}

#[inline(always)]
unsafe fn store_acc(
    y   : *mut f64,
    re  : *const f64,
    im  : *const f64,
    len : usize,
) { unsafe {
    for i in 0..len {
        *y.add(2 * i)     = *re.add(i);
        *y.add(2 * i + 1) = *im.add(i);
    }
}}

#[inline(always)]
unsafe fn ztrlmv_unblk_scalar(
    n    : usize,
    unit_diag: bool,
    a    : *const f64,
    ir   : isize,
    ic   : isize,
    x    : *mut f64,
    incx : isize,
) { unsafe {
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn aij(
        a  : *const f64,
        i  : isize,
        j  : isize,
        ir : isize,
        ic : isize,
    ) -> *const f64 { unsafe {
        a.offset(2 * (i * ir + j * ic))
    }}

    let mut x0 = x;
    if incx < 0 {
        x0 = x0.offset(2 * ((n as isize - 1) * incx));
    }

    for i in (0..n).rev() {
        let ii = i as isize;

        let xr = *x0.offset(2 * (ii * incx));
        let xi = *x0.offset(2 * (ii * incx) + 1);

        let (mut sr, mut si) = if unit_diag {
            (xr, xi)
        } else {
            let p        = aij(a, ii, ii, ir, ic);
            let (ar, ai) = (*p, *p.add(1));
            cmul(ar, ai, xr, xi)
        };

        for j in 0..i {
            let jj       = j as isize;
            let p        = aij(a, ii, jj, ir, ic);
            let (ar, ai) = (*p, *p.add(1));
            let xrj      = *x0.offset(2 * (jj * incx));
            let xij      = *x0.offset(2 * (jj * incx) + 1);
            let (rr, ii2)= cmul(ar, ai, xrj, xij);
            sr += rr;
            si += ii2;
        }

        *x0.offset(2 * (ii * incx))     = sr;
        *x0.offset(2 * (ii * incx) + 1) = si;
    }
}}

#[inline(always)]
unsafe fn ztrlmv_unblk_scalar_trans(
    n    : usize,
    unit_diag: bool,
    conj : bool,
    a    : *const f64,
    ir   : isize,
    ic   : isize,
    x    : *mut f64,
    incx : isize,
) { unsafe {
    if n == 0 { return; }

    #[inline(always)]
    unsafe fn aij(
        a  : *const f64,
        i  : isize,
        j  : isize,
        ir : isize,
        ic : isize,
    ) -> *const f64 { unsafe {
        a.offset(2 * (i * ir + j * ic))
    }}

    let mut x0 = x;
    if incx < 0 {
        x0 = x0.offset(2 * ((n as isize - 1) * incx));
    }

    for i in 0..n {
        let ii = i as isize;

        let xr = *x0.offset(2 * (ii * incx));
        let xi = *x0.offset(2 * (ii * incx) + 1);

        let (mut sr, mut si) = if unit_diag {
            (xr, xi)
        } else {
            let p        = aij(a, ii, ii, ir, ic);
            let (ar, ai) = (*p, *p.add(1));
            if conj {
                cmul_conj_a(ar, ai, xr, xi)
            } else {
                cmul(ar, ai, xr, xi)
            }
        };

        for j in (i + 1)..n {
            let jj       = j as isize;
            let p        = aij(a, jj, ii, ir, ic);
            let (ar, ai) = (*p, *p.add(1));
            let xrj      = *x0.offset(2 * (jj * incx));
            let xij      = *x0.offset(2 * (jj * incx) + 1);
            let (rr, ii2)= if conj {
                cmul_conj_a(ar, ai, xrj, xij)
            } else {
                cmul(ar, ai, xrj, xij)
            };
            sr += rr;
            si += ii2;
        }

        *x0.offset(2 * (ii * incx))     = sr;
        *x0.offset(2 * (ii * incx) + 1) = si;
    }
}}

#[inline(always)]
unsafe fn dot_tail_c64_neon(
    col  : *const f64, // interleaved
    x    : *const f64, // interleaved
    len  : usize,
    conj : bool,
) -> (f64, f64) { unsafe {
    let mut sr = vdupq_n_f64(0.0);
    let mut si = vdupq_n_f64(0.0);

    let mut i = 0usize;
    while i + 2 <= len {
        let a0 = vld1q_f64(col.add(2 * i));
        let a1 = vld1q_f64(col.add(2 * i + 2));
        let ar = vuzp1q_f64(a0, a1);
        let ai = vuzp2q_f64(a0, a1);

        let x0 = vld1q_f64(x.add(2 * i));
        let x1 = vld1q_f64(x.add(2 * i + 2));
        let xr = vuzp1q_f64(x0, x1);
        let xi = vuzp2q_f64(x0, x1);

        sr = vfmaq_f64(sr, ar, xr);
        si = vfmaq_f64(si, ar, xi);

        let t1 = if conj { xi } else { vnegq_f64(xi) };
        let t2 = if conj { vnegq_f64(xr) } else { xr   };

        sr = vfmaq_f64(sr, ai, t1);
        si = vfmaq_f64(si, ai, t2);

        i += 2;
    }

    let mut rr = vaddvq_f64(sr);
    let mut ii = vaddvq_f64(si);

    while i < len {
        let (ar, ai) = load2(col.add(2 * i));
        let (xr, xi) = load2(x.add(2 * i));
        let (tr, ti) = if conj {
            cmul_conj_a(ar, ai, xr, xi)
        } else {
            cmul(ar, ai, xr, xi)
        };
        rr += tr;
        ii += ti;
        i  += 1;
    }

    (rr, ii)
}}

#[inline]
fn ztrlmv_notrans(
    n          : usize,
    unit_diag  : bool,
    a          : &[f64],   // interleaved
    inc_row_a  : isize,    // complex units
    inc_col_a  : isize,    // complex units
    x          : &mut [f64],
    incx       : isize,    // complex units
) {
    if n == 0 { return; }

    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        unsafe {
            let lda_c = inc_col_a as usize;
            let bf    = BF_C64;

            let mut j = n;
            while j >= bf {
                j -= bf;

                let ablk = a.as_ptr().add(2 * (j + j * lda_c));
                let xblk = x.as_ptr().add(2 * j);

                let mut acc_re: [f64; BF_C64] = [0.0; BF_C64];
                let mut acc_im: [f64; BF_C64] = [0.0; BF_C64];

                ztrlmv_unblk_accum_cm::<BF_C64>(
                    unit_diag,
                    ablk,
                    lda_c,
                    xblk,
                    &mut acc_re,
                    &mut acc_im,
                );

                if j > 0 {
                    let cols_left = j;
                    let a_panel   = a.as_ptr().add(2 * j);

                    caxpyf_panel_cm(
                        bf,
                        cols_left,
                        x.as_ptr(),
                        a_panel,
                        lda_c,
                        acc_re.as_mut_ptr(),
                        acc_im.as_mut_ptr(),
                    );
                }

                store_acc(
                    x.as_mut_ptr().add(2 * j),
                    acc_re.as_ptr(),
                    acc_im.as_ptr(),
                    bf,
                );
            }

            if j > 0 {
                let ablk = a.as_ptr();
                let xblk = x.as_ptr();

                let mut acc_re: [f64; BF_C64] = [0.0; BF_C64];
                let mut acc_im: [f64; BF_C64] = [0.0; BF_C64];

                ztrlmv_unblk_accum_cm_var(
                    j,
                    unit_diag,
                    ablk,
                    lda_c,
                    xblk,
                    &mut acc_re,
                    &mut acc_im,
                );

                store_acc(
                    x.as_mut_ptr(),
                    acc_re.as_ptr(),
                    acc_im.as_ptr(),
                    j,
                );
            }
        }
    } else {
        unsafe {
            ztrlmv_unblk_scalar(
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
fn ztrlmv_trans_conj(
    n          : usize,
    unit_diag  : bool,
    conj       : bool,
    a          : &[f64],   // interleaved
    inc_row_a  : isize,    // complex units
    inc_col_a  : isize,    // complex units
    x          : &mut [f64],
    incx       : isize,    // complex units
) {
    if n == 0 { return; }

    debug_assert!(inc_row_a != 0 && inc_col_a != 0 && incx != 0);
    debug_assert!(required_len_ok_vec(x.len(), n, incx));
    debug_assert!(required_len_ok_mat(a.len(), n, n, inc_row_a, inc_col_a));

    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 {
        unsafe {
            let lda_c = inc_col_a as usize;
            let bf    = BF_C64;

            let mut j = 0usize;
            while j + bf <= n {
                for r in 0..bf {
                    let c   = j + r;
                    let col = a.as_ptr().add(2 * (c * lda_c));

                    let (xr, xi) = load2(x.as_ptr().add(2 * c));

                    let (mut sr, mut si) = if unit_diag {
                        (xr, xi)
                    } else {
                        let (ar, ai) = load2(col.add(2 * c));
                        if conj {
                            cmul_conj_a(ar, ai, xr, xi)
                        } else {
                            cmul(ar, ai, xr, xi)
                        }
                    };

                    if c + 1 < n {
                        let (tr, ti) = dot_tail_c64_neon(
                            col.add(2 * (c + 1)),
                            x.as_ptr().add(2 * (c + 1)),
                            n - (c + 1),
                            conj,
                        );
                        sr += tr;
                        si += ti;
                    }

                    *x.as_mut_ptr().add(2 * c)     = sr;
                    *x.as_mut_ptr().add(2 * c + 1) = si;
                }
                j += bf;
            }

            let rem = n - j;
            if rem > 0 {
                for r in 0..rem {
                    let c   = j + r;
                    let col = a.as_ptr().add(2 * (c * lda_c));

                    let (xr, xi) = load2(x.as_ptr().add(2 * c));

                    let (mut sr, mut si) = if unit_diag {
                        (xr, xi)
                    } else {
                        let (ar, ai) = load2(col.add(2 * c));
                        if conj {
                            cmul_conj_a(ar, ai, xr, xi)
                        } else {
                            cmul(ar, ai, xr, xi)
                        }
                    };

                    if c + 1 < n {
                        let (tr, ti) = dot_tail_c64_neon(
                            col.add(2 * (c + 1)),
                            x.as_ptr().add(2 * (c + 1)),
                            n - (c + 1),
                            conj,
                        );
                        sr += tr;
                        si += ti;
                    }

                    *x.as_mut_ptr().add(2 * c)     = sr;
                    *x.as_mut_ptr().add(2 * c + 1) = si;
                }
            }
        }
    } else {
        unsafe {
            ztrlmv_unblk_scalar_trans(
                n,
                unit_diag,
                conj,
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
pub(crate) fn ztrlmv(
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
        Trans::NoTrans   => ztrlmv_notrans(
            n,
            unit_diag,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
        Trans::Trans     => ztrlmv_trans_conj(
            n,
            unit_diag,
            false,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
        Trans::ConjTrans => ztrlmv_trans_conj(
            n,
            unit_diag,
            true,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
    }
}

