use core::slice;
use core::arch::aarch64::{vld1q_f64, vfmaq_f64, vst1q_f64, vdupq_n_f64};
use crate::level1::extensions::daxpyf::daxpyf;
use crate::level2::assert_length_helpers::{required_len_ok_mat, required_len_ok_vec};
use crate::level2::enums::{Trans, Diag};

const BF_F64: usize = 16;

#[inline(always)]
unsafe fn add_col_upper_f64_neon(
    acc: *mut f64,
    col: *const f64,
    k  : usize,
    xk : f64,
) { unsafe {
    if k == 0 { return; }

    let vx = vdupq_n_f64(xk);
    let mut i = 0usize;

    while i + 8 <= k {
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

    while i + 2 <= k {
        let a = vld1q_f64(col.add(i));
        let y = vld1q_f64(acc.add(i));
        let y = vfmaq_f64(y, a, vx);
        vst1q_f64(acc.add(i), y);
        i += 2;
    }

    while i < k {
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
unsafe fn dtrumv_unblk_accum_cm<const BF: usize>(
    unit_diag: bool,
    ablk     : *const f64,
    lda      : usize,
    xblk     : *const f64,
    yblk     : *mut f64,
) { unsafe {
    let mut xbuf: [f64; BF] = [0.0; BF];
    for k in 0..BF { xbuf[k] = *xblk.add(k); }

    let mut acc: [f64; BF] = [0.0; BF];

    for k in 0..BF {
        let xk  = xbuf[k];
        let col = ablk.add(k * lda);

        add_col_upper_f64_neon(
            acc.as_mut_ptr(),
            col,
            k,
            xk
        );

        if unit_diag {
            *acc.get_unchecked_mut(k) += xk;
        } else {
            *acc.get_unchecked_mut(k) += *col.add(k) * xk;
        }
    }

    for i in 0..BF { *yblk.add(i) = acc[i]; }
}}

#[inline(always)]
unsafe fn dtrumv_unblk_accum_cm_var(
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

    for k in 0..bf {
        let xk  = xbuf[k];
        let col = ablk.add(k * lda);

        let mut i = 0;
        while i + 2 <= k {
            let a = vld1q_f64(col.add(i));
            let y = vld1q_f64(acc.as_ptr().add(i));
            let y = vfmaq_f64(y, a, vdupq_n_f64(xk));
            vst1q_f64(acc.as_mut_ptr().add(i), y);
            i += 2;
        }
        while i < k {
            acc[i] += *col.add(i) * xk;
            i += 1;
        }

        if unit_diag {
            acc[k] += xk;
        } else {
            acc[k] += *col.add(k) * xk;
        }
    }

    for i in 0..bf { *yblk.add(i) = acc[i]; }
}}

#[inline(always)]
unsafe fn dtrumv_unblk_scalar(
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
unsafe fn dtrumv_unblk_scalar_trans(
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
pub fn dtrumv_notrans(
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
        let lda = inc_col_a as usize;
        let bf  = BF_F64;
        let nl  = n % bf;

        unsafe {
            let mut j = 0usize;
            while j + bf <= n {
                let ablk = a.as_ptr().add(j + j * lda);
                let xblk = x.as_ptr().add(j);
                let yblk = x.as_mut_ptr().add(j);

                dtrumv_unblk_accum_cm::<BF_F64>(
                    unit_diag,
                    ablk,
                    lda,
                    xblk,
                    yblk
                );

                let tail = j + bf;
                if tail < n {
                    let cols        = n - tail;
                    let a_panel_ptr = a.as_ptr().add(j + tail * lda);
                    let a_panel     = colmajor_panel(a_panel_ptr, lda, bf, cols);
                    let y_block     = slice::from_raw_parts_mut(yblk, bf);
                    daxpyf(
                        bf,
                        cols,
                        &x[tail..],
                        1,
                        a_panel,
                        lda,
                        y_block,
                        1
                    );
                }

                j += bf;
            }

            if nl > 0 {
                let j0   = n - nl;
                let ablk = a.as_ptr().add(j0 + j0 * lda);
                let xblk = x.as_ptr().add(j0);
                let yblk = x.as_mut_ptr().add(j0);
                dtrumv_unblk_accum_cm_var(
                    nl,
                    unit_diag,
                    ablk,
                    lda,
                    xblk,
                    yblk
                );
            }
        }
    } else {
        unsafe {
            dtrumv_unblk_scalar(
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
pub fn dtrumv_trans(
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
                let blk_start = j;
                let blk_len   = bf;

                let xblk_ptr = x.as_ptr().add(blk_start);
                let mut xbuf: [f64; BF_F64] = [0.0; BF_F64];
                for r in 0..blk_len { xbuf[r] = *xblk_ptr.add(r); }

                let mut acc: [f64; BF_F64] = [0.0; BF_F64];

                for r in 0..blk_len {
                    let c   = blk_start + r;
                    let col = a.as_ptr().add(c * lda);
                    let mut s = 0.0_f64;

                    if blk_start > 0 {
                        let mut t = 0usize;
                        while t + 2 <= blk_start {
                            s += *col.add(t    ) * *x.as_ptr().add(t    );
                            s += *col.add(t + 1) * *x.as_ptr().add(t + 1);
                            t += 2;
                        }
                        while t < blk_start {
                            s += *col.add(t) * *x.as_ptr().add(t);
                            t += 1;
                        }
                    }

                    if r > 0 {
                        let colj = col.add(blk_start);
                        let mut t = 0usize;
                        while t + 2 <= r {
                            s += *colj.add(t    ) * xbuf[t    ];
                            s += *colj.add(t + 1) * xbuf[t + 1];
                            t += 2;
                        }
                        while t < r {
                            s += *colj.add(t) * xbuf[t];
                            t += 1;
                        }
                    }

                    if unit_diag {
                        s += xbuf[r];
                    } else {
                        s += *col.add(blk_start + r) * xbuf[r];
                    }

                    acc[r] = s;
                }

                let yblk = x.as_mut_ptr().add(blk_start);
                for r in 0..blk_len { *yblk.add(r) = acc[r]; }
            }

            if j > 0 {
                let blk_start = 0usize;
                let blk_len   = j;

                let mut xbuf: [f64; BF_F64] = [0.0; BF_F64];
                for r in 0..blk_len { xbuf[r] = *x.as_ptr().add(r); }

                let mut acc: [f64; BF_F64] = [0.0; BF_F64];

                for r in 0..blk_len {
                    let c   = blk_start + r;
                    let col = a.as_ptr().add(c * lda);
                    let mut s = 0.0_f64;

                    if r > 0 {
                        let mut t = 0usize;
                        while t + 2 <= r {
                            s += *col.add(t    ) * xbuf[t    ];
                            s += *col.add(t + 1) * xbuf[t + 1];
                            t += 2;
                        }
                        while t < r {
                            s += *col.add(t) * xbuf[t];
                            t += 1;
                        }
                    }

                    if unit_diag {
                        s += xbuf[r];
                    } else {
                        s += *col.add(r) * xbuf[r];
                    }

                    acc[r] = s;
                }

                let yblk = x.as_mut_ptr().add(blk_start);
                for r in 0..blk_len { *yblk.add(r) = acc[r]; }
            }
        }
    } else {
        unsafe {
            dtrumv_unblk_scalar_trans(
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
pub fn dtrumv(
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
        Trans::NoTrans => dtrumv_notrans(n, unit_diag, a, inc_row_a, inc_col_a, x, incx),
        Trans::Trans   => dtrumv_trans  (n, unit_diag, a, inc_row_a, inc_col_a, x, incx)
    }
}

