use crate::level3::f64_packers::{MR, NR}; 
use core::arch::aarch64::{ 
    vld1q_f64, 
    vst1q_f64, 
    vfmaq_laneq_f64, 
    vmulq_n_f64, 
    vdupq_n_f64, 
    float64x2_t
};

#[inline(always)] 
fn load_a_rowpairs(ap: *const f64) -> (float64x2_t, float64x2_t, float64x2_t) { 
    unsafe { 
        // packed A per k-step; [a0..a5] 
        let a01 = vld1q_f64(ap.add(0)); 
        let a23 = vld1q_f64(ap.add(2)); 
        let a45 = vld1q_f64(ap.add(4));

        (a01, a23, a45)
    }
}

#[inline(always)] 
fn load_b_colpairs(bp: *const f64) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) { 
    unsafe { 
        // packed B per k-step; [b0..b7] 
        let b01 = vld1q_f64(bp.add(0)); 
        let b23 = vld1q_f64(bp.add(2)); 
        let b45 = vld1q_f64(bp.add(4)); 
        let b67 = vld1q_f64(bp.add(6)); 

        (b01, b23, b45, b67)
    }
}

#[inline(always)] 
fn store_col6(colp: *mut f64, v01: float64x2_t, v23: float64x2_t, v45: float64x2_t) { 
    unsafe { 
        vst1q_f64(colp.add(0), v01);
        vst1q_f64(colp.add(2), v23);
        vst1q_f64(colp.add(4), v45);
    }
}

#[inline(always)]
fn load_col6(colp: *const f64) -> (float64x2_t, float64x2_t, float64x2_t) {
    unsafe { 
        (vld1q_f64(colp.add(0)), vld1q_f64(colp.add(2)), vld1q_f64(colp.add(4))) 
    }
}

#[inline(always)] 
fn kstep_accumulate_laneq( 
    acc01 : &mut [float64x2_t; NR],
    acc23 : &mut [float64x2_t; NR],
    acc45 : &mut [float64x2_t; NR],
    a01   : float64x2_t,
    a23   : float64x2_t,
    a45   : float64x2_t,
    b01   : float64x2_t,
    b23   : float64x2_t,
    b45   : float64x2_t,
    b67   : float64x2_t,
) { 
    unsafe {
        // cols 0..7 as lanes out of b01/23/45/67
        acc01[0] = vfmaq_laneq_f64(acc01[0], a01, b01, 0);
        acc23[0] = vfmaq_laneq_f64(acc23[0], a23, b01, 0);
        acc45[0] = vfmaq_laneq_f64(acc45[0], a45, b01, 0);

        acc01[1] = vfmaq_laneq_f64(acc01[1], a01, b01, 1);
        acc23[1] = vfmaq_laneq_f64(acc23[1], a23, b01, 1);
        acc45[1] = vfmaq_laneq_f64(acc45[1], a45, b01, 1);

        acc01[2] = vfmaq_laneq_f64(acc01[2], a01, b23, 0);
        acc23[2] = vfmaq_laneq_f64(acc23[2], a23, b23, 0);
        acc45[2] = vfmaq_laneq_f64(acc45[2], a45, b23, 0);

        acc01[3] = vfmaq_laneq_f64(acc01[3], a01, b23, 1);
        acc23[3] = vfmaq_laneq_f64(acc23[3], a23, b23, 1);
        acc45[3] = vfmaq_laneq_f64(acc45[3], a45, b23, 1);

        acc01[4] = vfmaq_laneq_f64(acc01[4], a01, b45, 0);
        acc23[4] = vfmaq_laneq_f64(acc23[4], a23, b45, 0);
        acc45[4] = vfmaq_laneq_f64(acc45[4], a45, b45, 0);

        acc01[5] = vfmaq_laneq_f64(acc01[5], a01, b45, 1);
        acc23[5] = vfmaq_laneq_f64(acc23[5], a23, b45, 1);
        acc45[5] = vfmaq_laneq_f64(acc45[5], a45, b45, 1);

        acc01[6] = vfmaq_laneq_f64(acc01[6], a01, b67, 0);
        acc23[6] = vfmaq_laneq_f64(acc23[6], a23, b67, 0);
        acc45[6] = vfmaq_laneq_f64(acc45[6], a45, b67, 0);

        acc01[7] = vfmaq_laneq_f64(acc01[7], a01, b67, 1);
        acc23[7] = vfmaq_laneq_f64(acc23[7], a23, b67, 1);
        acc45[7] = vfmaq_laneq_f64(acc45[7], a45, b67, 1);
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f64_mrxnr_beta0(
    kc    : usize,
    a     : *const f64, // packed; each k-step is [a0..a5]
    b     : *const f64, // packed; each k-step is [b0..b7]
    c     : *mut f64,
    ldc   : usize,
    alpha : f64,
) { 
    unsafe { 
        let mut acc01: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc23: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc45: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            // k
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45, 
                a01, a23, a45, b01, b23, b45, b67
            );

            // k+1
            let (b01n, b23n, b45n, b67n) = load_b_colpairs(bp.add(NR));
            let (a01n, a23n, a45n)       = load_a_rowpairs(ap.add(MR));
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45, 
                a01n, a23n, a45n, b01n, b23n, b45n, b67n
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }

        if kc & 1 != 0 {
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45, 
                a01, a23, a45, b01, b23, b45, b67
            );
        }

        // beta = 0; no read of C 
        for j in 0..NR {
            let colp = c.add(j * ldc);

            let v01 = vmulq_n_f64(acc01[j], alpha);
            let v23 = vmulq_n_f64(acc23[j], alpha);
            let v45 = vmulq_n_f64(acc45[j], alpha);

            store_col6(colp, v01, v23, v45);
        }
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f64_mrxnr_beta1(
    kc    : usize,
    a     : *const f64, 
    b     : *const f64, 
    c     : *mut f64,
    ldc   : usize,
    alpha : f64,
) { 
    unsafe { 
        let mut acc01: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc23: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc45: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        // Unroll ×2 over k
        let k2 = kc / 2;
        for _ in 0..k2 {
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45,
                a01, a23, a45, b01, b23, b45, b67
            );

            let (b01n, b23n, b45n, b67n) = load_b_colpairs(bp.add(NR));
            let (a01n, a23n, a45n)       = load_a_rowpairs(ap.add(MR));
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45,
                a01n, a23n, a45n, b01n, b23n, b45n, b67n
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }
        if kc & 1 != 0 {
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45, 
                a01, a23, a45, b01, b23, b45, b67
            );
        }

        // beta = 1: C += alpha * acc
        for j in 0..NR {
            let colp = c.add(j * ldc);
            let (mut c01, mut c23, mut c45) = load_col6(colp);

            c01 = vfmaq_laneq_f64(c01, acc01[j], vdupq_n_f64(alpha), 0);
            c23 = vfmaq_laneq_f64(c23, acc23[j], vdupq_n_f64(alpha), 0);
            c45 = vfmaq_laneq_f64(c45, acc45[j], vdupq_n_f64(alpha), 0);

            store_col6(colp, c01, c23, c45);
        }
    } 
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f64_mrxnr_betax(
    kc    : usize,
    a     : *const f64,
    b     : *const f64,
    c     : *mut f64,
    ldc   : usize,
    alpha : f64,
    beta  : f64,
) { 
    unsafe { 
        let mut acc01: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc23: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc45: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45,
                a01, a23, a45, b01, b23, b45, b67
            );

            let (b01n, b23n, b45n, b67n) = load_b_colpairs(bp.add(NR));
            let (a01n, a23n, a45n)       = load_a_rowpairs(ap.add(MR));
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45,
                a01n, a23n, a45n, b01n, b23n, b45n, b67n
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }
        if kc & 1 != 0 {
            let (b01, b23, b45, b67) = load_b_colpairs(bp);
            let (a01, a23, a45)      = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc01, &mut acc23, &mut acc45,
                a01, a23, a45, b01, b23, b45, b67
            );
        }

        // general beta: C := beta*C + alpha*acc
        for j in 0..NR {
            let colp = c.add(j * ldc);
            let (mut c01, mut c23, mut c45) = load_col6(colp);

            // cXY *= beta
            c01 = vmulq_n_f64(c01, beta);
            c23 = vmulq_n_f64(c23, beta);
            c45 = vmulq_n_f64(c45, beta);

            // cXY += alpha * accXY
            c01 = vfmaq_laneq_f64(c01, acc01[j], vdupq_n_f64(alpha), 0);
            c23 = vfmaq_laneq_f64(c23, acc23[j], vdupq_n_f64(alpha), 0);
            c45 = vfmaq_laneq_f64(c45, acc45[j], vdupq_n_f64(alpha), 0);

            store_col6(colp, c01, c23, c45);
        }
    } 
}



