use crate::level3::f32_packers::{MR, NR};
use core::arch::aarch64::{
    float32x4_t,
    vdupq_n_f32, 
    vld1q_f32, 
    vfmaq_laneq_f32, 
    vmulq_n_f32, 
    vst1q_f32,
};

#[inline(always)]
fn load_a_rowquads(ap: *const f32) -> (float32x4_t, float32x4_t) {
    unsafe {
        // packed A per k-step; [a0..a7]
        let a0123 = vld1q_f32(ap.add(0)); 
        let a4567 = vld1q_f32(ap.add(4)); 
        (a0123, a4567)
    }
}

#[inline(always)]
fn load_b_colquads(bp: *const f32) -> (float32x4_t, float32x4_t, float32x4_t) {
    unsafe {
        // packed B per k-step; [b0..b11]
        let b0123 = vld1q_f32(bp.add(0));  
        let b4567 = vld1q_f32(bp.add(4));  
        let b8911 = vld1q_f32(bp.add(8));  
        (b0123, b4567, b8911)
    }
}

#[inline(always)]
fn store_col8(colp: *mut f32, v0123: float32x4_t, v4567: float32x4_t) {
    unsafe {
        vst1q_f32(colp.add(0), v0123); 
        vst1q_f32(colp.add(4), v4567); 
    }
}

#[inline(always)]
fn load_col8(colp: *const f32) -> (float32x4_t, float32x4_t) {
    unsafe { (vld1q_f32(colp.add(0)), vld1q_f32(colp.add(4))) }
}

#[inline(always)]
fn kstep_accumulate_laneq(
    acc0123 : &mut [float32x4_t; NR],
    acc4567 : &mut [float32x4_t; NR],
    a0123   : float32x4_t,
    a4567   : float32x4_t,
    b0123   : float32x4_t,
    b4567   : float32x4_t,
    b8911   : float32x4_t,
) {
    unsafe {
        // cols 0..3 from lanes of b0123
        acc0123[0] = vfmaq_laneq_f32(acc0123[0], a0123, b0123, 0);
        acc4567[0] = vfmaq_laneq_f32(acc4567[0], a4567, b0123, 0);

        acc0123[1] = vfmaq_laneq_f32(acc0123[1], a0123, b0123, 1);
        acc4567[1] = vfmaq_laneq_f32(acc4567[1], a4567, b0123, 1);

        acc0123[2] = vfmaq_laneq_f32(acc0123[2], a0123, b0123, 2);
        acc4567[2] = vfmaq_laneq_f32(acc4567[2], a4567, b0123, 2);

        acc0123[3] = vfmaq_laneq_f32(acc0123[3], a0123, b0123, 3);
        acc4567[3] = vfmaq_laneq_f32(acc4567[3], a4567, b0123, 3);

        // cols 4..7 from lanes of b4567
        acc0123[4] = vfmaq_laneq_f32(acc0123[4], a0123, b4567, 0);
        acc4567[4] = vfmaq_laneq_f32(acc4567[4], a4567, b4567, 0);

        acc0123[5] = vfmaq_laneq_f32(acc0123[5], a0123, b4567, 1);
        acc4567[5] = vfmaq_laneq_f32(acc4567[5], a4567, b4567, 1);

        acc0123[6] = vfmaq_laneq_f32(acc0123[6], a0123, b4567, 2);
        acc4567[6] = vfmaq_laneq_f32(acc4567[6], a4567, b4567, 2);

        acc0123[7] = vfmaq_laneq_f32(acc0123[7], a0123, b4567, 3);
        acc4567[7] = vfmaq_laneq_f32(acc4567[7], a4567, b4567, 3);

        // cols 8..11 from lanes of b8911
        acc0123[8] = vfmaq_laneq_f32(acc0123[8], a0123, b8911, 0);
        acc4567[8] = vfmaq_laneq_f32(acc4567[8], a4567, b8911, 0);

        acc0123[9] = vfmaq_laneq_f32(acc0123[9], a0123, b8911, 1);
        acc4567[9] = vfmaq_laneq_f32(acc4567[9], a4567, b8911, 1);

        acc0123[10] = vfmaq_laneq_f32(acc0123[10], a0123, b8911, 2);
        acc4567[10] = vfmaq_laneq_f32(acc4567[10], a4567, b8911, 2);

        acc0123[11] = vfmaq_laneq_f32(acc0123[11], a0123, b8911, 3);
        acc4567[11] = vfmaq_laneq_f32(acc4567[11], a4567, b8911, 3);
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f32_mrxnr_beta0(
    kc: usize,
    a: *const f32, // packed; each k-step is [a0..a7]
    b: *const f32, // packed; each k-step is [b0..b11]
    c: *mut f32,
    ldc: usize,
    alpha: f32,
) {
    unsafe {
        let mut acc0123: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc4567: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            // k
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc0123, 
                &mut acc4567, 
                a0123,
                a4567,
                b0123,
                b4567, 
                b8911
            );

            // k+1
            let (b0123n, b4567n, b8911n) = load_b_colquads(bp.add(NR));
            let (a0123n, a4567n) = load_a_rowquads(ap.add(MR));
            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567,
                a0123n,
                a4567n,
                b0123n,
                b4567n,
                b8911n,
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }

        if kc & 1 != 0 {
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc0123, 
                &mut acc4567,
                a0123,
                a4567,
                b0123,
                b4567,
                b8911
            );
        }

        // beta = 0; no read of C
        for j in 0..NR {
            let colp = c.add(j * ldc);

            let v0123 = vmulq_n_f32(acc0123[j], alpha);
            let v4567 = vmulq_n_f32(acc4567[j], alpha);

            store_col8(colp, v0123, v4567);
        }
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f32_mrxnr_beta1(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
) {
    unsafe {
        let mut acc0123: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc4567: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567, 
                a0123,
                a4567,
                b0123, 
                b4567, 
                b8911
            );

            let (b0123n, b4567n, b8911n) = load_b_colquads(bp.add(NR));
            let (a0123n, a4567n) = load_a_rowquads(ap.add(MR));
            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567,
                a0123n,
                a4567n,
                b0123n,
                b4567n,
                b8911n,
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }
        if kc & 1 != 0 {
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);
            kstep_accumulate_laneq(&mut acc0123, 
                &mut acc4567,
                a0123,
                a4567, 
                b0123, 
                b4567,
                b8911
            );
        }

        // beta = 1: C += alpha * acc
        let alphv = vdupq_n_f32(alpha);
        for j in 0..NR {
            let colp = c.add(j * ldc);
            let (mut c0123, mut c4567) = load_col8(colp);

            c0123 = vfmaq_laneq_f32(c0123, acc0123[j], alphv, 0);
            c4567 = vfmaq_laneq_f32(c4567, acc4567[j], alphv, 0);

            store_col8(colp, c0123, c4567);
        }
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn f32_mrxnr_betax(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    unsafe {
        let mut acc0123: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc4567: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);

            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567, 
                a0123, a4567,
                b0123,
                b4567,
                b8911
            );

            let (b0123n, b4567n, b8911n) = load_b_colquads(bp.add(NR));
            let (a0123n, a4567n) = load_a_rowquads(ap.add(MR));

            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567,
                a0123n,
                a4567n,
                b0123n,
                b4567n,
                b8911n,
            );

            ap = ap.add(2 * MR);
            bp = bp.add(2 * NR);
        }
        if kc & 1 != 0 {
            let (b0123, b4567, b8911) = load_b_colquads(bp);
            let (a0123, a4567) = load_a_rowquads(ap);

            kstep_accumulate_laneq(
                &mut acc0123,
                &mut acc4567,
                a0123,
                a4567, 
                b0123,
                b4567, 
                b8911
            );
        }

        // general beta: C := beta*C + alpha*acc
        let alphv = vdupq_n_f32(alpha);
        for j in 0..NR {
            let colp = c.add(j * ldc);
            let (mut c0123, mut c4567) = load_col8(colp);

            // cXY *= beta
            c0123 = vmulq_n_f32(c0123, beta);
            c4567 = vmulq_n_f32(c4567, beta);

            // cXY += alpha * accXY
            c0123 = vfmaq_laneq_f32(c0123, acc0123[j], alphv, 0);
            c4567 = vfmaq_laneq_f32(c4567, acc4567[j], alphv, 0);

            store_col8(colp, c0123, c4567);
        }
    }
}

