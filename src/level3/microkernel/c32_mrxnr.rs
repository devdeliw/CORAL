use crate::level3::c32_packers::{MR, NR};
use core::arch::aarch64::{
    float32x4_t, float32x4x2_t, vdupq_n_f32, vld1q_f32, vld2q_f32, vfmaq_laneq_f32,
    vfmsq_laneq_f32, vmulq_n_f32, vst2q_f32,
};

#[repr(C)] // HFA 
#[derive(Copy, Clone)]
pub(crate) struct Complex32 {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

#[inline(always)]
fn load_a_rowquads(ap: *const f32) -> (float32x4_t, float32x4_t) {
    unsafe {
        // A per k-step; MR is 4
        // [a_re(0..MR), a_im(0..MR)]
        let a_re = vld1q_f32(ap.add(0));
        let a_im = vld1q_f32(ap.add(MR));
        (a_re, a_im)
    }
}

#[inline(always)]
fn load_b_colquads(
    bp: *const f32,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    unsafe {
        // B per k-step; NR is 8
        // [b_re(0..NR), b_im(0..NR)] 
        let br0123 = vld1q_f32(bp.add(0));
        let br4567 = vld1q_f32(bp.add(4));
        let bi0123 = vld1q_f32(bp.add(NR + 0));
        let bi4567 = vld1q_f32(bp.add(NR + 4));
        (br0123, br4567, bi0123, bi4567)
    }
}


#[inline(always)]
fn store_col4(colp: *mut f32, v_re: float32x4_t, v_im: float32x4_t, ldc: usize) {
    let _ = ldc; // complex elems  
    unsafe {
        // [re0, im0, re1, im1, re2, im2, re3,im3]
        let pair = float32x4x2_t(v_re, v_im);

        vst2q_f32(colp, pair);
    }
}

#[inline(always)]
fn load_col4(colp: *const f32, ldc: usize) -> (float32x4_t, float32x4_t) {
    let _ = ldc;
    unsafe {
        let pair = vld2q_f32(colp);

        (pair.0, pair.1)
    }
}

#[inline(always)]
fn kstep_accumulate_laneq(
    acc_re : &mut [float32x4_t; NR],
    acc_im : &mut [float32x4_t; NR],
    a_re   : float32x4_t,
    a_im   : float32x4_t,
    br0123 : float32x4_t,
    br4567 : float32x4_t,
    bi0123 : float32x4_t,
    bi4567 : float32x4_t,
) {
    unsafe {
        // cols 0..3
        acc_re[0] = vfmaq_laneq_f32(acc_re[0], a_re, br0123, 0);
        acc_re[0] = vfmsq_laneq_f32(acc_re[0], a_im, bi0123, 0);
        acc_im[0] = vfmaq_laneq_f32(acc_im[0], a_re, bi0123, 0);
        acc_im[0] = vfmaq_laneq_f32(acc_im[0], a_im, br0123, 0);

        acc_re[1] = vfmaq_laneq_f32(acc_re[1], a_re, br0123, 1);
        acc_re[1] = vfmsq_laneq_f32(acc_re[1], a_im, bi0123, 1);
        acc_im[1] = vfmaq_laneq_f32(acc_im[1], a_re, bi0123, 1);
        acc_im[1] = vfmaq_laneq_f32(acc_im[1], a_im, br0123, 1);

        acc_re[2] = vfmaq_laneq_f32(acc_re[2], a_re, br0123, 2);
        acc_re[2] = vfmsq_laneq_f32(acc_re[2], a_im, bi0123, 2);
        acc_im[2] = vfmaq_laneq_f32(acc_im[2], a_re, bi0123, 2);
        acc_im[2] = vfmaq_laneq_f32(acc_im[2], a_im, br0123, 2);

        acc_re[3] = vfmaq_laneq_f32(acc_re[3], a_re, br0123, 3);
        acc_re[3] = vfmsq_laneq_f32(acc_re[3], a_im, bi0123, 3);
        acc_im[3] = vfmaq_laneq_f32(acc_im[3], a_re, bi0123, 3);
        acc_im[3] = vfmaq_laneq_f32(acc_im[3], a_im, br0123, 3);

        // cols 4..7
        acc_re[4] = vfmaq_laneq_f32(acc_re[4], a_re, br4567, 0);
        acc_re[4] = vfmsq_laneq_f32(acc_re[4], a_im, bi4567, 0);
        acc_im[4] = vfmaq_laneq_f32(acc_im[4], a_re, bi4567, 0);
        acc_im[4] = vfmaq_laneq_f32(acc_im[4], a_im, br4567, 0);

        acc_re[5] = vfmaq_laneq_f32(acc_re[5], a_re, br4567, 1);
        acc_re[5] = vfmsq_laneq_f32(acc_re[5], a_im, bi4567, 1);
        acc_im[5] = vfmaq_laneq_f32(acc_im[5], a_re, bi4567, 1);
        acc_im[5] = vfmaq_laneq_f32(acc_im[5], a_im, br4567, 1);

        acc_re[6] = vfmaq_laneq_f32(acc_re[6], a_re, br4567, 2);
        acc_re[6] = vfmsq_laneq_f32(acc_re[6], a_im, bi4567, 2);
        acc_im[6] = vfmaq_laneq_f32(acc_im[6], a_re, bi4567, 2);
        acc_im[6] = vfmaq_laneq_f32(acc_im[6], a_im, br4567, 2);

        acc_re[7] = vfmaq_laneq_f32(acc_re[7], a_re, br4567, 3);
        acc_re[7] = vfmsq_laneq_f32(acc_re[7], a_im, bi4567, 3);
        acc_im[7] = vfmaq_laneq_f32(acc_im[7], a_re, bi4567, 3);
        acc_im[7] = vfmaq_laneq_f32(acc_im[7], a_im, br4567, 3);
    }
}

#[inline(always)]
fn apply_alpha(
    acc_re : float32x4_t,
    acc_im : float32x4_t,
    alpha  : Complex32,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        // (acc_re*ar - acc_im*ai,  acc_re*ai + acc_im*ar)
        let ar = alpha.re;
        let ai = alpha.im;

        let mut out_re = vmulq_n_f32(acc_re, ar);
        out_re = vfmsq_laneq_f32(out_re, acc_im, vdupq_n_f32(ai), 0);
        let mut out_im = vmulq_n_f32(acc_im, ar);
        out_im = vfmaq_laneq_f32(out_im, acc_re, vdupq_n_f32(ai), 0);

        (out_re, out_im)
    }
}

#[inline(always)]
fn apply_beta(
    c_re : float32x4_t,
    c_im : float32x4_t,
    beta : Complex32,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        // (c_re*br - c_im*bi,  c_re*bi + c_im*br)
        let br = beta.re;
        let bi = beta.im;

        let mut out_re = vmulq_n_f32(c_re, br);
        out_re = vfmsq_laneq_f32(out_re, c_im, vdupq_n_f32(bi), 0);
        let mut out_im = vmulq_n_f32(c_im, br);
        out_im = vfmaq_laneq_f32(out_im, c_re, vdupq_n_f32(bi), 0);

        (out_re, out_im)
    }
}

// beta = 0 

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c32_mrxnr_beta0(
    kc    : usize,
    a     : *const f32, // packed A; per k-step [MR re | MR im]
    b     : *const f32, // packed B; per k-step [NR re | NR im]
    c     : *mut f32,   
    ldc   : usize,      // complex elems 
    alpha : Complex32,
) {
    unsafe {
        let mut acc_re: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc_im: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            // k
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_re, a_im, 
                br0123, 
                br4567,
                bi0123, 
                bi4567,
            );

            // k+1
            let (brn0123, brn4567, bin0123, bin4567) = load_b_colquads(bp.add(2 * NR));
            let (a_ren, a_imn) = load_a_rowquads(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_ren, 
                a_imn,
                brn0123,
                brn4567,
                bin0123, 
                bin4567,
            );

            ap = ap.add(4 * MR); 
            bp = bp.add(4 * NR); 
        }

        if kc & 1 != 0 {
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_re,
                a_im, 
                br0123, 
                br4567,
                bi0123, 
                bi4567,
            );
        }

        for j in 0..NR {
            let colp = c.add(2 * j * ldc);
            let (out_re, out_im) = apply_alpha(acc_re[j], acc_im[j], alpha);

            store_col4(colp, out_re, out_im, ldc);
        }
    }
}

// beta = 1 

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c32_mrxnr_beta1(
    kc    : usize,
    a     : *const f32,
    b     : *const f32,
    c     : *mut f32,
    ldc   : usize,
    alpha : Complex32,
) {
    unsafe {
        let mut acc_re: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc_im: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_re,
                a_im,
                br0123,
                br4567,
                bi0123, 
                bi4567,
            );

            let (brn0123, brn4567, bin0123, bin4567) = load_b_colquads(bp.add(2 * NR));
            let (a_ren, a_imn) = load_a_rowquads(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_ren,
                a_imn,
                brn0123, 
                brn4567, 
                bin0123, 
                bin4567,
            );

            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }
        if kc & 1 != 0 {
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_re, 
                a_im, 
                br0123,
                br4567,
                bi0123,
                bi4567,
            );
        }

        let one = vdupq_n_f32(1.0);
        for j in 0..NR {
            let colp = c.add(2 * j * ldc);
            let (mut c_re, mut c_im) = load_col4(colp, ldc);
            let (add_re, add_im) = apply_alpha(acc_re[j], acc_im[j], alpha); 

            c_re = vfmaq_laneq_f32(c_re, add_re, one, 0);
            c_im = vfmaq_laneq_f32(c_im, add_im, one, 0);

            store_col4(colp, c_re, c_im, ldc);
        }
    }
}

// general beta 

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c32_mrxnr_betax(
    kc    : usize,
    a     : *const f32,
    b     : *const f32,
    c     : *mut f32,
    ldc   : usize,
    alpha : Complex32,
    beta  : Complex32,
) {
    unsafe {
        let mut acc_re: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];
        let mut acc_im: [float32x4_t; NR] = [vdupq_n_f32(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im,
                a_re, 
                a_im,
                br0123,
                br4567,
                bi0123, 
                bi4567,
            );

            let (brn0123, brn4567, bin0123, bin4567) = load_b_colquads(bp.add(2 * NR));
            let (a_ren, a_imn) = load_a_rowquads(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_ren, 
                a_imn,
                brn0123,
                brn4567,
                bin0123, 
                bin4567,
            );

            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }
        if kc & 1 != 0 {
            let (br0123, br4567, bi0123, bi4567) = load_b_colquads(bp);
            let (a_re, a_im) = load_a_rowquads(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im,
                a_re, 
                a_im, 
                br0123,
                br4567, 
                bi0123,
                bi4567,
            );
        }

        let one = vdupq_n_f32(1.0);
        for j in 0..NR {
            let colp = c.add(2 * j * ldc);

            let (c_re0, c_im0) = load_col4(colp, ldc);
            let (mut c_re, mut c_im) = apply_beta(c_re0, c_im0, beta);
            let (add_re, add_im) = apply_alpha(acc_re[j], acc_im[j], alpha);

            c_re = vfmaq_laneq_f32(c_re, add_re, one, 0);
            c_im = vfmaq_laneq_f32(c_im, add_im, one, 0);

            store_col4(colp, c_re, c_im, ldc);
        }
    }
}


