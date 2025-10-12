pub const MR: usize = 4; 
pub const NR: usize = 8; 

#[inline(always)] 
const fn round_up(x: usize, b: usize) -> usize { 
    (x + b - 1) / b * b 
}

#[inline(always)] 
pub(crate) fn a_buf_len(mc: usize, kc: usize) -> usize { 
    2 * round_up(mc, MR) * kc 
}

#[inline(always)] 
pub(crate) fn b_buf_len(kc: usize, nc: usize) -> usize { 
    2 * round_up(nc, NR) * kc 
}

// A side 

/// pack one `MR x k` micro-panel from A 
/// a_base points to &A[base_row + base_col * lda]; complex idxs 
#[inline(always)] 
fn pack_a_mrxk( 
    k      : usize, 
    a_base : *const f32, 
    lda_c  : usize, 
    dst    : *mut f32 
) { 
    unsafe { 
        let lda_f = 2 * lda_c; 
        let mut ap = a_base; 
        let mut dp = dst; 

        for _ in 0..k { 
            let dp_re = dp; 
            let dp_im = dp.add(MR); 

            for r in 0..MR { 
                let pr = ap.add(2 * r); 
                *dp_re.add(r) = *pr;        
                *dp_im.add(r) = *pr.add(1); 
            }

            ap = ap.add(lda_f); 
            dp = dp.add(2 * MR);
        }
    } 
}

// transpose 
#[inline(always)]
fn pack_at_mrxk(
    k: usize,
    a_base: *const f32,
    lda_c: usize,
    dst: *mut f32,
) {
    unsafe {
        let lda_f = 2 * lda_c;
        let mut ap = a_base;
        let mut dp = dst;

        for _ in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(MR);

            for t in 0..MR {
                let pr = ap.add(t * lda_f);
                *dp_re.add(t) = *pr;
                *dp_im.add(t) = *pr.add(1);
            }

            ap = ap.add(2);        
            dp = dp.add(2 * MR);   
        }
    }
}

// conjugate transpose 
#[inline(always)]
fn pack_act_mrxk(
    k: usize,
    a_base: *const f32,
    lda_c: usize,
    dst: *mut f32,
) {
    unsafe {
        let lda_f = 2 * lda_c;
        let mut ap = a_base;
        let mut dp = dst;

        for _ in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(MR);

            for t in 0..MR {
                let pr = ap.add(t * lda_f);
                *dp_re.add(t) = *pr;            
                *dp_im.add(t) = -*pr.add(1); // -im
            }

            ap = ap.add(2);
            dp = dp.add(2 * MR);
        }
    }
}

// tails; < MR rows
// zero-pad remaining rows in both stripes
#[inline(always)]
fn pack_a_mrxk_tail(
    k: usize,
    a_base: *const f32,
    lda_c: usize,
    mr_tail: usize,
    dst: *mut f32, // len >= 2 * MR * k
) {
    unsafe {
        let lda_f = 2 * lda_c;
        let mut ap = a_base;
        let mut dp = dst;

        for _ in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(MR);

            // copy current rows
            for r in 0..mr_tail {
                let pr = ap.add(2 * r);
                *dp_re.add(r) = *pr;
                *dp_im.add(r) = *pr.add(1);
            }

            // zero-pad remainder
            core::ptr::write_bytes(
                dp_re.add(mr_tail),
                0, 
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes
                (dp_im.add(mr_tail),
                0, 
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );

            ap = ap.add(lda_f);
            dp = dp.add(2 * MR);
        }
    }
}

#[inline(always)]
fn pack_at_mrxk_tail(
    k       : usize,
    a_base  : *const f32,
    lda_c   : usize,
    mr_tail : usize,
    dst     : *mut f32,
) {
    unsafe {
        let lda_f = 2 * lda_c;
        let mut ap = a_base;
        let mut dp = dst;

        for _ in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(MR);

            for t in 0..mr_tail {
                let pr = ap.add(t * lda_f);
                *dp_re.add(t) = *pr;
                *dp_im.add(t) = *pr.add(1);
            }

            core::ptr::write_bytes(
                dp_re.add(mr_tail),
                0, 
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes(
                dp_im.add(mr_tail),
                0, 
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );

            ap = ap.add(2);
            dp = dp.add(2 * MR);
        }
    }
}

#[inline(always)]
fn pack_act_mrxk_tail(
    k: usize,
    a_base  : *const f32,
    lda_c   : usize,
    mr_tail : usize,
    dst     : *mut f32,
) {
    unsafe {
        let lda_f = 2 * lda_c;
        let mut ap = a_base;
        let mut dp = dst;

        for _ in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(MR);

            for t in 0..mr_tail {
                let pr = ap.add(t * lda_f);
                *dp_re.add(t) = *pr;
                *dp_im.add(t) = -*pr.add(1);
            }

            core::ptr::write_bytes(
                dp_re.add(mr_tail),
                0,
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes(
                dp_im.add(mr_tail),
                0, 
                (MR - mr_tail) * core::mem::size_of::<f32>()
            );

            ap = ap.add(2);
            dp = dp.add(2 * MR);
        }
    }
}


#[inline(always)]
pub(crate) fn pack_a_block(
    mc           : usize,
    kc           : usize,
    a_block_base : *const f32, 
    lda_c        : usize,             
    dst          : *mut f32,            
) {
    unsafe {
        let mp = mc / MR;
        let mr_tail = mc % MR;

        let mut dp = dst;
        let mut abp = a_block_base;

        // full MR cols 
        for _ in 0..mp {
            pack_a_mrxk(kc, abp, lda_c, dp);
            dp  = dp.add(2 * MR * kc);
            abp = abp.add(MR * 2); 
        }

        // tail 
        if mr_tail != 0 {
            pack_a_mrxk_tail(kc, abp, lda_c, mr_tail, dp);
        }
    }
}

#[inline(always)]
pub(crate) fn pack_a_block_t(
    mc           : usize,
    kc           : usize,
    a_block_base : *const f32,
    lda_c        : usize,
    dst          : *mut f32,
) {
    unsafe {
        let mp = mc / MR;
        let mr_tail = mc % MR;

        let mut dp = dst;
        let mut abp = a_block_base;

        // full MR cols 
        for _ in 0..mp {
            pack_at_mrxk(kc, abp, lda_c, dp);
            dp  = dp.add(2 * MR * kc);
            abp = abp.add(MR * 2 * lda_c); 
        }
 
        // tail 
        if mr_tail != 0 {
            pack_at_mrxk_tail(kc, abp, lda_c, mr_tail, dp);
        }
    }
}

#[inline(always)]
pub(crate) fn pack_a_block_ct(
    mc           : usize,
    kc           : usize,
    a_block_base : *const f32,
    lda_c        : usize,
    dst          : *mut f32,
) {
    unsafe {
        let mp = mc / MR;
        let mr_tail = mc % MR;

        let mut dp = dst;
        let mut abp = a_block_base;

        for _ in 0..mp {
            pack_act_mrxk(kc, abp, lda_c, dp);
            dp = dp.add(2 * MR * kc);
            abp = abp.add(MR * 2 * lda_c); // move by MR columns (AoS)
        }

        if mr_tail != 0 {
            pack_act_mrxk_tail(kc, abp, lda_c, mr_tail, dp);
        }
    }
}

// B  side 

// notranspose 
#[inline(always)]
fn pack_b_kxnr(
    k      : usize,
    b_base : *const f32, 
    ldb_c  : usize,       
    dst    : *mut f32,      
) {
    unsafe {
        let ldb_f = 2 * ldb_c;
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..NR {
                let pr = b_base.add(2 * (i + j * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = *pr.add(1);
            }

            dp = dp.add(2 * NR);
        }
    }
}

// transpose 
#[inline(always)]
fn pack_bt_kxnr(
    k      : usize,
    b_base : *const f32,
    ldb_c  : usize,
    dst    : *mut f32,
) {
    unsafe {
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..NR {

                // transpose; element is at (j, i) 
                let pr = b_base.add(2 * (j + i * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = *pr.add(1);
            }

            dp = dp.add(2 * NR);
        }
    }
}

// conjugate-transpose 
#[inline(always)]
fn pack_bct_kxnr(
    k      : usize,
    b_base : *const f32,
    ldb_c  : usize,
    dst    : *mut f32,
) {
    unsafe {
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..NR {
                let pr = b_base.add(2 * (j + i * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = -*pr.add(1);
            }

            dp = dp.add(2 * NR);
        }
    }
}

// < NR columns
// zero-pad remaining entries in both stripes.
#[inline(always)]
fn pack_b_kxnr_tail(
    k       : usize,
    b_base  : *const f32,
    ldb_c   : usize,
    nr_tail : usize,
    dst     : *mut f32,
) {
    unsafe {
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..nr_tail {
                let pr = b_base.add(2 * (i + j * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = *pr.add(1);
            }

            // zero pad remainder to NR in both stripes
            core::ptr::write_bytes(
                dp_re.add(nr_tail), 
                0, 
                (NR - nr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes(
                dp_im.add(nr_tail), 
                0, 
                (NR - nr_tail) * core::mem::size_of::<f32>()
                );

            dp = dp.add(2 * NR);
        }
    }
}

#[inline(always)]
fn pack_bt_kxnr_tail(
    k       : usize,
    b_base  : *const f32,
    ldb_c   : usize,
    nr_tail : usize,
    dst     : *mut f32,
) {
    unsafe {
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..nr_tail {
                let pr = b_base.add(2 * (j + i * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = *pr.add(1);
            }

            core::ptr::write_bytes(
                dp_re.add(nr_tail), 
                0, 
                (NR - nr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes(
                dp_im.add(nr_tail), 
                0, 
                (NR - nr_tail) * core::mem::size_of::<f32>()
            );

            dp = dp.add(2 * NR);
        }
    }
}

#[inline(always)]
fn pack_bct_kxnr_tail(
    k       : usize,
    b_base  : *const f32,
    ldb_c   : usize,
    nr_tail : usize,
    dst     : *mut f32,
) {
    unsafe {
        let mut dp = dst;

        for i in 0..k {
            let dp_re = dp;
            let dp_im = dp.add(NR);

            for j in 0..nr_tail {
                let pr = b_base.add(2 * (j + i * ldb_c));
                *dp_re.add(j) = *pr;
                *dp_im.add(j) = -*pr.add(1);
            }

            core::ptr::write_bytes(
                dp_re.add(nr_tail),
                0,
                (NR - nr_tail) * core::mem::size_of::<f32>()
            );
            core::ptr::write_bytes(
                dp_im.add(nr_tail),
                0, 
                (NR - nr_tail) * core::mem::size_of::<f32>()
            );

            dp = dp.add(2 * NR);
        }
    }
}


#[inline(always)]
pub(crate) fn pack_b_block(
    kc           : usize,
    nc           : usize,
    b_block_base : *const f32, 
    ldb_c        : usize,             
    dst          : *mut f32,            
) {
    unsafe {
        let np = nc / NR;
        let nr_tail = nc % NR;

        let mut dp = dst;
        let mut bbp = b_block_base;

        for _ in 0..np {
            pack_b_kxnr(kc, bbp, ldb_c, dp);
            dp = dp.add(2 * kc * NR);
            bbp = bbp.add(2 * NR * ldb_c); 
        }

        if nr_tail != 0 {
            pack_b_kxnr_tail(kc, bbp, ldb_c, nr_tail, dp);
        }
    }
}

#[inline(always)]
pub(crate) fn pack_b_block_t(
    kc: usize,
    nc: usize,
    b_block_base: *const f32,
    ldb_c: usize,
    dst: *mut f32,
) {
    unsafe {
        let np = nc / NR;
        let nr_tail = nc % NR;

        let mut dp = dst;
        let mut bbp = b_block_base;

        for _ in 0..np {
            pack_bt_kxnr(kc, bbp, ldb_c, dp);
            dp = dp.add(2 * kc * NR);
            bbp = bbp.add(2 * NR); 
        }

        if nr_tail != 0 {
            pack_bt_kxnr_tail(kc, bbp, ldb_c, nr_tail, dp);
        }
    }
}

#[inline(always)]
pub(crate) fn pack_b_block_ct(
    kc: usize,
    nc: usize,
    b_block_base: *const f32,
    ldb_c: usize,
    dst: *mut f32,
) {
    unsafe {
        let np = nc / NR;
        let nr_tail = nc % NR;

        let mut dp = dst;
        let mut bbp = b_block_base;

        for _ in 0..np {
            pack_bct_kxnr(kc, bbp, ldb_c, dp);
            dp  = dp.add(2 * kc * NR);
            bbp = bbp.add(2 * NR); 
        }

        if nr_tail != 0 {
            pack_bct_kxnr_tail(kc, bbp, ldb_c, nr_tail, dp);
        }
    }
}
