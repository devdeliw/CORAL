pub const MR: usize = 6; // fixed; don't change
pub const NR: usize = 8; // fixed; don't change 

#[inline(always)]
const fn round_up(x: usize, b: usize) -> usize {
    (x + b - 1) / b * b
}

#[inline(always)] 
pub(crate) fn a_buf_len(mc: usize, kc: usize) -> usize { 
    round_up(mc, MR) * kc 
}

#[inline(always)] 
pub(crate) fn b_buf_len(kc: usize, nc: usize) -> usize { 
    kc * round_up(nc, NR) 
}

// A side 

/// pack one `MR x k` micro-panel from A; no padding.
/// A_base points to A[base_row + base_col*lda].
#[inline(always)]
fn pack_a_mrxk(
    k       : usize,
    a_base  : *const f64, // &A[base_row + base_col*lda]
    lda     : usize,
    dst     : *mut f64,      
) {
    unsafe { 
        let mut ap = a_base;
        let mut dp = dst;
        for _ in 0..k {
            core::ptr::copy_nonoverlapping(ap, dp, MR);

            ap = ap.add(lda);
            dp = dp.add(MR);
        }
    }
}

#[inline(always)] 
fn pack_at_mrxk(
    k      : usize, 
    a_base : *const f64, 
    lda    : usize, 
    dst    : *mut f64, 
) { 
    unsafe { 
        let mut ap = a_base; 
        let mut dp = dst; 

        for _ in 0..k { 

            for t in 0..MR { 
                *dp.add(t) = *ap.add(t * lda); 
            }

            ap = ap.add(1); 
            dp = dp.add(MR); 
        }
    }
}

/// tail; 
/// pack `mr_tail x k` and zero-pad to `MR` per `k`-step.
#[inline(always)]
fn pack_a_mrxk_tail(
    k       : usize,
    a_base  : *const f64,
    lda     : usize,
    mr_tail : usize,    // 1..MR-1
    dst     : *mut f64, // len >= MR*k
) {
    unsafe { 
        let mut ap = a_base;
        let mut dp = dst;
        for _ in 0..k {
            core::ptr::copy_nonoverlapping(ap, dp, mr_tail);

            // zero pad 
            core::ptr::write_bytes(dp.add(mr_tail), 0, MR - mr_tail); 
            ap = ap.add(lda);
            dp = dp.add(MR);
        }
    }
}

fn pack_at_mrxk_tail(
    k       : usize, 
    a_base  : *const f64, 
    lda     : usize, 
    mr_tail : usize, 
    dst     : *mut f64,
) { 
    unsafe { 
        let mut ap = a_base; 
        let mut dp = dst; 

        for _ in 0..k { 

            for t in 0..mr_tail { 
                *dp.add(t) = *ap.add(t * lda); 
            }

            // zero-pad remainder 
            core::ptr::write_bytes(dp.add(mr_tail), 0, MR - mr_tail);

            ap = ap.add(1); 
            dp = dp.add(MR); 
        }       
    }
}

/// pack an `mc x kc` A-block; pads the last partial MR.
#[inline(always)]
pub(crate) fn pack_a_block(
    mc          : usize,
    kc          : usize,
    a_block_base: *const f64, // &A[base_row + base_col*lda]
    lda         : usize,
    dst         : *mut f64,          
) {
    unsafe { 
        let mp = mc / MR;
        let mr_tail = mc % MR;

        let mut dp = dst;
        let mut abp = a_block_base;

        // full MR
        for _ in 0..mp {
            pack_a_mrxk(kc, abp, lda, dp);
            dp  = dp.add(MR * kc);
            abp = abp.add(MR); 
        }

        // tail MR
        if mr_tail != 0 {
            pack_a_mrxk_tail(kc, abp, lda, mr_tail, dp);
        }
    } 
}

#[inline(always)]
pub(crate) fn pack_a_block_t( 
    mc           : usize, 
    kc           : usize, 
    a_block_base : *const f64, 
    lda          : usize, 
    dst          : *mut f64, 
) { 
    unsafe { 
        let mp      = mc / MR; 
        let mr_tail = mc % MR; 

        let mut dp  = dst; 
        let mut abp = a_block_base; 

        // full MR cols 
        for _ in 0..mp { 
            pack_at_mrxk(kc, abp, lda, dp);

            dp  = dp.add(MR * kc); 
            abp = abp.add(MR * lda); 
        }

        // tail cols 
        if mr_tail != 0 { 
            pack_at_mrxk_tail(kc, abp, lda, mr_tail, dp);
        }
    }
}

// B side 

/// pack one `k x NR` micro-panel from B; no padding.
/// B_base points to B[base_row + base_col*ldb].
#[inline(always)]
fn pack_b_kxnr(
    k:       usize,
    b_base  : *const f64,    // &B[base_row + base_col*ldb]
    ldb     : usize,
    dst     : *mut f64,      // len >= k*NR
) {
    unsafe { 
        let mut dp = dst;
        for i in 0..k {
            let rp = b_base.add(i);
            *dp.add(0) = *rp.add(0 * ldb);
            *dp.add(1) = *rp.add(1 * ldb);
            *dp.add(2) = *rp.add(2 * ldb);
            *dp.add(3) = *rp.add(3 * ldb);
            *dp.add(4) = *rp.add(4 * ldb);
            *dp.add(5) = *rp.add(5 * ldb);
            *dp.add(6) = *rp.add(6 * ldb);
            *dp.add(7) = *rp.add(7 * ldb);
            dp = dp.add(NR);
        }
    }
}

#[inline(always)] 
fn pack_bt_kxnr( 
    k      : usize, 
    b_base : *const f64, 
    ldb    : usize, 
    dst    : *mut f64, 
) { 
    unsafe { 
        let mut dp = dst; 

        for i in 0..k { 
            let src = b_base.add(i * ldb); 

            core::ptr::copy_nonoverlapping(src, dp, NR);

            dp = dp.add(NR) 
        }
    }
}

/// tail; 
/// pack `k x nr_tail` and zero-pad to NR per k-step. 
#[inline(always)]
fn pack_b_kxnr_tail(
    k       : usize,
    b_base  : *const f64,
    ldb     : usize,
    nr_tail : usize,     // 1..NR-1
    dst     : *mut f64,  // len >= k*NR
) {
    unsafe { 
        let mut dp = dst;
        for i in 0..k {
            let rp = b_base.add(i);
            for j in 0..nr_tail {
                *dp.add(j) = *rp.add(j * ldb);
            }

            // zero pad
            core::ptr::write_bytes(dp.add(nr_tail), 0, NR - nr_tail);
            dp = dp.add(NR);
        }
    }
}

#[inline(always)] 
fn pack_bt_kxnr_tail( 
    k       : usize, 
    b_base  : *const f64, 
    ldb     : usize, 
    nr_tail : usize, 
    dst     : *mut f64
) { 
    unsafe { 
        let mut dp = dst; 
        
        for i in 0..k { 
            let src = b_base.add(i * ldb); 

            core::ptr::copy_nonoverlapping(src, dp, nr_tail);

            // zero pad remainder to NR 
            core::ptr::write_bytes(dp.add(nr_tail), 0, NR - nr_tail);

            dp = dp.add(NR);
        }
    }
}

/// pack a `kc x nc` B-block; pads the last partial NR.
#[inline(always)]
pub(crate) fn pack_b_block(
    kc           : usize, 
    nc           : usize,
    b_block_base : *const f64, // &B[base_row + base_col*ldb]
    ldb          : usize,
    dst          : *mut f64,            
) {
    unsafe { 
        let np = nc / NR;
        let nr_tail = nc % NR;

        let mut dp  = dst;
        let mut bbp = b_block_base;

        // full NR 
        for _ in 0..np {
            pack_b_kxnr(kc, bbp, ldb, dp);
            dp  = dp.add(kc * NR);
            bbp = bbp.add(NR * ldb); 
        }

        // tail NR
        if nr_tail != 0 {
            pack_b_kxnr_tail(kc, bbp, ldb, nr_tail, dp);
        }
    }
}

#[inline(always)] 
pub(crate) fn pack_b_block_t( 
    kc           : usize, 
    nc           : usize, 
    b_block_base : *const f64, 
    ldb          : usize, 
    dst          : *mut f64, 
) { 
    unsafe { 
        let np      = nc / NR; 
        let nr_tail = nc % NR; 

        let mut dp  = dst; 
        let mut bbp = b_block_base; 

        // full NR cols 
        for _ in 0..np { 
            pack_bt_kxnr(kc, bbp, ldb, dp);

            dp  = dp.add(kc * NR); 
            bbp = bbp.add(NR); 
        }

        // tail NR 
        if nr_tail != 0 { 
            pack_bt_kxnr_tail(kc, bbp, ldb, nr_tail, dp);
        }
    }   
}

