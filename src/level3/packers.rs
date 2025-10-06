pub const MR: usize = 6;
pub const NR: usize = 8; // locked; don't change 

#[inline(always)]
const fn round_up(x: usize, b: usize) -> usize {
    (x + b - 1) / b * b
}

// A side 

/// Pack one `MR x k` micro-panel from A; no padding.
/// A_base points to A[base_row + base_col*lda].
#[inline(always)]
pub fn pack_a_mrxk(
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

/// tail; 
/// pack `mr_tail x k` and zero-pad to `MR` per `k`-step.
#[inline(always)]
pub fn pack_a_mrxk_tail(
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

/// Pack an `mc x kc` A-block; pads the last partial MR.
#[inline(always)]
pub fn pack_a_block(
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

// B side 

/// Pack one `k x NR` micro-panel from B; no padding.
/// B_base points to B[base_row + base_col*ldb].
#[inline(always)]
pub fn pack_b_kxnr(
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

/// tail; 
/// pack `k x nr_tail` and zero-pad to NR per k-step. 
#[inline(always)]
pub fn pack_b_kxnr_tail(
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

/// Pack a `kc x nc` B-block; pads the last partial NR.
#[inline(always)]
pub fn pack_b_block(
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

