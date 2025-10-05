pub const MR: usize = 6;  
pub const NR: usize = 8;

#[inline(always)]
const fn round_up(
    x: usize, 
    b: usize
) -> usize { 
    (x + b - 1) / b * b 
}

#[inline(always)]
fn idx(
    base_row    : usize, 
    base_col    : usize, 
    r           : usize,
    c           : usize, 
    row_stride  : usize, 
    col_stride  : usize
) -> usize {
    (base_row + r) * row_stride + (base_col + c) * col_stride
}


// A side 

/// Packs one `MR x k` micro-panel from A without padding. 
#[inline(always)]
pub fn pack_mrxk_a(
    k: usize,
    a: &[f64],
    row_stride : usize,
    col_stride : usize,
    base_row   : usize,
    base_col   : usize,
    dst: &mut [f64],        
) {
    debug_assert!(dst.len() >= MR * k);

    let mut d = 0;
    for j in 0..k {
        if row_stride == 1 {
            let off = idx(base_row, base_col, 0, j, row_stride, col_stride);

            unsafe {
                let src = core::slice::from_raw_parts(a.as_ptr().add(off), MR);
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(d), MR);
            }
        } else {
            for i in 0..MR {
                let off = idx(base_row, base_col, i, j, row_stride, col_stride);

                unsafe { 
                    *dst.get_unchecked_mut(d + i) = *a.get_unchecked(off); 
                }
            }
        }
        d += MR;
    }
}

/// Pack an `mc, kc` block of A, padding tail rows up to MR. 
#[inline(always)]
pub fn pack_a(
    mc : usize,
    kc : usize,
    a  : &[f64],
    row_stride : usize,
    col_stride : usize,
    base_row : usize,
    base_col : usize,
    dst: &mut [f64],
) {
    let need = round_up(mc, MR) * kc;
    debug_assert!(dst.len() >= need);

    let mp = mc / MR;
    let mr_tail = mc % MR;

    // full MR blocks
    let mut d_off = 0;
    for blk in 0..mp {
        let br = base_row + blk * MR;

        pack_mrxk_a(
            kc,
            a, 
            row_stride,
            col_stride,
            br, 
            base_col, 
            &mut dst[d_off .. d_off + MR * kc]
        );

        d_off += MR * kc;
    }

    // tail rows; pad to MR
    if mr_tail > 0 {
        for j in 0..kc {
            // copy mr_tail rows
            for i in 0..mr_tail {
                let off = idx(base_row, base_col, mp*MR + i, j, row_stride, col_stride);
                unsafe { *dst.get_unchecked_mut(d_off + j*MR + i) = *a.get_unchecked(off); }
            }

            // zero-pad remaining
            for i in mr_tail..MR {
                unsafe { *dst.get_unchecked_mut(d_off + j*MR + i) = 0.0; }
            }
        }
    }
}

// B side

/// Pack one `k x NR` panel from B without padding. 
#[inline(always)]
pub fn pack_kxnr_b(
    k : usize,
    b : &[f64],
    row_stride : usize,
    col_stride : usize,
    base_row   : usize,
    base_col   : usize,
    dst: &mut [f64],         
) {
    debug_assert!(dst.len() >= k * NR);

    let mut d = 0;
    for i in 0..k {
        if col_stride == 1 {
            let off = idx(base_row, base_col, i, 0, row_stride, col_stride);

            unsafe {
                let src = core::slice::from_raw_parts(b.as_ptr().add(off), NR);
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(d), NR);
            }
        } else {
            for j in 0..NR {
                let off = idx(base_row, base_col, i, j, row_stride, col_stride);

                unsafe { *dst.get_unchecked_mut(d + j) = *b.get_unchecked(off); }
            }
        }
        d += NR;
    }
}

/// Pack a `kc x nc` block of B; padding tail cols up to NR
#[inline(always)]
pub fn pack_b(
    kc  : usize,
    nc  : usize,
    b   : &[f64],
    row_stride : usize,
    col_stride : usize,
    base_row   : usize,
    base_col   : usize,
    dst: &mut [f64],
) {
    let need = kc * round_up(nc, NR);
    debug_assert!(dst.len() >= need);

    let np = nc / NR;
    let nr_tail = nc % NR;

    // full NR blocks
    let mut d_off = 0;
    for blk in 0..np {
        let bc = base_col + blk * NR;

        pack_kxnr_b(
            kc, 
            b,
            row_stride,
            col_stride, 
            base_row, bc,
            &mut dst[d_off .. d_off + kc * NR]
        );

        d_off += kc * NR;
    }

    // tail cols; pad to NR
    if nr_tail > 0 {
        for i in 0..kc {
            // copy tail
            for j in 0..nr_tail {
                let off = idx(base_row, base_col, i, np*NR + j, row_stride, col_stride);
                unsafe { *dst.get_unchecked_mut(d_off + j) = *b.get_unchecked(off); }
            }

            // zero-pad remaining
            for j in nr_tail..NR {
                unsafe { *dst.get_unchecked_mut(d_off + j) = 0.0; }
            }

            d_off += NR;
        }
    }
}

