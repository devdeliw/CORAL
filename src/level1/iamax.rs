use core::arch::aarch64::{
    vld1q_f32, vabsq_f32, vaddq_f32, vceqq_f32, vmaxvq_f32, vbslq_f32, vrev64q_f32,
    vld1q_f64, vabsq_f64, vaddq_f64, vceqq_f64, vmaxvq_f64, vbslq_f64, vextq_f64,
    vld1q_u32, vdupq_n_u32, vaddq_u32, vbslq_u32, vminvq_u32,
    vld1q_u64, vdupq_n_u64, vaddq_u64, vbslq_u64,
    vdupq_n_f32, vdupq_n_f64,
    vgetq_lane_u64, 
};

#[inline]
pub fn isamax(n: usize, x: &[f32], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }
    debug_assert!(x.len() >= 1 + (n - 1).saturating_mul(incx as usize));

    unsafe {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i = 0usize;
            let iota   = vld1q_u32([0u32, 1, 2, 3].as_ptr());
            let allmax = vdupq_n_u32(u32::MAX);
            let ninf   = vdupq_n_f32(f32::NEG_INFINITY);

            while i + 16 <= n {
                let p = x.as_ptr().add(i);

                let mut v0 = vabsq_f32(vld1q_f32(p.add(0)));
                let mut v1 = vabsq_f32(vld1q_f32(p.add(4)));
                let mut v2 = vabsq_f32(vld1q_f32(p.add(8)));
                let mut v3 = vabsq_f32(vld1q_f32(p.add(12)));
                
                // replace nans with -inf 
                v0 = vbslq_f32(vceqq_f32(v0, v0), v0, ninf);
                v1 = vbslq_f32(vceqq_f32(v1, v1), v1, ninf);
                v2 = vbslq_f32(vceqq_f32(v2, v2), v2, ninf);
                v3 = vbslq_f32(vceqq_f32(v3, v3), v3, ninf);

                
                let m0 = vmaxvq_f32(v0);
                let m1 = vmaxvq_f32(v1);
                let m2 = vmaxvq_f32(v2);
                let m3 = vmaxvq_f32(v3);
                let mut block_max = m0;
                if m1 > block_max { block_max = m1; }
                if m2 > block_max { block_max = m2; }
                if m3 > block_max { block_max = m3; }
                // block_max is the max value of i..i+16 

                if block_max > best_val {
                    let vm   = vdupq_n_f32(block_max);
                    let idx0 = vaddq_u32(vdupq_n_u32(i as u32      ), iota);
                    let idx1 = vaddq_u32(vdupq_n_u32((i + 4) as u32), iota);
                    let idx2 = vaddq_u32(vdupq_n_u32((i + 8) as u32), iota);
                    let idx3 = vaddq_u32(vdupq_n_u32((i +12) as u32), iota);

                    // find idx associated with best_val 
                    let cand0 = vbslq_u32(vceqq_f32(v0, vm), idx0, allmax);
                    let mut local = vminvq_u32(cand0);
                    if local == u32::MAX {
                        let cand1 = vbslq_u32(vceqq_f32(v1, vm), idx1, allmax);
                        local = vminvq_u32(cand1);
                        if local == u32::MAX {
                            let cand2 = vbslq_u32(vceqq_f32(v2, vm), idx2, allmax);
                            local = vminvq_u32(cand2);
                            if local == u32::MAX {
                                let cand3 = vbslq_u32(vceqq_f32(v3, vm), idx3, allmax);
                                local = vminvq_u32(cand3);
                            }
                        }
                    }

                    best_val = block_max;
                    best_idx = local as usize;
                }

                i += 16;
            }

            // tail
            let mut p = x.as_ptr().add(i);
            while i < n {
                let v = (*p).abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(1); i += 1;
            }
        } else {
            // non unit stride 
            let step = incx as usize;
            let mut i = 0usize;
            let mut p = x.as_ptr();
            while i < n {
                let v = (*p).abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(step); i += 1;
            }
        }

        best_idx + 1
    }
}


#[inline]
pub fn idamax(n: usize, x: &[f64], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }
    debug_assert!(x.len() >= 1 + (n - 1).saturating_mul(incx as usize));

    unsafe {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i = 0usize;
            let iota   = vld1q_u64([0u64, 1].as_ptr());
            let allmax = vdupq_n_u64(u64::MAX);
            let ninf   = vdupq_n_f64(f64::NEG_INFINITY);

            while i + 8 <= n {
                let p = x.as_ptr().add(i);

                let mut v0 = vabsq_f64(vld1q_f64(p.add(0)));
                let mut v1 = vabsq_f64(vld1q_f64(p.add(2)));
                let mut v2 = vabsq_f64(vld1q_f64(p.add(4)));
                let mut v3 = vabsq_f64(vld1q_f64(p.add(6)));

                // replace nans with -inf 
                v0 = vbslq_f64(vceqq_f64(v0, v0), v0, ninf);
                v1 = vbslq_f64(vceqq_f64(v1, v1), v1, ninf);
                v2 = vbslq_f64(vceqq_f64(v2, v2), v2, ninf);
                v3 = vbslq_f64(vceqq_f64(v3, v3), v3, ninf);

                let m0 = vmaxvq_f64(v0);
                let m1 = vmaxvq_f64(v1);
                let m2 = vmaxvq_f64(v2);
                let m3 = vmaxvq_f64(v3);
                let mut block_max = m0;
                if m1 > block_max { block_max = m1; }
                if m2 > block_max { block_max = m2; }
                if m3 > block_max { block_max = m3; }
                // block_max is the max value i..i+8

                if block_max > best_val {
                    let vm = vdupq_n_f64(block_max);

                    let idx0 = vaddq_u64(vdupq_n_u64(i      as u64), iota); // [i,i+1]
                    let idx1 = vaddq_u64(vdupq_n_u64((i+2)  as u64), iota);
                    let idx2 = vaddq_u64(vdupq_n_u64((i+4)  as u64), iota);
                    let idx3 = vaddq_u64(vdupq_n_u64((i+6)  as u64), iota);

                    // find idx associated with best_val 
                    let cand0 = vbslq_u64(vceqq_f64(v0, vm), idx0, allmax);
                    let mut local = {
                        let a = vgetq_lane_u64(cand0, 0);
                        let b = vgetq_lane_u64(cand0, 1);
                        a.min(b)
                    };
                    if local == u64::MAX {
                        let cand1 = vbslq_u64(vceqq_f64(v1, vm), idx1, allmax);
                        let a = vgetq_lane_u64(cand1, 0);
                        let b = vgetq_lane_u64(cand1, 1);
                        local = a.min(b);
                        if local == u64::MAX {
                            let cand2 = vbslq_u64(vceqq_f64(v2, vm), idx2, allmax);
                            let a = vgetq_lane_u64(cand2, 0);
                            let b = vgetq_lane_u64(cand2, 1);
                            local = a.min(b);
                            if local == u64::MAX {
                                let cand3 = vbslq_u64(vceqq_f64(v3, vm), idx3, allmax);
                                let a = vgetq_lane_u64(cand3, 0);
                                let b = vgetq_lane_u64(cand3, 1);
                                local = a.min(b);
                            }
                        }
                    }

                    best_val = block_max;
                    best_idx = local as usize;
                }

                i += 8;
            }

            // tail
            let mut p = x.as_ptr().add(i);
            while i < n {
                let v = (*p).abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(1); i += 1;
            }
        } else {
            // non unit stride 
            let step = incx as usize;
            let mut i = 0usize;
            let mut p = x.as_ptr();
            while i < n {
                let v = (*p).abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(step); i += 1;
            }
        }

        best_idx + 1
    }
}


#[inline]
pub fn icamax(n: usize, x: &[f32], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }
    debug_assert!(x.len() >= 2 + (n - 1).saturating_mul((incx as usize) * 2));

    unsafe {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i = 0usize;            
            let iota_c = vld1q_u32([0u32, 0, 1, 1].as_ptr());
            let allmax = vdupq_n_u32(u32::MAX);
            let ninf   = vdupq_n_f32(f32::NEG_INFINITY);

            while i + 8 <= n {
                let p = x.as_ptr().add(2*i); 

                let a0 = vabsq_f32(vld1q_f32(p.add( 0))); 
                let a1 = vabsq_f32(vld1q_f32(p.add( 4)));
                let a2 = vabsq_f32(vld1q_f32(p.add( 8)));
                let a3 = vabsq_f32(vld1q_f32(p.add(12)));

                // calculate complex sums; 
                // duplicates across lanes, but thats fine 
                // since iota idx is same 
                let s0 = vaddq_f32(a0, vrev64q_f32(a0));
                let s1 = vaddq_f32(a1, vrev64q_f32(a1));
                let s2 = vaddq_f32(a2, vrev64q_f32(a2));
                let s3 = vaddq_f32(a3, vrev64q_f32(a3));

                // replace nans with -inf
                let s0 = vbslq_f32(vceqq_f32(s0, s0), s0, ninf);
                let s1 = vbslq_f32(vceqq_f32(s1, s1), s1, ninf);
                let s2 = vbslq_f32(vceqq_f32(s2, s2), s2, ninf);
                let s3 = vbslq_f32(vceqq_f32(s3, s3), s3, ninf);

                let m0 = vmaxvq_f32(s0);
                let m1 = vmaxvq_f32(s1);
                let m2 = vmaxvq_f32(s2);
                let m3 = vmaxvq_f32(s3);
                let mut block_max = m0;
                if m1 > block_max { block_max = m1; }
                if m2 > block_max { block_max = m2; }
                if m3 > block_max { block_max = m3; }

                if block_max > best_val {
                    let vm = vdupq_n_f32(block_max);

                    let idx0 = vaddq_u32(vdupq_n_u32((i    ) as u32), iota_c);
                    let idx1 = vaddq_u32(vdupq_n_u32((i + 2) as u32), iota_c);
                    let idx2 = vaddq_u32(vdupq_n_u32((i + 4) as u32), iota_c);
                    let idx3 = vaddq_u32(vdupq_n_u32((i + 6) as u32), iota_c);

                    let cand0 = vbslq_u32(vceqq_f32(s0, vm), idx0, allmax);
                    let mut local = vminvq_u32(cand0);
                    if local == u32::MAX {
                        let cand1 = vbslq_u32(vceqq_f32(s1, vm), idx1, allmax);
                        local = vminvq_u32(cand1);
                        if local == u32::MAX {
                            let cand2 = vbslq_u32(vceqq_f32(s2, vm), idx2, allmax);
                            local = vminvq_u32(cand2);
                            if local == u32::MAX {
                                let cand3 = vbslq_u32(vceqq_f32(s3, vm), idx3, allmax);
                                local = vminvq_u32(cand3);
                            }
                        }
                    }

                    best_val = block_max;
                    best_idx = local as usize;
                }

                i += 8;
            }

            // tail 
            let mut k = i;
            let mut p = x.as_ptr().add(2*i);
            while k < n {
                let re = *p;
                let im = *p.add(1);
                let v = re.abs() + im.abs();
                if v > best_val { best_val = v; best_idx = k; }
                p = p.add(2);
                k += 1;
            }
        } else {
            // non unit stride 
            let step = (incx as usize) * 2;
            let mut i = 0usize;
            let mut p = x.as_ptr();
            while i < n {
                let re = *p;
                let im = *p.add(1);
                let v = re.abs() + im.abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(step); i += 1;
            }
        }

        best_idx + 1
    }
}


#[inline]
pub fn izamax(n: usize, x: &[f64], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }
    debug_assert!(x.len() >= 2 + (n - 1).saturating_mul((incx as usize) * 2));

    unsafe {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i = 0usize;
            let allmax = vdupq_n_u64(u64::MAX);
            let ninf   = vdupq_n_f64(f64::NEG_INFINITY);

            while i + 4 <= n {
                let p = x.as_ptr().add(2*i);

                let a0 = vabsq_f64(vld1q_f64(p.add(0)));
                let a1 = vabsq_f64(vld1q_f64(p.add(2)));
                let a2 = vabsq_f64(vld1q_f64(p.add(4)));
                let a3 = vabsq_f64(vld1q_f64(p.add(6)));

                // calculate complex sums; 
                // duplicates across lanes, but thats fine 
                // since iota idx is same 
                let s0 = vaddq_f64(a0, vextq_f64(a0, a0, 1));
                let s1 = vaddq_f64(a1, vextq_f64(a1, a1, 1));
                let s2 = vaddq_f64(a2, vextq_f64(a2, a2, 1));
                let s3 = vaddq_f64(a3, vextq_f64(a3, a3, 1));

                // replace nans with -inf 
                let s0 = vbslq_f64(vceqq_f64(s0, s0), s0, ninf);
                let s1 = vbslq_f64(vceqq_f64(s1, s1), s1, ninf);
                let s2 = vbslq_f64(vceqq_f64(s2, s2), s2, ninf);
                let s3 = vbslq_f64(vceqq_f64(s3, s3), s3, ninf);

                let m0 = vmaxvq_f64(s0);
                let m1 = vmaxvq_f64(s1);
                let m2 = vmaxvq_f64(s2);
                let m3 = vmaxvq_f64(s3);
                let mut block_max = m0;
                if m1 > block_max { block_max = m1; }
                if m2 > block_max { block_max = m2; }
                if m3 > block_max { block_max = m3; }

                if block_max > best_val {
                    let vm = vdupq_n_f64(block_max);

                    let idx0 = vdupq_n_u64((i    ) as u64);
                    let idx1 = vdupq_n_u64((i + 1) as u64);
                    let idx2 = vdupq_n_u64((i + 2) as u64);
                    let idx3 = vdupq_n_u64((i + 3) as u64);

                    let cand0 = vbslq_u64(vceqq_f64(s0, vm), idx0, allmax);
                    let mut local = {
                        let a = vgetq_lane_u64(cand0, 0);
                        let b = vgetq_lane_u64(cand0, 1);
                        a.min(b)
                    };
                    if local == u64::MAX {
                        let cand1 = vbslq_u64(vceqq_f64(s1, vm), idx1, allmax);
                        let a = vgetq_lane_u64(cand1, 0);
                        let b = vgetq_lane_u64(cand1, 1);
                        local = a.min(b);
                        if local == u64::MAX {
                            let cand2 = vbslq_u64(vceqq_f64(s2, vm), idx2, allmax);
                            let a = vgetq_lane_u64(cand2, 0);
                            let b = vgetq_lane_u64(cand2, 1);
                            local = a.min(b);
                            if local == u64::MAX {
                                let cand3 = vbslq_u64(vceqq_f64(s3, vm), idx3, allmax);
                                let a = vgetq_lane_u64(cand3, 0);
                                let b = vgetq_lane_u64(cand3, 1);
                                local = a.min(b);
                            }
                        }
                    }

                    best_val = block_max;
                    best_idx = local as usize;
                }

                i += 4;
            }

            // tail
            let mut k = i;
            let mut p = x.as_ptr().add(2*i);
            while k < n {
                let re = *p;
                let im = *p.add(1);
                let v = re.abs() + im.abs();
                if v > best_val { best_val = v; best_idx = k; }
                p = p.add(2); k += 1;
            }
        } else {
            // non unit scalar 
            let step = (incx as usize) * 2;
            let mut i = 0usize;
            let mut p = x.as_ptr();
            while i < n {
                let re = *p;
                let im = *p.add(1);
                let v = re.abs() + im.abs();
                if v > best_val { best_val = v; best_idx = i; }
                p = p.add(step); i += 1;
            }
        }

        best_idx + 1
    }
}

