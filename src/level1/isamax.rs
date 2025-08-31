use core::arch::aarch64::{ 
    vld1q_u32, vdupq_n_u32, vaddq_u32, vbslq_u32, vminvq_u32, 
    vld1q_f32, vdupq_n_f32, vabsq_f32, vbslq_f32, vceqq_f32, vmaxvq_f32, 
}; 
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline] 
pub fn isamax(n: usize, x: &[f32], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    unsafe {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i  = 0usize;
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
