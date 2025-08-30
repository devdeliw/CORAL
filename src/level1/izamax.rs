use core::arch::aarch64::{ 
    vdupq_n_u64, vbslq_u64, vgetq_lane_u64, 
    vld1q_f64, vaddq_f64, vextq_f64, vdupq_n_f64, vabsq_f64, vceqq_f64, vmaxvq_f64, vbslq_f64, 
}; 
use crate::level1::assert_length_helpers::required_len_ok_cplx;

pub fn izamax(n: usize, x: &[f64], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }

    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");

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


