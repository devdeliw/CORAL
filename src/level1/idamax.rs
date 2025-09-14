//! Finds the index of the element with maximum absolute value in a double precision vector.
//!
//! This function implements the BLAS [`idamax`] routine, returning the 1-based index
//! of the first element of maximum absolute value over `n` elements of the input
//! vector `x` with a specified stride.
//!
//! # Arguments
//! - `n`    (usize)  : Number of elements in the vector.
//! - `x`    (&[f64]) : Input slice containing vector elements.
//! - `incx` (usize)  : Stride between consecutive elements of `x`.
//!
//! # Returns
//! - `usize` 0-based index of the first element with maximum absolute value.
//!
//! # Notes
//! - For `incx == 1`, [`idamax`] uses unrolled NEON SIMD instructions for optimized
//!   performance on AArch64, with NaN values treated as negative infinity.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `incx <= 0`, the function returns `0`.
//!
//! # Author
//! Deval Deliwala

#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{ 
    vld1q_u64, 
    vdupq_n_u64,
    vaddq_u64, 
    vbslq_u64, 
    vgetq_lane_u64, 
    vld1q_f64, 
    vdupq_n_f64, 
    vabsq_f64,
    vceqq_f64,
    vmaxvq_f64,
    vbslq_f64, 
};
use crate::level1::assert_length_helpers::required_len_ok; 


#[inline]
#[cfg(target_arch = "aarch64")] 
pub fn idamax(
    n       : usize,
    x       : &[f64],
    incx    : usize
) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    unsafe {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0;

        // fast path 
        if incx == 1 {
            let mut i = 0;
            let iota   = vld1q_u64([0, 1].as_ptr());
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

                    let idx0 = vaddq_u64(vdupq_n_u64(i      as u64), iota); 
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

                if v > best_val { 
                    best_val = v; 
                    best_idx = i; 
                }
                p = p.add(1);
                i += 1;
            }
        } else {
            // non unit stride 
            let mut i = 0;
            let mut p = x.as_ptr();

            while i < n {
                let v = (*p).abs();

                if v > best_val { 
                    best_val = v; 
                    best_idx = i; 
                }
                p = p.add(incx); i += 1;
            }
        }

        best_idx + 1
    }
}
