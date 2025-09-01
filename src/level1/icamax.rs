//! Finds the index of the element with maximum absolute value in a complex single precision vector.
//!
//! This function implements the BLAS [`icamax`] routine, returning the 1-based index
//! of the first complex element of maximum absolute value over `n` elements of the input
//! vector `x` with a specified stride.
//!
//! The absolute value of a complex number is defined here as |Re(x)| + |Im(x)|.
//!
//! # Arguments
//! - `n`    : Number of complex elements in the vector.
//! - `x`    : Input slice containing interleaved complex vector elements
//!            `[re0, im0, re1, im1, ...]`.
//! - `incx` : Stride between consecutive complex elements of `x`
//!            (measured in complex numbers; every step advances two scalar idxs).
//!
//! # Returns
//! - `usize` 1-based index of the first complex element with maximum absolute value.
//!
//! # Notes
//! - For `incx == 1`, [`icamax`] uses unrolled NEON SIMD instructions for optimized
//!   performance on AArch64, with NaN values treated as negative infinity.
//! - For non unit strides, the function falls back to a scalar loop.
//! - If `n == 0` or `incx <= 0`, the function returns `0`.
//!
//! # Author
//! Deval Deliwala


use core::arch::aarch64::{ 
    vld1q_u32, vdupq_n_u32, vaddq_u32, vbslq_u32, vminvq_u32, 
    vld1q_f32, vdupq_n_f32, vaddq_f32, vabsq_f32, vbslq_f32, vceqq_f32, vmaxvq_f32, vrev64q_f32
}; 
use crate::level1::assert_length_helpers::required_len_ok_cplx; 


#[inline] 
pub fn icamax(n: usize, x: &[f32], incx: isize) -> usize {
    // quick return 
    if n == 0 || incx <= 0 { return 0; }

    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");

    unsafe {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        // fast path 
        if incx == 1 {
            let mut i  = 0usize;            
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
                // but thats fine since iota idx is same 
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
                if v > best_val { 
                    best_val = v; 
                    best_idx = i; 
                }
                p = p.add(step); i += 1;
            }
        }

        best_idx + 1
    }
}
