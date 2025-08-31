use core::arch::aarch64::{ 
    vld1q_f32, vdupq_n_f32, vfmaq_f32, vaddvq_f32, vaddq_f32 
};
use crate::level1::assert_length_helpers::required_len_ok; 

#[inline] 
pub fn sdot(n: usize, x: &[f32], incx: isize, y: &[f32], incy: isize) -> f32 { 
    // quick return 
    if n == 0 { return 0.0; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    let px = x.as_ptr(); 
    let py = y.as_ptr(); 

    // fast path 
    if incx == 1 && incy == 1 {
        unsafe { 
            let mut acc0 = vdupq_n_f32(0.0); 
            let mut acc1 = vdupq_n_f32(0.0); 
            let mut acc2 = vdupq_n_f32(0.0); 
            let mut acc3 = vdupq_n_f32(0.0); 

            let mut i = 0usize; 

            while i + 16 <= n { 
                let ax0 = vld1q_f32(px.add(i)); 
                let ax1 = vld1q_f32(px.add(i + 4)); 
                let ax2 = vld1q_f32(px.add(i + 8)); 
                let ax3 = vld1q_f32(px.add(i + 12)); 

                let ay0 = vld1q_f32(py.add(i)); 
                let ay1 = vld1q_f32(py.add(i + 4)); 
                let ay2 = vld1q_f32(py.add(i + 8)); 
                let ay3 = vld1q_f32(py.add(i + 12)); 

                acc0 = vfmaq_f32(acc0, ax0, ay0);
                acc1 = vfmaq_f32(acc1, ax1, ay1); 
                acc2 = vfmaq_f32(acc2, ax2, ay2); 
                acc3 = vfmaq_f32(acc3, ax3, ay3); 

                i += 16; 
            }

            while i + 4 <= n { 
                let ax = vld1q_f32(px.add(i)); 
                let ay = vld1q_f32(py.add(i));
                acc0   = vfmaq_f32(acc0, ax, ay); 

                i += 4; 
            }

            let accv    = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)); 
            let mut acc = vaddvq_f32(accv); 

            // tail 
            while i < n { 
                acc += *px.add(i) * *py.add(i); 

                i += 1
            } 

            return acc; 
        } 
    } 
    // non unit stride 
    unsafe {
        let mut ix: isize = if incx >= 0 { 0 } else { ((n - 1) as isize) * (-incx) }; 
        let mut iy: isize = if incy >= 0 { 0 } else { ((n - 1) as isize) * (-incy) }; 

        let mut acc = 0.0f32; 
        for _ in 0..n { 
            acc += *px.offset(ix) * *py.offset(iy); 
            ix += incx; 
            iy += incy; 
        }

        acc
    }
}

