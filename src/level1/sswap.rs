use core::arch::aarch64::{
    vld1q_f32, vst1q_f32
};
use crate::level1::assert_length_helpers::required_len_ok; 


pub fn sswap(n: usize, x: &mut [f32], incx: isize, y: &mut [f32], incy: isize) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "increments must be nonzero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");
    
    unsafe { 
        // fast path 
        if incx == 1 && incy == 1 { 
            let px = x.as_mut_ptr(); 
            let py = y.as_mut_ptr(); 

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

                vst1q_f32(py.add(i), ax0);
                vst1q_f32(py.add(i + 4), ax1);
                vst1q_f32(py.add(i + 8), ax2);
                vst1q_f32(py.add(i + 12), ax3);

                vst1q_f32(px.add(i), ay0);
                vst1q_f32(px.add(i + 4), ay1);
                vst1q_f32(px.add(i + 8), ay2);
                vst1q_f32(px.add(i + 12), ay3);

                i += 16; 
            }

            while i + 4 < n { 
                let ax = vld1q_f32(px.add(i)); 
                let ay = vld1q_f32(py.add(i)); 
                vst1q_f32(px.add(i), ay);
                vst1q_f32(py.add(i), ax);

                i += 4; 
            }

            // tail 
            while i < n { 
                let a = *px.add(i); 
                *px.add(i) = *py.add(i); 
                *py.add(i) = a; 
                
                i += 1; 
            }
            return; 
        } 

        // non unit stride 
        let px = x.as_mut_ptr(); 
        let py = y.as_mut_ptr(); 

        let stepx = if incx > 0 { incx as usize } else { (-incx) as usize }; 
        let stepy = if incy > 0 { incy as usize } else { (-incy) as usize }; 

        let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx }; 
        let mut iy = if incy >= 0 { 0usize } else { (n - 1) * stepy }; 

        for _ in 0..n { 
            let a = *px.add(ix); 
            *px.add(ix) = *py.add(iy); 
            *py.add(iy) = a; 

            if incx >= 0 { ix += stepx } else { ix = ix.wrapping_sub(stepx) };
            if incy >= 0 { iy += stepy } else { iy = iy.wrapping_sub(stepy) }; 
        }
    }
}


