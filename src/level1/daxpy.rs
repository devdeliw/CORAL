use core::arch::aarch64::{ 
    vld1q_f64, vdupq_n_f64, vfmaq_f64, vst1q_f64 
}; 
use crate::level1::assert_length_helpers::required_len_ok; 

#[inline(always)]
pub fn daxpy(n: usize, alpha: f64, x: &[f64], incx: isize, y: &mut [f64], incy: isize) { 
    // quick return 
    if n == 0 || alpha == 0.0 { 
        return; 
    } 

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    unsafe { 
        let av = vdupq_n_f64(alpha); 

        // fast path 
        if incx == 1 && incy == 1 { 
            let mut i = 0; 
            while i + 8 <= n { 
                let x0 = vld1q_f64(x.as_ptr().add(i)); 
                let x1 = vld1q_f64(x.as_ptr().add(i + 2)); 
                let x2 = vld1q_f64(x.as_ptr().add(i + 4)); 
                let x3 = vld1q_f64(x.as_ptr().add(i + 6));

                let y0 = vld1q_f64(y.as_ptr().add(i)); 
                let y1 = vld1q_f64(y.as_ptr().add(i + 2)); 
                let y2 = vld1q_f64(y.as_ptr().add(i + 4)); 
                let y3 = vld1q_f64(y.as_ptr().add(i + 6)); 

                let r0 = vfmaq_f64(y0, av, x0); 
                let r1 = vfmaq_f64(y1, av, x1); 
                let r2 = vfmaq_f64(y2, av, x2); 
                let r3 = vfmaq_f64(y3, av, x3); 

                vst1q_f64(y.as_mut_ptr().add(i), r0);
                vst1q_f64(y.as_mut_ptr().add(i + 2), r1);
                vst1q_f64(y.as_mut_ptr().add(i + 4), r2);
                vst1q_f64(y.as_mut_ptr().add(i + 6), r3);

                i += 8; 
            }

            while i + 4 <= n { 
                let x0 = vld1q_f64(x.as_ptr().add(i)); 
                let x1 = vld1q_f64(x.as_ptr().add(i + 2)); 

                let y0 = vld1q_f64(y.as_ptr().add(i)); 
                let y1 = vld1q_f64(y.as_ptr().add(i + 2)); 

                let r0 = vfmaq_f64(y0, av, x0); 
                let r1 = vfmaq_f64(y1, av, x1); 

                vst1q_f64(y.as_mut_ptr().add(i), r0);
                vst1q_f64(y.as_mut_ptr().add(i + 2), r1);

                i += 4; 
            }

            // tail 
            while i < n { 
                let x0 = *x.as_ptr().add(i); 
                let y0 = y.as_mut_ptr().add(i); 
    
                *y0 += alpha * x0; 

                i += 1; 

            }
        } else { 
            // non unit stride 
            let stepx = if incx > 0 { incx as usize } else { (-incx) as usize }; 
            let stepy = if incy > 0 { incy as usize } else { (-incy) as usize }; 

            let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx }; 
            let mut iy = if incy >= 0 { 0usize } else { (n - 1) * stepy }; 

            for _ in 0..n { 
                let x0 = *x.as_ptr().add(ix); 
                let y0 = y.as_mut_ptr().add(iy); 

                *y0 += alpha * x0; 

                ix += stepx; 
                iy += stepy;  
            }
            
        }
    }
}

