use crate::level3::f32_packers::{MR, NR}; 

#[inline(always)] 
pub(crate) fn f32_edge( 
    mr    : usize, 
    nr    : usize, 
    kc    : usize, 
    a     : *const f32, 
    b     : *const f32, 
    c     : *mut f32, 
    ldc   : usize, 
    alpha : f32, 
    beta  : f32, 
) { 
    unsafe { 
        let mut acc = [[0.0; NR]; MR]; 
        let mut ap  = a; 
        let mut bp  = b; 

        for _ in 0..kc { 
            let mut btmp = [0.0; NR]; 
           
            core::ptr::copy_nonoverlapping(bp, btmp.as_mut_ptr(), nr);

            for r in 0..mr { 
                let ar = *ap.add(r); 

                for ccol in 0..nr { 
                    acc[r][ccol] += ar * btmp[ccol]; 
                }
            }

            ap = ap.add(MR); 
            bp = bp.add(NR); 
        }

        for ccol in 0..nr { 
            let colp = c.add(ccol * ldc); 
            
            if beta == 0.0 { 

                for r in 0..mr { 
                    *colp.add(r) = alpha * acc[r][ccol];
                } 

            } else if beta == 1.0 {

                for r in 0..mr {
                    *colp.add(r) += alpha * acc[r][ccol];
                } 

            } else { 

                for r in 0..mr {
                    *colp.add(r) = beta * *colp.add(r) + alpha * acc[r][ccol];
                } 
            }
        }

    }       
}

