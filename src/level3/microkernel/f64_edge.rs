use crate::level3::packers::{MR, NR}; 

#[inline(always)] 
pub(crate) fn f64_edge( 
    mr    : usize, 
    nr    : usize, 
    kc    : usize, 
    a     : *const f64, 
    b     : *const f64, 
    c     : *mut f64, 
    ldc   : usize, 
    alpha : f64, 
    beta  : f64, 
) { 
    unsafe { 
        let mut acc = [[0.0; 8]; 6]; 
        let mut ap  = a; 
        let mut bp  = b; 

        for _ in 0..kc { 
            let mut btmp = [0.0; 8]; 
           
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
