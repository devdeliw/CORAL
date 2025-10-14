use crate::level3::c32_packers::{MR, NR};
use crate::level3::microkernel::c32_mrxnr::Complex32; 

#[inline(always)]
pub(crate) fn c32_edge(
    mr    : usize,
    nr    : usize,
    kc    : usize,
    a     : *const f32, // packed A; per k-step: [MR re | MR im]
    b     : *const f32, // packed B; per k-step: [NR re | NR im]
    c     : *mut f32,   // [re, im, re,im, ...]
    ldc   : usize,      // complex elems
    alpha : Complex32,
    beta  : Complex32,
) {
    unsafe {
        let mut acc_re = [[0.0f32; NR]; MR];
        let mut acc_im = [[0.0f32; NR]; MR];

        let mut ap = a;
        let mut bp = b;

        for _ in 0..kc {
            let mut br = [0.0f32; NR];
            let mut bi = [0.0f32; NR];

            core::ptr::copy_nonoverlapping(bp, br.as_mut_ptr(), nr);
            core::ptr::copy_nonoverlapping(bp.add(NR), bi.as_mut_ptr(), nr);

            for r in 0..mr {
                let ar = *ap.add(r);
                let ai = *ap.add(MR + r);

                // (ar + i ai) * (br + i bi)
                // re += ar*br - ai*bi
                // im += ar*bi + ai*br
                for j in 0..nr {
                    acc_re[r][j] += ar * br[j] - ai * bi[j];
                    acc_im[r][j] += ar * bi[j] + ai * br[j];
                }
            }

            ap = ap.add(2 * MR); 
            bp = bp.add(2 * NR); 
        }

        let ar = alpha.re; let ai = alpha.im;
        let br = beta.re;  let bi = beta.im;

        let beta_is_zero = br == 0.0 && bi == 0.0;
        let beta_is_one  = br == 1.0 && bi == 0.0;

        for j in 0..nr {
            let colp = c.add(2 * j * ldc);

            for r in 0..mr {
                let are = acc_re[r][j];
                let aim = acc_im[r][j];

                // alpha * acc
                let add_re = are * ar - aim * ai;
                let add_im = are * ai + aim * ar;

                if beta_is_zero {
                    *colp.add(2 * r + 0) = add_re;
                    *colp.add(2 * r + 1) = add_im;
                } else if beta_is_one {
                    *colp.add(2 * r + 0) += add_re;
                    *colp.add(2 * r + 1) += add_im;
                } else {
                    // C := beta*C + alpha*acc   
                    let cre = *colp.add(2 * r + 0);
                    let cim = *colp.add(2 * r + 1);

                    let scaled_re = cre * br - cim * bi;
                    let scaled_im = cre * bi + cim * br;

                    *colp.add(2 * r + 0) = scaled_re + add_re;
                    *colp.add(2 * r + 1) = scaled_im + add_im;
                }
            }
        }
    }
}

