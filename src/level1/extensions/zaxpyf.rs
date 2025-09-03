use core::arch::aarch64::{
    vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld2q_f64, vst2q_f64,
    float64x2x2_t
};
use crate::level1::assert_length_helpers::required_len_ok_cplx;
use crate::level1::zaxpy::zaxpy; 

#[inline(always)]
pub fn zaxpyf(
    m     : usize,        
    n     : usize,        
    x     : &[f64],       
    incx  : isize,        
    a     : &[f64],       
    lda   : usize,        
    y     : &mut [f64],   
    incy  : isize,        
) {
    // quick return
    if m == 0 || n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok_cplx(y.len(), m, incy), "y too short for m/incy");
    if n > 0 {
        debug_assert!(lda >= m, "lda must be >= m (in complexes)");
        let need = 2*((n - 1).saturating_mul(lda) + m);
        debug_assert!(a.len() >= need, "A too small for m,n,lda; need at least 2*((n-1)*lda + m)");
    }

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            let mut j = 0usize;

            // unrolled over cols
            while j + 4 <= n {
                let x_ptr0 = x.as_ptr().add(2*(j + 0));
                let x_ptr1 = x.as_ptr().add(2*(j + 1));
                let x_ptr2 = x.as_ptr().add(2*(j + 2));
                let x_ptr3 = x.as_ptr().add(2*(j + 3));

                let xr0 = *x_ptr0;       let xi0 = *x_ptr0.add(1);
                let xr1 = *x_ptr1;       let xi1 = *x_ptr1.add(1);
                let xr2 = *x_ptr2;       let xi2 = *x_ptr2.add(1);
                let xr3 = *x_ptr3;       let xi3 = *x_ptr3.add(1);

                let sr0 =xr0 - xi0; let si0 =xi0 + xr0;
                let sr1 =xr1 - xi1; let si1 =xi1 + xr1;
                let sr2 =xr2 - xi2; let si2 =xi2 + xr2;
                let sr3 =xr3 - xi3; let si3 =xi3 + xr3;

                if (sr0 == 0.0 && si0 == 0.0) &&
                   (sr1 == 0.0 && si1 == 0.0) &&
                   (sr2 == 0.0 && si2 == 0.0) &&
                   (sr3 == 0.0 && si3 == 0.0) {
                    j += 4;
                    continue;
                }

                let s0r = vdupq_n_f64(sr0); let s0i = vdupq_n_f64(si0);
                let s1r = vdupq_n_f64(sr1); let s1i = vdupq_n_f64(si1);
                let s2r = vdupq_n_f64(sr2); let s2i = vdupq_n_f64(si2);
                let s3r = vdupq_n_f64(sr3); let s3i = vdupq_n_f64(si3);

                let pa0 = a.as_ptr().add(2*((j + 0)*lda));
                let pa1 = a.as_ptr().add(2*((j + 1)*lda));
                let pa2 = a.as_ptr().add(2*((j + 2)*lda));
                let pa3 = a.as_ptr().add(2*((j + 3)*lda));

                let mut i = 0usize;

                // unrolled over rows
                while i + 4 <= m {
                    let p0 = 2*i;

                    let y01 = vld2q_f64(y.as_ptr().add(p0));
                    let mut yr0 = y01.0;
                    let mut yi0 = y01.1;

                    let a0_01 = vld2q_f64(pa0.add(p0));
                    let a1_01 = vld2q_f64(pa1.add(p0));
                    let a2_01 = vld2q_f64(pa2.add(p0));
                    let a3_01 = vld2q_f64(pa3.add(p0));

                    yr0 = vfmaq_f64(yr0, s0r, a0_01.0); 
                    yr0 = vfmsq_f64(yr0, s0i, a0_01.1);  
                    yi0 = vfmaq_f64(yi0, s0r, a0_01.1);  
                    yi0 = vfmaq_f64(yi0, s0i, a0_01.0);  

                    yr0 = vfmaq_f64(yr0, s1r, a1_01.0);
                    yr0 = vfmsq_f64(yr0, s1i, a1_01.1);
                    yi0 = vfmaq_f64(yi0, s1r, a1_01.1);
                    yi0 = vfmaq_f64(yi0, s1i, a1_01.0);

                    yr0 = vfmaq_f64(yr0, s2r, a2_01.0);
                    yr0 = vfmsq_f64(yr0, s2i, a2_01.1);
                    yi0 = vfmaq_f64(yi0, s2r, a2_01.1);
                    yi0 = vfmaq_f64(yi0, s2i, a2_01.0);

                    yr0 = vfmaq_f64(yr0, s3r, a3_01.0);
                    yr0 = vfmsq_f64(yr0, s3i, a3_01.1);
                    yi0 = vfmaq_f64(yi0, s3r, a3_01.1);
                    yi0 = vfmaq_f64(yi0, s3i, a3_01.0);

                    vst2q_f64(y.as_mut_ptr().add(p0), float64x2x2_t(yr0, yi0));

                    let p1  = p0 + 4;
                    let y23 = vld2q_f64(y.as_ptr().add(p1));
                    let mut yr1 = y23.0;
                    let mut yi1 = y23.1;

                    let a0_23 = vld2q_f64(pa0.add(p1));
                    let a1_23 = vld2q_f64(pa1.add(p1));
                    let a2_23 = vld2q_f64(pa2.add(p1));
                    let a3_23 = vld2q_f64(pa3.add(p1));

                    yr1 = vfmaq_f64(yr1, s0r, a0_23.0);  
                    yr1 = vfmsq_f64(yr1, s0i, a0_23.1);
                    yi1 = vfmaq_f64(yi1, s0r, a0_23.1);  
                    yi1 = vfmaq_f64(yi1, s0i, a0_23.0);

                    yr1 = vfmaq_f64(yr1, s1r, a1_23.0);  
                    yr1 = vfmsq_f64(yr1, s1i, a1_23.1);
                    yi1 = vfmaq_f64(yi1, s1r, a1_23.1);  
                    yi1 = vfmaq_f64(yi1, s1i, a1_23.0);

                    yr1 = vfmaq_f64(yr1, s2r, a2_23.0);  
                    yr1 = vfmsq_f64(yr1, s2i, a2_23.1);
                    yi1 = vfmaq_f64(yi1, s2r, a2_23.1);  
                    yi1 = vfmaq_f64(yi1, s2i, a2_23.0);

                    yr1 = vfmaq_f64(yr1, s3r, a3_23.0);  
                    yr1 = vfmsq_f64(yr1, s3i, a3_23.1);
                    yi1 = vfmaq_f64(yi1, s3r, a3_23.1);  
                    yi1 = vfmaq_f64(yi1, s3i, a3_23.0);

                    vst2q_f64(y.as_mut_ptr().add(p1), float64x2x2_t(yr1, yi1));

                    i += 4;
                }

                while i + 2 <= m {
                    let p = 2*i;

                    let y01 = vld2q_f64(y.as_ptr().add(p));
                    let mut yr = y01.0;
                    let mut yi = y01.1;

                    let a0 = vld2q_f64(pa0.add(p));
                    let a1 = vld2q_f64(pa1.add(p));
                    let a2 = vld2q_f64(pa2.add(p));
                    let a3 = vld2q_f64(pa3.add(p));

                    yr = vfmaq_f64(yr, s0r, a0.0); 
                    yr = vfmsq_f64(yr, s0i, a0.1);
                    yi = vfmaq_f64(yi, s0r, a0.1); 
                    yi = vfmaq_f64(yi, s0i, a0.0);

                    yr = vfmaq_f64(yr, s1r, a1.0); 
                    yr = vfmsq_f64(yr, s1i, a1.1);
                    yi = vfmaq_f64(yi, s1r, a1.1); 
                    yi = vfmaq_f64(yi, s1i, a1.0);

                    yr = vfmaq_f64(yr, s2r, a2.0); 
                    yr = vfmsq_f64(yr, s2i, a2.1);
                    yi = vfmaq_f64(yi, s2r, a2.1); 
                    yi = vfmaq_f64(yi, s2i, a2.0);

                    yr = vfmaq_f64(yr, s3r, a3.0); 
                    yr = vfmsq_f64(yr, s3i, a3.1);
                    yi = vfmaq_f64(yi, s3r, a3.1); 
                    yi = vfmaq_f64(yi, s3i, a3.0);

                    vst2q_f64(y.as_mut_ptr().add(p), float64x2x2_t(yr, yi));

                    i += 2;
                }

                // row tail
                while i < m {
                    let p = 2*i;

                    let ar0 = *pa0.add(p); let ai0 = *pa0.add(p+1);
                    let ar1 = *pa1.add(p); let ai1 = *pa1.add(p+1);
                    let ar2 = *pa2.add(p); let ai2 = *pa2.add(p+1);
                    let ar3 = *pa3.add(p); let ai3 = *pa3.add(p+1);

                    let yrp = y.as_mut_ptr().add(p);
                    let yip = y.as_mut_ptr().add(p+1);

                    let mut yr = *yrp;
                    let mut yi = *yip;

                    yr += sr0*ar0 - si0*ai0;  yi += sr0*ai0 + si0*ar0;
                    yr += sr1*ar1 - si1*ai1;  yi += sr1*ai1 + si1*ar1;
                    yr += sr2*ar2 - si2*ai2;  yi += sr2*ai2 + si2*ar2;
                    yr += sr3*ar3 - si3*ai3;  yi += sr3*ai3 + si3*ar3;

                    *yrp = yr; *yip = yi;

                    i += 1;
                }

                j += 4;
            }

            // col tail 
            let rem = n - j;
            for k in 0..rem {
                let xpk = x.as_ptr().add(2*(j+k));
                let xr = *xpk; let xi = *xpk.add(1);
                let sr =xr - xi;
                let si =xi + xr;
                if sr == 0.0 && si == 0.0 { continue; }

                let sr_v = vdupq_n_f64(sr);
                let si_v = vdupq_n_f64(si);

                let pac = a.as_ptr().add(2*((j+k)*lda));

                let mut i = 0usize;

                // unrolled over rows
                while i + 4 <= m {
                    let p0 = 2*i;
                    let y01 = vld2q_f64(y.as_ptr().add(p0));
                    let mut yr0 = y01.0;
                    let mut yi0 = y01.1;

                    let a01 = vld2q_f64(pac.add(p0));
                    yr0 = vfmaq_f64(yr0, sr_v, a01.0);
                    yr0 = vfmsq_f64(yr0, si_v, a01.1);
                    yi0 = vfmaq_f64(yi0, sr_v, a01.1);
                    yi0 = vfmaq_f64(yi0, si_v, a01.0);
                    vst2q_f64(y.as_mut_ptr().add(p0), float64x2x2_t(yr0, yi0));

                    let p1 = p0 + 4;
                    let y23 = vld2q_f64(y.as_ptr().add(p1));
                    let mut yr1 = y23.0;
                    let mut yi1 = y23.1;

                    let a23 = vld2q_f64(pac.add(p1));
                    yr1 = vfmaq_f64(yr1, sr_v, a23.0);
                    yr1 = vfmsq_f64(yr1, si_v, a23.1);
                    yi1 = vfmaq_f64(yi1, sr_v, a23.1);
                    yi1 = vfmaq_f64(yi1, si_v, a23.0);
                    vst2q_f64(y.as_mut_ptr().add(p1), float64x2x2_t(yr1, yi1));

                    i += 4;
                }
                while i + 2 <= m {
                    let p = 2*i;
                    let y01 = vld2q_f64(y.as_ptr().add(p));
                    let mut yr = y01.0;
                    let mut yi = y01.1;

                    let a01 = vld2q_f64(pac.add(p));
                    yr = vfmaq_f64(yr, sr_v, a01.0);
                    yr = vfmsq_f64(yr, si_v, a01.1);
                    yi = vfmaq_f64(yi, sr_v, a01.1);
                    yi = vfmaq_f64(yi, si_v, a01.0);
                    vst2q_f64(y.as_mut_ptr().add(p), float64x2x2_t(yr, yi));

                    i += 2;
                }

                // tail 
                while i < m {
                    let p = 2*i;
                    let ar0 = *pac.add(p); let ai0 = *pac.add(p+1);

                    let yrp = y.as_mut_ptr().add(p);
                    let yip = y.as_mut_ptr().add(p+1);

                    *yrp += sr*ar0 - si*ai0;
                    *yip += sr*ai0 + si*ar0;

                    i += 1;
                }
            }

            return;
        }

        // non unit stride
        let stepx  = if incx > 0 { incx as usize } else { (-incx) as usize };
        let mut ix = if incx >= 0 { 0usize } else { (n - 1) * stepx };

        for j in 0..n {
            let xr = *x.as_ptr().add(2*ix);
            let xi = *x.as_ptr().add(2*ix + 1);
            let sr =xr - xi;
            let si =xi + xr;
            if sr != 0.0 || si != 0.0 {
                // col major; contiguous 
                let col_ptr = a.as_ptr().add(2*(j*lda)); 
                let col = core::slice::from_raw_parts(col_ptr, 2*m);
                zaxpy(m, [sr, si], col, 1, y, incy);
            }

            ix = if incx >= 0 { ix + stepx } else { ix.wrapping_sub(stepx) };
        }
    }
}

