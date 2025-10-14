use crate::level3::f32_packers::{MR, NR};
use crate::level3::microkernel::{
    f32_edge::f32_edge,
    f32_mrxnr::{
        f32_mrxnr_beta0, 
        f32_mrxnr_beta1,
        f32_mrxnr_betax
    },
};

#[inline(always)]
pub(crate) fn macro_kernel(
    mc     : usize,
    nc     : usize,
    kc     : usize,
    alpha  : f32,
    beta_panel: f32,
    a_pack : *const f32,
    b_pack : *const f32,
    c_base : *mut f32,
    ldc    : usize,
) {
    unsafe {
        let np = (nc + NR - 1) / NR;
        let mp = (mc + MR - 1) / MR;

        for jp in 0..np {
            let nr = core::cmp::min(NR, nc - jp * NR);
            let bp = b_pack.add(jp * kc * NR);

            for ip in 0..mp {
                let mr = core::cmp::min(MR, mc - ip * MR);
                let ap = a_pack.add(ip * kc * MR);
                let cptr = c_base.add(ip * MR + (jp * NR) * ldc);

                if mr == MR && nr == NR {
                    if beta_panel == 0.0 {

                        f32_mrxnr_beta0(kc, ap, bp, cptr, ldc, alpha);

                    } else if beta_panel == 1.0 {

                        f32_mrxnr_beta1(kc, ap, bp, cptr, ldc, alpha);

                    } else {

                        f32_mrxnr_betax(kc, ap, bp, cptr, ldc, alpha, beta_panel);

                    }
                } else {

                    f32_edge(mr, nr, kc, ap, bp, cptr, ldc, alpha, beta_panel);
                }
            }
        }
    }
}

