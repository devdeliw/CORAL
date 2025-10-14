use crate::level3::{
    cgemm::{MC, NC, KC},
    c32_macro_kernel::macro_kernel,
    c32_packers::{
        pack_a_block, pack_b_block,
        a_buf_len, b_buf_len,
    },
};
use crate::level3::microkernel::c32_mrxnr::Complex32;

#[inline(always)]
fn is_zero(z: Complex32) -> bool { 
    z.re == 0.0 && z.im == 0.0 
}

const ONE_C : Complex32 = Complex32 { 
    re: 1.0,
    im: 0.0 
};

pub(crate) fn cgemm_nn(
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : Complex32,
    a     : *const f32, 
    lda   : usize,
    b     : *const f32, 
    ldb   : usize,
    beta  : Complex32,
    c     : *mut f32,   
    ldc   : usize,
) {
    debug_assert!(
        ldc >= m && lda >= m && ldb >= k,
        "matrix dimensions don't satisfy lda/ldb/ldc (in complex elements)"
    );

    unsafe {
        // fast-path;
        // alpha==0 or k==0 => C := beta * C
        if is_zero(alpha) || k == 0 {
            if is_zero(beta) {
                // zero C
                for j in 0..n {
                    let col = c.add(2 * j * ldc);
                    core::ptr::write_bytes(col, 0, 2 * m);
                }
            } else if beta.re == 1.0 && beta.im == 0.0 {
                // C := C 
            } else {
                // complex scale each entry;
                // (re,im) *= beta 
                let br = beta.re;
                let bi = beta.im;

                for j in 0..n {
                    let col = c.add(2 * j * ldc);

                    for i in 0..m {
                        let re = *col.add(2 * i + 0);
                        let im = *col.add(2 * i + 1);
                        *col.add(2 * i + 0) = re * br - im * bi;
                        *col.add(2 * i + 1) = re * bi + im * br;
                    }
                }
            }
            return;
        }

        let mut a_buf = vec![0.0; a_buf_len(MC, KC)];
        let mut b_buf = vec![0.0; b_buf_len(KC, NC)];

        let mut j0 = 0;
        while j0 < n {
            let nc = core::cmp::min(NC, n - j0);

            let mut l0 = 0;
            while l0 < k {
                let kcblk = core::cmp::min(KC, k - l0);

                // pack B (kcblk x nc), base at (l0, j0)
                {
                    let b_block_base = b.add(2 * (l0 + j0 * ldb));
                    pack_b_block(kcblk, nc, b_block_base, ldb, b_buf.as_mut_ptr());
                }

                let beta_panel = if l0 == 0 { beta } else { ONE_C };

                let mut i0 = 0;
                while i0 < m {
                    let mc = core::cmp::min(MC, m - i0);

                    // pack A (mc x kcblk), base at (i0, l0)
                    {
                        let a_block_base = a.add(2 * (i0 + l0 * lda));
                        pack_a_block(mc, kcblk, a_block_base, lda, a_buf.as_mut_ptr());
                    }

                    let c_base = c.add(2 * (i0 + j0 * ldc));

                    // calc C[i0..i0+mc, j0..j0+nc] += alpha * Ablk * Bblk
                    macro_kernel(
                        mc,
                        nc,
                        kcblk,
                        alpha,
                        beta_panel,
                        a_buf.as_ptr(),
                        b_buf.as_ptr(),
                        c_base,
                        ldc,
                    );

                    i0 += mc;
                }

                l0 += kcblk;
            }

            j0 += nc;
        }
    }
}

