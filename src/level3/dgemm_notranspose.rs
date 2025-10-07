use crate::level3::packers::{
    a_buf_len, b_buf_len, 
    pack_a_block, pack_b_block, 
}; 
use crate::level3::macro_kernel::macro_kernel; 

pub(crate) fn dgemm_notranspose( 
    m     : usize, 
    n     : usize, 
    k     : usize, 
    alpha : f64, 
    a     : *const f64, 
    lda   : usize, 
    b     : *const f64, 
    ldb   : usize, 
    beta  : f64, 
    c     : *mut f64, 
    ldc   : usize, 
) { 
    unsafe { 
        if alpha == 0.0 || k == 0 { 
            // scale C by beta 
            if beta == 0.0 { 
                for j in 0..n { 
                    let col = c.add(j * ldc); 
                    core::ptr::write_bytes(col, 0, m);
                }
            } else if beta != 1.0 { 
                for j in 0..n { 
                    let col = c.add(j * ldc); 

                    for i in 0..m { 
                        *col.add(i) *= beta; 
                    }
                }
            } 

            return;
        }

        let mut a_buf = vec![0.0; a_buf_len(MC, KC).max(a_buf_len(m, KC))]; 
        let mut b_buf = vec![0.0; b_buf_len(KC, NC).max(b_buf_len(KC, n))]; 

        let mut j0 = 0; 
        while j0 < n { 
            let nc = core::cmp::min(NC, n - j0); 

            let mut l0 = 0; 
            while l0 < k { 
                let kcblk = core::cmp::min(KC, k - l0); 

                // pack B(kcblk x nc) starting at (l0, j0) 
                { 
                    let b_block_base = b.add(l0 * j0 * ldb); 
                    pack_b_block(kcblk, nc, b_block_base, ldb, b_buf.as_mut_ptr());
                }

                let beta_panel = if l0 == 0 { beta } else { 1.0 }; 

                let mut i0 = 0; 
                while i0 < m { 
                    let mc = core::cmp::min(MC, m - i0); 

                    // pack A(mc x kcblk) at (i0, l0) 
                    { 
                        let a_block_base = a.add(i0 * l0 * lda); 
                        pack_a_block(mc, kcblk, a_block_base, lda, a_buf.as_mut_ptr());
                    }

                    let c_base = c.add(i0 + j0 * ldc); 

                    macro_kernel( 
                        mc, 
                        nc, 
                        kcblk, 
                        alpha, 
                        beta_panel, 
                        a_buf.as_ptr(), 
                        b_buf.as_ptr(), 
                        c_base, 
                        ldc
                    ); 

                    i0 += MC; 
                }

                l0 += KC; 
            }

            j0 += NC; 
        }

    }
}
