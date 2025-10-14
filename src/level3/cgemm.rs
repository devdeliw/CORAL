use crate::level3::{
    cgemm_nn::cgemm_nn,
    cgemm_nt::cgemm_nt,
    cgemm_tn::cgemm_tn,
    cgemm_tt::cgemm_tt,
    cgemm_nc::cgemm_nc,
    cgemm_tc::cgemm_tc,
    cgemm_cn::cgemm_cn,
    cgemm_ct::cgemm_ct,
    cgemm_cc::cgemm_cc,
};

use crate::enums::CoralTranspose;
use crate::level3::microkernel::c32_mrxnr::Complex32; 

pub const MC: usize = 384;
pub const NC: usize = 576;
pub const KC: usize = 256;

#[inline(always)]
fn c32(x: [f32; 2]) -> Complex32 {
    Complex32 { 
        re: x[0], 
        im: x[1]
    }
}

#[inline(always)]
pub fn cgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : usize,
    b     : *const f32,
    ldb   : usize,
    beta  : [f32; 2],
    c     : *mut f32,
    ldc   : usize,
) {
    
    let alpha = c32(alpha); 
    let beta  = c32(beta); 

    match (op_a, op_b) {
        (CoralTranspose::NoTranspose,        CoralTranspose::NoTranspose)        => cgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::Transpose)          => cgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::NoTranspose)        => cgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::Transpose)          => cgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::ConjugateTranspose) => cgemm_nc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::ConjugateTranspose) => cgemm_tc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::NoTranspose)        => cgemm_cn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::Transpose)          => cgemm_ct(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::ConjugateTranspose) => cgemm_cc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}

