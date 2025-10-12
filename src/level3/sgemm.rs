use crate::level3::{
    sgemm_nn::sgemm_nn, 
    sgemm_nt::sgemm_nt, 
    sgemm_tn::sgemm_tn, 
    sgemm_tt::sgemm_tt
};
use crate::enums::CoralTranspose;

pub const MC: usize = 384; 
pub const NC: usize = 576; 
pub const KC: usize = 256;

#[inline(always)]
fn is_transpose(op: CoralTranspose) -> bool {
    match op {
        CoralTranspose::NoTranspose        => false,
        CoralTranspose::Transpose          => true, 
        CoralTranspose::ConjugateTranspose => true, 
    }
}

#[inline(always)]
pub fn sgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : f32,
    a     : *const f32,
    lda   : usize,
    b     : *const f32,
    ldb   : usize,
    beta  : f32,
    c     : *mut f32,
    ldc   : usize,
) {
    let a_t = is_transpose(op_a);
    let b_t = is_transpose(op_b);

    match (a_t, b_t) {
        (false, false) => sgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (false, true ) => sgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , false) => sgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , true ) => sgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}


