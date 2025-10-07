use crate::level3::{
    dgemm_nn::dgemm_nn, 
    dgemm_nt::dgemm_nt, 
    dgemm_tn::dgemm_tn, 
    dgemm_tt::dgemm_tt
};
use crate::level2::enums::CoralTranspose;

pub const MC: usize = 258; 
pub const NC: usize = 384; 
pub const KC: usize = 256;

#[inline(always)]
fn is_transpose(op: CoralTranspose) -> bool {
    match op {
        CoralTranspose::NoTranspose          => false,
        CoralTranspose::Transpose 
        | CoralTranspose::ConjugateTranspose => true, 
    }
}

#[inline(always)]
pub fn dgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
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
    let a_t = is_transpose(op_a);
    let b_t = is_transpose(op_b);

    match (a_t, b_t) {
        (false, false) => dgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (false, true ) => dgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , false) => dgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , true ) => dgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}

