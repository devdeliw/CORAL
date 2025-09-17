use crate::level2::{
    enums::{CoralTriangular, CoralDiagonal, CoralTranspose}, 
    strlsv::strlsv, 
    strusv::strusv,
}; 

#[inline] 
pub fn strsv( 
    uplo        : CoralTriangular, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    n           : usize, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    match uplo { 
        CoralTriangular::UpperTriangular => strusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => strlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}
