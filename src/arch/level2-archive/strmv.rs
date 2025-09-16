use crate::level2::{
    strlmv::strlmv,
    strumv::strumv,
    enums::{ 
        UpLo, 
        Trans, 
        Diag, 
    }, 
}; 

#[inline]
pub fn strmv ( 
    uplo      : UpLo, 
    trans     : Trans,
    diag      : Diag, 
    n         : usize, 
    a         : &[f32],  
    inc_row_a : isize, 
    inc_col_a : isize, 
    x         : &mut [f32], 
    incx      : isize, 
) { 
    match uplo { 
       UpLo::UpperTriangular => strumv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
       UpLo::LowerTriangular => strlmv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
    }
}

