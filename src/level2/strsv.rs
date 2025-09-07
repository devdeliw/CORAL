use crate::level2::{
    strlsv::strlsv, 
    strusv::strusv, 
    enums::{ 
        UpLo, 
        Trans, 
        Diag, 
    }, 
}; 

#[inline]
pub fn strsv ( 
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
       UpLo::UpperTriangular => strusv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
       UpLo::LowerTriangular => strlsv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
    }
}
