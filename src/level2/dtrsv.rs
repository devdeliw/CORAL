use crate::level2::{
    dtrlsv::dtrlsv, 
    dtrusv::dtrusv, 
    enums::{ 
        UpLo, 
        Trans, 
        Diag, 
    }, 
}; 

#[inline]
pub fn dtrsv ( 
    uplo      : UpLo, 
    trans     : Trans,
    diag      : Diag, 
    n         : usize, 
    a         : &[f64],  
    inc_row_a : isize, 
    inc_col_a : isize, 
    x         : &mut [f64], 
    incx      : isize, 
) { 
    match uplo { 
       UpLo::UpperTriangular => dtrusv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
       UpLo::LowerTriangular => dtrlsv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx), 
    }
}

