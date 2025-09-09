use crate::level2::{
    enums::{
        Diag,
        Trans,
        UpLo,
    }, 
    ztrlmv::ztrlmv, 
    ztrumv::ztrumv, 
}; 

#[inline]
pub fn ztrmv(
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
        UpLo::LowerTriangular => ztrlmv(
            n,
            diag,
            trans,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
        UpLo::UpperTriangular => ztrumv(
            n,
            diag,
            trans,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
    }
}


