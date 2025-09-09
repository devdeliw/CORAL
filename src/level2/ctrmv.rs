use crate::level2::{
    enums::{
        Diag,
        Trans,
        UpLo,
    }, 
    ctrlmv::ctrlmv, 
    ctrumv::ctrumv, 
}; 

#[inline]
pub fn ctrmv(
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
        UpLo::LowerTriangular => ctrlmv(
            n,
            diag,
            trans,
            a,
            inc_row_a,
            inc_col_a,
            x,
            incx,
        ),
        UpLo::UpperTriangular => ctrumv(
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

