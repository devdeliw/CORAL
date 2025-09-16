use crate::level2::{
    dtrlmv::dtrlmv,
    dtrumv::dtrumv,
    enums::{
        UpLo,
        Trans,
        Diag,
    },
};

#[inline]
pub fn dtrmv(
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
        UpLo::UpperTriangular => dtrumv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx),
        UpLo::LowerTriangular => dtrlmv(n, diag, trans, a, inc_row_a, inc_col_a, x, incx),
    }
}

