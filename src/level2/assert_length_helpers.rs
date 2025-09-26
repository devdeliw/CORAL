/// Checks if a column-major matrix buffer of length `len`
/// can hold an `n_rows x n_cols` panel with leading dimension `lda`.
#[inline(always)]
pub(crate) fn required_len_ok_matrix(
    len     : usize,   
    n_rows  : usize,
    n_cols  : usize,
    lda     : usize,
) -> bool {
    if n_rows == 0 || n_cols == 0 {
        // empty matrix valid
        true 
    } else {
        len >= (n_cols - 1) * lda + n_rows
    }
}

/// Checks if a complex column-major matrix buffer of length `len`
/// can hold an `n_rows x n_cols` panel with leading dimension `lda`.
#[inline(always)]
pub(crate) fn required_len_ok_matrix_cplx(
    len     : usize,   
    n_rows  : usize,
    n_cols  : usize,
    lda     : usize,
) -> bool {
    if n_rows == 0 || n_cols == 0 {
        // empty matrix valid
        true 
    } else {
        len >= 2 * (n_cols - 1) * lda + 2 * n_rows
    }
}


