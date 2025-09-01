//! Checks whether a slice is long enough for a BLAS-style vector operation.
//!
//! Given a slice length `len`, number of elements `n`, and stride `inc`,
//! this returns `true` if the slice contains at least `n` accessible elements.
//! For complex data, each element occupy two slots (real + imag).

#[inline]
pub(crate) fn required_len_ok(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = if inc >= 0 { inc as usize } else { (-inc) as usize };
    len >= (n - 1).saturating_mul(step) + 1
}


#[inline]
pub(crate) fn required_len_ok_cplx(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = (if inc >= 0 { inc as usize } else { (-inc) as usize }) * 2;
    len >= (n - 1).saturating_mul(step) + 2 
}
