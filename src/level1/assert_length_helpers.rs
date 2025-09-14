//! Checks whether a slice is long enough for a BLAS-style vector operation.
//!
//! Given a slice length `len`, number of elements `n`, and stride `inc`,
//! this returns `true` if the slice contains at least `n` accessible elements.
//! For complex data, each element occupies two indices (real & imag).

#[inline]
pub(crate) fn required_len_ok(len: usize, n: usize, inc: usize) -> bool {
    if n == 0 { return true; }
    len >= (n - 1).saturating_mul(inc) + 1
}


#[inline]
pub(crate) fn required_len_ok_cplx(len: usize, n: usize, inc: usize) -> bool {
    if n == 0 { return true; }
    len >= (n - 1).saturating_mul(inc * 2) + 2 
}
