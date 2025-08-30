#[inline]
pub(crate) fn required_len_ok(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = if inc >= 0 { inc as usize } else { (-inc) as usize };
    len >= 1 + (n - 1).saturating_mul(step)
}


#[inline]
pub(crate) fn required_len_ok_cplx(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = (if inc >= 0 { inc as usize } else { (-inc) as usize }) * 2;
    len >= (n - 1).saturating_mul(step) + 2
}
