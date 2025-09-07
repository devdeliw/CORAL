#[inline(always)]
pub(crate) fn required_len_ok_vec(len: usize, n: usize, inc: isize) -> bool {
    if n == 0 { return true; }
    let step = inc.unsigned_abs() as usize;
    1 + (n - 1) * step <= len
}

#[inline(always)]
pub(crate) fn required_len_ok_mat(len: usize, m: usize, n: usize, rs: isize, cs: isize) -> bool {
    let (rmin, rmax) = if rs >= 0 { (0, (m - 1) as isize * rs) } else { ((m - 1) as isize * rs, 0) };
    let (cmin, cmax) = if cs >= 0 { (0, (n - 1) as isize * cs) } else { ((n - 1) as isize * cs, 0) };
    let min = rmin + cmin;
    let max = rmax + cmax;
    min >= 0 && (max as usize) < len
}
