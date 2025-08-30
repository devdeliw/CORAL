use crate::level1::assert_length_helpers::required_len_ok; 

pub fn dcopy(n: usize, x: &[f64], incx: isize, y: &mut [f64], incy: isize) {
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");
    debug_assert!(required_len_ok(y.len(), n, incy), "y too short for n/incy");

    unsafe {
        // fast path 
        if incx == 1 && incy == 1 {
            core::ptr::copy(x.as_ptr(), y.as_mut_ptr(), n);
            return;
        }

        // non unit stride 
        let mut ix: isize = if incx >= 0 { 0 } else { (n as isize - 1) * (-incx) };
        let mut iy: isize = if incy >= 0 { 0 } else { (n as isize - 1) * (-incy) };
        for _ in 0..n {
            *y.get_unchecked_mut(iy as usize) = *x.get_unchecked(ix as usize);
            ix += incx;
            iy += incy;
        }
    }
}
