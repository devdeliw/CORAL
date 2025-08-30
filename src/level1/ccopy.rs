use crate::level1::assert_length_helpers::required_len_ok_cplx; 

pub fn ccopy(n: usize, x: &[f32], incx: isize, y: &mut [f32], incy: isize) {
    // quick return 
    if n == 0 { return; }

    debug_assert!(incx != 0 && incy != 0, "BLAS increments must be non-zero");
    debug_assert!(required_len_ok_cplx(x.len(), n, incx), "x too short for n/incx (complex)");
    debug_assert!(required_len_ok_cplx(y.len(), n, incy), "y too short for n/incy (complex)");

    unsafe {
        if incx == 1 && incy == 1 {
            core::ptr::copy(x.as_ptr(), y.as_mut_ptr(), 2 * n);
            return;
        }

        let stepx: isize = incx * 2; 
        let stepy: isize = incy * 2;
        let mut ix: isize = if stepx >= 0 { 0 } else { (n as isize - 1) * (-stepx) };
        let mut iy: isize = if stepy >= 0 { 0 } else { (n as isize - 1) * (-stepy) };
        for _ in 0..n {
            let ixi = ix as usize;
            let iyi = iy as usize;
            *y.get_unchecked_mut(iyi)     = *x.get_unchecked(ixi);
            *y.get_unchecked_mut(iyi + 1) = *x.get_unchecked(ixi + 1);
            ix += stepx;
            iy += stepy;
        }
    }
}
