#[inline(always)]
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p  = p.add(1);
                i += 1;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let mut p = x.as_mut_ptr();
            let step = incx as usize;
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(step);
                i += 1;
            }
        }
    }
}
