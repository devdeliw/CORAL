use crate::level2::assert_length_helpers::{
    required_len_ok_vec,
    required_len_ok_mat,
};
use crate::level1::saxpy::saxpy; 


#[inline(always)]
unsafe fn a_ij<'a>(
    a: *mut f32,
    i: isize,
    j: isize,
    inc_row_a: isize,
    inc_col_a: isize,
) -> *mut f32 { unsafe { 
    a.offset(i * inc_row_a + j * inc_col_a)
}}

#[inline]
pub fn sger(
    m          : usize,
    n          : usize,
    alpha      : f32,
    x          : &[f32],
    incx       : isize,
    y          : &[f32],
    incy       : isize,
    a          : &mut [f32],
    inc_row_a  : isize,   // col-major; should be 1 
    inc_col_a  : isize,   // lda > 0
) {
    // quick return 
    if m == 0 || n == 0 { return; }
    if alpha == 0.0 { return; }

    debug_assert!(incx != 0 && incy != 0, "vector increments must be non-zero");
    debug_assert!(inc_row_a != 0 && inc_col_a != 0, "matrix strides must be non-zero");
    debug_assert!(required_len_ok_vec(x.len(), m, incx), "x too short for m/incx");
    debug_assert!(required_len_ok_vec(y.len(), n, incy), "y too short for n/incy");
    debug_assert!(
        required_len_ok_mat(a.len(), m, n, inc_row_a, inc_col_a),
        "a too short for m x n with given strides"
    );

    // fast path 
    if inc_row_a == 1 && inc_col_a > 0 && incx == 1 && incy == 1 {
        let lda = inc_col_a as usize;
        debug_assert!(lda >= m);

        // A[:, j] += (alpha * y[j]) * x
        for j in 0..n {
            let coeff = alpha * unsafe { *y.as_ptr().add(j) };
            if coeff != 0.0 {
                saxpy(m, coeff, x, 1, &mut a[j*lda .. j*lda + m], 1); 
            }
        }
        return;
    }

    // non unit stride 
    unsafe {
        let aptr = a.as_mut_ptr();

        let mut x0 = x.as_ptr();
        if incx < 0 {
            x0 = x0.offset((m as isize - 1) * incx);
        }

        let mut y0 = y.as_ptr();
        if incy < 0 {
            y0 = y0.offset((n as isize - 1) * incy);
        }

        for j in 0..n {
            let yj = *y0.offset(j as isize * incy);
            let coeff = alpha * yj;
            if coeff != 0.0 {
                for i in 0..m {
                    let xi = *x0.offset(i as isize * incx);
                    let aij = a_ij(aptr, i as isize, j as isize, inc_row_a, inc_col_a);
                    *aij += xi * coeff;
                }
            }
        }
    }
}

