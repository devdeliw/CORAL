use blas_src as _;
use cblas_sys::{
    cblas_dsyr2,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
};

use coral::level2::dsyr2::dsyr2;
use coral::level2::enums::CoralTriangular;

fn cblas_dsyr2_wrapper(
    uplo  : CBLAS_UPLO,
    n     : i32,
    alpha : f64,
    x     : *const f64,
    incx  : i32,
    y     : *const f64,
    incy  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_dsyr2(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            n,
            alpha,
            x,
            incx,
            y,
            incy,
            a,
            lda,
        );
    }
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v = vec![0.0f64; (len_logical - 1) * inc + 1];
    let mut idx = 0usize;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn make_symmetric_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; lda * n];
    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f64;
            let hi  = i.max(j) as f64;
            let val = 0.1 + 0.5 * lo + 0.25 * hi;
            a[i + j * lda] = val;
            a[j + i * lda] = val;
        }
    }
    a
}

fn assert_allclose(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

fn assert_upper_lower_equal(
    au   : &[f64],
    al   : &[f64],
    n    : usize,
    lda  : usize,
    rtol : f64,
    atol : f64,
) {
    assert_eq!(au.len(), al.len());
    for j in 0..n {
        for i in 0..=j {
            let x   = au[i + j * lda];
            let y   = al[j + i * lda];
            let diff= (x - y).abs();
            let tol = atol + rtol * x.abs().max(y.abs());
            assert!(diff <= tol, "mirror mismatch at (i={}, j={}): {} vs {} (|Δ|={}, tol={})", i, j, x, y, diff, tol);
        }
    }
}

fn assert_only_triangle_touched(
    updated : &[f64],
    baseline: &[f64],
    n       : usize,
    lda     : usize,
    upper   : bool,
) {
    assert_eq!(updated.len(), baseline.len());
    for j in 0..n {
        for i in 0..n {
            let idx = i + j * lda;
            let untouched = if upper { i > j } else { i < j };
            if untouched {
                assert!(
                    updated[idx].to_bits() == baseline[idx].to_bits(),
                    "untouched half modified at (i={}, j={}): {} -> {}",
                    i, j, baseline[idx], updated[idx]
                );
            }
        }
    }
}

const RTOL: f64 = 1e-15;
const ATOL: f64 = 1e-15;

#[test]
fn upper_small() {
    let n     = 7usize;
    let lda   = n;
    let alpha = 1.25f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| 0.2  + 0.1  * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i| -0.3 + 0.05 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched(&a_coral, &a0, n, lda, true);
}

#[test]
fn lower_small() {
    let n     = 7usize;
    let lda   = n;
    let alpha = -0.8f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| -0.2 + 0.07  * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i|  0.4 - 0.055 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched(&a_coral, &a0, n, lda, false);
}

#[test]
fn large_upper() {
    let n     = 1024usize;
    let lda   = n;
    let alpha = -0.37f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| 0.05 + 0.002 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i| 0.4  - 0.003 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn large_lower() {
    let n     = 512usize;
    let lda   = n;
    let alpha = 0.93f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| -0.2   + 0.0015 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i|  0.1   + 0.0020 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn lower_padded_strided() {
    let n     = 9usize;
    let lda   = n + 3;
    let alpha = -0.85f64;

    let a0   = make_symmetric_col_major(n, lda);
    let incx = 2usize;
    let incy = 3usize;
    let x    = make_strided_vec(n, incx, |i| 0.05 + 0.03 * (i as f64));
    let y    = make_strided_vec(n, incy, |i| 0.4  - 0.02 * (i as f64));

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        incx,
        &y,
        incy,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        incx as i32,
        y.as_ptr(),
        incy as i32,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched(&a_coral, &a0, n, lda, false);
}

#[test]
fn alpha_zero_does_nothing() {
    let n     = 300usize;
    let lda   = n + 5;
    let alpha = 0.0f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| 0.11 + 0.01 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i| 0.07 - 0.01 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
    assert_allclose(&a_coral, &a0, RTOL, ATOL);
}

#[test]
fn accumulate_twice() {
    let n      = 64usize;
    let lda    = n;
    let alpha1 = 1.2f64;
    let alpha2 = -0.7f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x1 = (0..n).map(|i| 0.2  + 0.01 * (i as f64)).collect::<Vec<_>>();
    let y1 = (0..n).map(|i| -0.1 + 0.02 * (i as f64)).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| -0.3 + 0.03 * (i as f64)).collect::<Vec<_>>();
    let y2 = (0..n).map(|i|  0.4 - 0.01 * (i as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha1,
        &x1,
        1,
        &y1,
        1,
        &mut a_coral,
        lda,
    );
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha2,
        &x2,
        1,
        &y2,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha1,
        x1.as_ptr(),
        1,
        y1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha2,
        x2.as_ptr(),
        1,
        y2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn upper_equals_lower() {
    let n     = 33usize;
    let lda   = n;
    let alpha = 0.77f64;

    let a0 = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|i| 0.2 + 0.005 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|i| 0.1 + 0.004 * (i as f64)).collect::<Vec<_>>();

    let mut a_upper = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_upper,
        lda,
    );

    let mut a_lower = a0.clone();
    dsyr2(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_lower,
        lda,
    );

    assert_upper_lower_equal(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let n     = 0usize;
    let lda   = 1usize;
    let alpha = 0.55f64;

    let a0 = vec![0.0f64; lda];
    let x  = vec![1.0f64; 1];
    let y  = vec![2.0f64; 1];

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

