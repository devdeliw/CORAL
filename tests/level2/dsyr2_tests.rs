use blas_src as _;
use cblas_sys::{ cblas_dsyr2, CBLAS_LAYOUT, CBLAS_UPLO };

use coral::level2::dsyr2::dsyr2;
use coral::enums::CoralTriangular;

fn cblas_dsyr2_ref(
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

#[inline(always)]
fn to_uplo(
    tri : CoralTriangular,
) -> CBLAS_UPLO {
    match tri {
        CoralTriangular::UpperTriangular => CBLAS_UPLO::CblasUpper,
        CoralTriangular::LowerTriangular => CBLAS_UPLO::CblasLower,
    }
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v = vec![0.0; (len_logical - 1) * inc + 1];
    let mut idx = 0;
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
    let mut a = vec![0.0; lda * n];
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
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
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
            let x    = au[i + j * lda];
            let y    = al[j + i * lda];
            let diff = (x - y).abs();
            let tol  = atol + rtol * x.abs().max(y.abs());
            assert!(diff <= tol, "mirror mismatch at (i={i}, j={j}): {x} vs {y} (delta={diff}, tol={tol})");
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
                    "untouched half modified at (i={i}, j={j}): {} -> {}",
                    baseline[idx], updated[idx]
                );
            }
        }
    }
}

const RTOL: f64 = 1e-6;
const ATOL: f64 = 1e-5;

fn run_case(
    tri           : CoralTriangular,
    n             : usize,
    lda           : usize,
    incx          : usize,
    incy          : usize,
    alpha         : f64,
    xgen          : impl Fn(usize) -> f64,
    ygen          : impl Fn(usize) -> f64,
    check_touched : bool,
) {
    let a0 = make_symmetric_col_major(n, lda);
    let x  = make_strided_vec(n, incx, xgen);
    let y  = make_strided_vec(n, incy, ygen);

    let mut a_coral = a0.clone();
    dsyr2(
        tri,
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
    cblas_dsyr2_ref(
        to_uplo(tri),
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
    if check_touched {
        assert_only_triangle_touched(
            &a_coral,
            &a0,
            n,
            lda,
            matches!(tri, CoralTriangular::UpperTriangular),
        );
    }
}

fn run_small() {
    run_case(
        CoralTriangular::UpperTriangular,
        7,
        7,
        1,
        1,
        1.25,
        |i| 0.2 + 0.1 * (i as f64),
        |i| -0.3 + 0.05 * (i as f64),
        true,
    );

    run_case(
        CoralTriangular::LowerTriangular,
        7,
        7,
        1,
        1,
        -0.8,
        |i| -0.2 + 0.07 * (i as f64),
        |i|  0.4 - 0.055 * (i as f64),
        true,
    );
}

fn run_large() {
    run_case(
        CoralTriangular::UpperTriangular,
        1024,
        1024,
        1,
        1,
        -0.37,
        |i| 0.05 + 0.002 * (i as f64),
        |i| 0.4  - 0.003 * (i as f64),
        false,
    );

    run_case(
        CoralTriangular::LowerTriangular,
        512,
        512,
        1,
        1,
        0.93,
        |i| -0.2 + 0.0015 * (i as f64),
        |i|  0.1 + 0.0020 * (i as f64),
        false,
    );
}

fn run_padded_strided() {
    run_case(
        CoralTriangular::LowerTriangular,
        9,
        9 + 3,
        2,
        3,
        -0.85,
        |i| 0.05 + 0.03 * (i as f64),
        |i| 0.4  - 0.02 * (i as f64),
        true,
    );
}

fn run_accumulate_twice() {
    let n   = 64;
    let lda = n;

    let a0 = make_symmetric_col_major(n, lda);
    let x1 = make_strided_vec(n, 1, |i| 0.2  + 0.01 * (i as f64));
    let y1 = make_strided_vec(n, 1, |i| -0.1 + 0.02 * (i as f64));
    let x2 = make_strided_vec(n, 1, |i| -0.3 + 0.03 * (i as f64));
    let y2 = make_strided_vec(n, 1, |i|  0.4 - 0.01 * (i as f64));

    let mut a_coral = a0.clone();
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        1.2,
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
        -0.7,
        &x2,
        1,
        &y2,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dsyr2_ref(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        1.2,
        x1.as_ptr(),
        1,
        y1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_dsyr2_ref(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        -0.7,
        x2.as_ptr(),
        1,
        y2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

fn run_upper_equals_lower() {
    let n   = 33;
    let lda = n;

    let x = make_strided_vec(n, 1, |i| 0.2 + 0.005 * (i as f64));
    let y = make_strided_vec(n, 1, |i| 0.1 + 0.004 * (i as f64));

    let mut a_upper = make_symmetric_col_major(n, lda);
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        0.77,
        &x,
        1,
        &y,
        1,
        &mut a_upper,
        lda,
    );

    let mut a_lower = make_symmetric_col_major(n, lda);
    dsyr2(
        CoralTriangular::LowerTriangular,
        n,
        0.77,
        &x,
        1,
        &y,
        1,
        &mut a_lower,
        lda,
    );

    assert_upper_lower_equal(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

fn run_alpha_zero() {
    run_case(
        CoralTriangular::UpperTriangular,
        300,
        300 + 5,
        1,
        1,
        0.0,
        |i| 0.11 + 0.01 * (i as f64),
        |i| 0.07 - 0.01 * (i as f64),
        false,
    );
}

fn run_quick_return_n_zero() {
    let n   = 0;
    let lda = 1;

    let x = vec![1.0];
    let y = vec![2.0];

    let mut a_coral: Vec<f64> = vec![0.0; lda];
    dsyr2(
        CoralTriangular::UpperTriangular,
        n,
        0.55,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = vec![0.0; lda];
    cblas_dsyr2_ref(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        0.55,
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
fn main_suites() {
    run_small();
    run_large();
    run_padded_strided();
    run_accumulate_twice();
}

#[test]
fn quick_returns() {
    run_alpha_zero();
    run_quick_return_n_zero();
}

#[test]
fn upper_equals_lower() { 
    run_upper_equals_lower();
}   


