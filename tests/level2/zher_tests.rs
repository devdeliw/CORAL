use blas_src as _;
use cblas_sys::{
    cblas_zher,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
};

use coral::level2::zher::zher;
use coral::level2::enums::CoralTriangular;

// cblas wrapper
fn cblas_zher_wrapper(
    uplo  : CBLAS_UPLO,
    n     : i32,
    alpha : f64,
    x     : *const f64,
    incx  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_zher(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            n,
            alpha,
            x       as *const [f64; 2],
            incx,
            a       as *mut [f64; 2],
            lda,
        );
    }
}

// helpers 
fn make_hermitian_col_major(n: usize, lda: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; 2 * lda * n];
    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f64;
            let hi  = i.max(j) as f64;
            let re  = 0.10 + 0.25 * lo + 0.15 * hi;
            let im  = if i == j { 0.0 } else { 0.04 + 0.02 * (hi - lo) };

            let idx_ij = 2 * (i + j * lda);
            a[idx_ij]     = re;
            a[idx_ij + 1] = im;

            let idx_ji = 2 * (j + i * lda);
            a[idx_ji]     = re;
            a[idx_ji + 1] = -im; // conjugate
        }
    }
    a
}

fn make_strided_cvec(
    len_logical : usize,
    inc         : usize,                
    f           : impl Fn(usize) -> [f64; 2],
) -> Vec<f64> {
    let mut v = vec![0.0f64; 2 * ((len_logical - 1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }
    v
}

fn assert_allclose_c(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

fn assert_upper_lower_equal_c(
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
            let idx_u = 2 * (i + j * lda);
            let idx_l = 2 * (j + i * lda);

            let xr = au[idx_u];
            let xi = au[idx_u + 1];
            let yr = al[idx_l];
            let yi = al[idx_l + 1];

            let dr = (xr - yr).abs();
            let di = (xi - (-yi)).abs(); // conjugate match

            let tol_r = atol + rtol * xr.abs().max(yr.abs());
            let tol_i = atol + rtol * xi.abs().max(yi.abs());

            assert!(dr <= tol_r, "mirror re mismatch at (i={}, j={}): {} vs {} (|Δ|={}, tol={})", i, j, xr, yr, dr, tol_r);
            assert!(di <= tol_i, "mirror im mismatch at (i={}, j={}): {} vs -{} (|Δ|={}, tol={})", i, j, xi, yi, di, tol_i);
        }
    }
}

fn assert_only_triangle_touched_c(
    updated : &[f64],
    baseline: &[f64],
    n       : usize,
    lda     : usize,
    upper   : bool,
) {
    assert_eq!(updated.len(), baseline.len());
    for j in 0..n {
        for i in 0..n {
            let untouched = if upper { i > j } else { i < j };
            if untouched {
                let idx = 2 * (i + j * lda);
                assert!(
                    updated[idx].to_bits() == baseline[idx].to_bits()
                        && updated[idx + 1].to_bits() == baseline[idx + 1].to_bits(),
                    "untouched half modified at (i={}, j={}): ({}, {}) -> ({}, {})",
                    i, j, baseline[idx], baseline[idx + 1], updated[idx], updated[idx + 1]
                );
            }
        }
    }
}

const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-13;

#[test]
fn upper_small() {
    let n    = 6usize;
    let lda  = n;
    let alpha= 0.9f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.2 + 0.1 * (i as f64), -0.05 + 0.02 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, true);
}

#[test]
fn lower_small() {
    let n    = 7usize;
    let lda  = n;
    let alpha= -0.8f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [-0.1 + 0.03 * (i as f64), 0.07 - 0.01 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, false);
}

#[test]
fn large_upper() {
    let n    = 512usize;
    let lda  = n;
    let alpha= -0.37f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.03 + 0.001 * (i as f64), -0.02 + 0.0007 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn large_lower() {
    let n    = 256usize;
    let lda  = n;
    let alpha= 0.93f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [-0.02 + 0.0005 * (i as f64), 0.01 - 0.0003 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn lower_padded() {
    let n    = 9usize;
    let lda  = n + 3;
    let alpha= -0.85f64;

    let a0   = make_hermitian_col_major(n, lda);
    let incx = 2usize; // non unit stride 
    let x    = make_strided_cvec(n, incx, |i| [0.05 + 0.03 * (i as f64), -0.02 + 0.01 * (i as f64)]);

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        incx,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasLower,
        n as i32,
        alpha,
        x.as_ptr(),
        incx as i32,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, false);
}

#[test]
fn alpha_zero_does_nothing() {
    let n    = 64usize;
    let lda  = n + 5;
    let alpha= 0.0f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.11 + 0.01 * (i as f64), -0.07 + 0.004 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_allclose_c(&a_coral, &a0, RTOL, ATOL);
}

#[test]
fn accumulate_twice() {
    let n     = 33usize;
    let lda   = n;
    let alpha1= 1.2f64;
    let alpha2= -0.7f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x1 = (0..n)
        .flat_map(|i| [0.20  + 0.01 * (i as f64), -0.03 + 0.005 * (i as f64)])
        .collect::<Vec<_>>();
    let x2 = (0..n)
        .flat_map(|i| [-0.30 + 0.03 * (i as f64),  0.06 - 0.004 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha1,
        &x1,
        1,
        &mut a_coral,
        lda,
    );
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha2,
        &x2,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha1,
        x1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha2,
        x2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn upper_equals_lower() {
    let n    = 25usize;
    let lda  = n;
    let alpha= 0.77f64;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.2 + 0.005 * (i as f64), -0.04 + 0.002 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_upper = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_upper,
        lda,
    );

    let mut a_lower = a0.clone();
    zher(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_lower,
        lda,
    );

    assert_upper_lower_equal_c(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let n    = 0usize;
    let lda  = 1usize;
    let alpha= 0.55f64;

    let a0 = vec![0.0f64; 2 * lda];
    let x  = vec![0.0f64; 2];

    let mut a_coral = a0.clone();
    zher(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

