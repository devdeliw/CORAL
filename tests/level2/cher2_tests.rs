use blas_src as _;
use cblas_sys::{
    cblas_cher2,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
};

use coral::level2::cher2::cher2;
use coral::level2::enums::CoralTriangular;

// cblas wrapper
fn cblas_cher2_wrapper(
    uplo  : CBLAS_UPLO,
    n     : i32,
    alpha : [f32; 2],
    x     : *const f32,
    incx  : i32,
    y     : *const f32,
    incy  : i32,
    a     : *mut f32,
    lda   : i32,
) {
    unsafe {
        cblas_cher2(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            x               as *const [f32; 2],
            incx,
            y               as *const [f32; 2],
            incy,
            a               as *mut   [f32; 2],
            lda,
        );
    }
}

// helpers 
fn make_hermitian_col_major(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f32;
            let hi  = i.max(j) as f32;
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
    f           : impl Fn(usize) -> [f32; 2],
) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * ((len_logical - 1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }
    v
}

fn assert_allclose_c(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
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
    au   : &[f32],
    al   : &[f32],
    n    : usize,
    lda  : usize,
    rtol : f32,
    atol : f32,
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
    updated : &[f32],
    baseline: &[f32],
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

const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-6;

#[test]
fn upper_small() {
    let n     = 7usize;
    let lda   = n;
    let alpha = [1.10f32, -0.35f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.2  + 0.10 * (i as f32), -0.05 + 0.02 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [-0.3 + 0.05 * (i as f32),  0.07 - 0.01 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, true);
}

#[test]
fn lower_small() {
    let n     = 7usize;
    let lda   = n;
    let alpha = [-0.80f32, 0.50f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [-0.2 + 0.07  * (i as f32), 0.06 - 0.01 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [ 0.4 - 0.055 * (i as f32), -0.03 + 0.02 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, false);
}

#[test]
fn large_upper() {
    let n     = 1024usize;
    let lda   = n;
    let alpha = [-0.37f32, 0.20f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.05 + 0.002 * (i as f32), -0.02 + 0.0007 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [0.40 - 0.003 * (i as f32),  0.01 - 0.0005 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn large_lower() {
    let n     = 512usize;
    let lda   = n;
    let alpha = [0.93f32, -0.10f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [-0.20 + 0.0015 * (i as f32), 0.03 - 0.0004 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [ 0.10 + 0.0020 * (i as f32), -0.01 + 0.0003 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn lower_padded_strided() {
    let n     = 9usize;
    let lda   = n + 3;
    let alpha = [-0.85f32, 0.25f32];

    let a0   = make_hermitian_col_major(n, lda);
    let incx = 2usize;
    let incy = 3usize;
    let x    = make_strided_cvec(n, incx, |i| [0.05 + 0.03 * (i as f32), -0.02 + 0.01 * (i as f32)]);
    let y    = make_strided_cvec(n, incy, |i| [0.40 - 0.02 * (i as f32),  0.03 - 0.01 * (i as f32)]);

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_only_triangle_touched_c(&a_coral, &a0, n, lda, false);
}

#[test]
fn alpha_zero_does_nothing() {
    let n     = 300usize;
    let lda   = n + 5;
    let alpha = [0.0f32, 0.0f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.11 + 0.01 * (i as f32), -0.07 + 0.004 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [0.08 - 0.01 * (i as f32),  0.02 + 0.003 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_allclose_c(&a_coral, &a0, RTOL, ATOL);
}

#[test]
fn accumulate_twice() {
    let n      = 64usize;
    let lda    = n;
    let alpha1 = [1.20f32, 0.30f32];
    let alpha2 = [-0.70f32, -0.20f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x1 = (0..n)
        .flat_map(|i| [0.20 + 0.01 * (i as f32), -0.03 + 0.005 * (i as f32)])
        .collect::<Vec<_>>();
    let y1 = (0..n)
        .flat_map(|i| [-0.10 + 0.02 * (i as f32), 0.04 - 0.003 * (i as f32)])
        .collect::<Vec<_>>();
    let x2 = (0..n)
        .flat_map(|i| [-0.30 + 0.03 * (i as f32), 0.06 - 0.004 * (i as f32)])
        .collect::<Vec<_>>();
    let y2 = (0..n)
        .flat_map(|i| [0.40 - 0.01 * (i as f32), -0.02 + 0.002 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher2(
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
    cher2(
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
    cblas_cher2_wrapper(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn upper_equals_lower() {
    let n     = 33usize;
    let lda   = n;
    let alpha = [0.77f32, 0.11f32];

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.2 + 0.005 * (i as f32), -0.04 + 0.002 * (i as f32)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [0.1 + 0.004 * (i as f32),  0.03 - 0.001 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_upper = a0.clone();
    cher2(
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
    cher2(
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

    assert_upper_lower_equal_c(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let n     = 0usize;
    let lda   = 1usize;
    let alpha = [0.55f32, -0.10f32];

    let a0 = vec![0.0f32; 2 * lda];
    let x  = vec![0.0f32; 2];
    let y  = vec![0.0f32; 2];

    let mut a_coral = a0.clone();
    cher2(
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
    cblas_cher2_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

