use blas_src as _;
use cblas_sys::{ cblas_zher2, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::level2::zher2::zher2;
use coral::enums::CoralTriangular;

fn cblas_zher2_wrapper(
    uplo  : CBLAS_UPLO,
    n     : i32,
    alpha : [f64; 2],
    x     : *const f64,
    incx  : i32,
    y     : *const f64,
    incy  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_zher2(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            x               as *const [f64; 2],
            incx,
            y               as *const [f64; 2],
            incy,
            a               as *mut   [f64; 2],
            lda,
        );
    }
}

fn make_hermitian_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0; 2 * lda * n];

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
            a[idx_ji + 1] = -im;
        }
    }

    a
}

fn make_strided_cvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f64; 2],
) -> Vec<f64> {
    let mut v = vec![0.0; 2 * ((len_logical - 1) * inc + 1)];

    let mut idx = 0;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }

    v
}

fn assert_allclose_c(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol, 
            "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})"
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
            let di = (xi - (-yi)).abs();

            let tol_r = atol + rtol * xr.abs().max(yr.abs());
            let tol_i = atol + rtol * xi.abs().max(yi.abs());

            assert!(
                dr <= tol_r, 
                "mirror re mismatch at (i={i}, j={j}): {xr} vs {yr} (delta={dr}, tol={tol_r})"
            );
            assert!(
                di <= tol_i,
                "mirror im mismatch at (i={i}, j={j}): {xi} vs -{yi} (delta={di}, tol={tol_i})"
            );
        }
    }
}

fn assert_only_triangle_touched_c(
    updated  : &[f64],
    baseline : &[f64],
    n        : usize,
    lda      : usize,
    upper    : bool,
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
                    "untouched half modified at (i={i}, j={j}): ({}, {}) -> ({}, {})",
                    baseline[idx], baseline[idx + 1], updated[idx], updated[idx + 1]
                );
            }
        }
    }
}

const RTOL: f64 = 1e-13;
const ATOL: f64 = 1e-13;

#[derive(Copy, Clone)]
enum Tri { Upper, Lower }

fn flags_for(
    tri : Tri,
) -> (CoralTriangular, CBLAS_UPLO, bool) {
    match tri {
        Tri::Upper => (
            CoralTriangular::UpperTriangular,
            CBLAS_UPLO::CblasUpper,
            true,
        ),
        Tri::Lower => (
            CoralTriangular::LowerTriangular,
            CBLAS_UPLO::CblasLower,
            false,
        ),
    }
}

fn run_case(
    tri   : Tri,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : [f64; 2],
    xgen  : impl Fn(usize) -> [f64; 2],
    ygen  : impl Fn(usize) -> [f64; 2],
    check_triangle : bool,
) {
    let (c_tri, uplo, is_upper) = flags_for(tri);

    let a0 = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(
        n,
        incx,
        xgen,
    );
    let y  = make_strided_cvec(
        n,
        incy,
        ygen,
    );

    let mut a_coral = a0.clone();
    zher2(
        c_tri,
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
    cblas_zher2_wrapper(
        uplo,
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

    if check_triangle {
        assert_only_triangle_touched_c(&a_coral, &a0, n, lda, is_upper);
    }
}

fn run_small_and_large(
    tri : Tri,
) {
    match tri {
        Tri::Upper => {
            run_case(
                Tri::Upper,
                7,
                7,
                1,
                1,
                [1.10, -0.35],
                |i| [0.2 + 0.10 * (i as f64), -0.05 + 0.02 * (i as f64)],
                |i| [-0.3 + 0.05 * (i as f64),  0.07 - 0.01 * (i as f64)],
                true,
            );
            run_case(
                Tri::Upper,
                1024,
                1024,
                1,
                1,
                [-0.37, 0.20],
                |i| [0.05 + 0.002 * (i as f64), -0.02 + 0.0007 * (i as f64)],
                |i| [0.40 - 0.003 * (i as f64),  0.01 - 0.0005 * (i as f64)],
                false,
            );
        }
        Tri::Lower => {
            run_case(
                Tri::Lower,
                7,
                7,
                1,
                1,
                [-0.80, 0.50],
                |i| [-0.2 + 0.07  * (i as f64), 0.06 - 0.01 * (i as f64)],
                |i| [ 0.4 - 0.055 * (i as f64), -0.03 + 0.02 * (i as f64)],
                true,
            );
            run_case(
                Tri::Lower,
                512,
                512,
                1,
                1,
                [0.93, -0.10],
                |i| [-0.20 + 0.0015 * (i as f64), 0.03 - 0.0004 * (i as f64)],
                |i| [ 0.10 + 0.0020 * (i as f64), -0.01 + 0.0003 * (i as f64)],
                false,
            );
        }
    }
}

fn run_lower_padded_strided() {
    let n     = 9;
    let lda   = n + 3;
    let incx  = 2;
    let incy  = 3;
    run_case(
        Tri::Lower,
        n,
        lda,
        incx,
        incy,
        [-0.85, 0.25],
        |i| [0.05 + 0.03 * (i as f64), -0.02 + 0.01 * (i as f64)],
        |i| [0.40 - 0.02 * (i as f64),  0.03 - 0.01 * (i as f64)],
        true,
    );
}

fn run_alpha_zero_does_nothing() {
    let n   = 300;
    let lda = n + 5;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.11 + 0.01 * (i as f64), -0.07 + 0.004 * (i as f64)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [0.08 - 0.01 * (i as f64),  0.02 + 0.003 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher2(
        CoralTriangular::UpperTriangular,
        n,
        [0.0, 0.0],
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        [0.0, 0.0],
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

fn run_accumulate_twice() {
    let n   = 64;
    let lda = n;

    let a0 = make_hermitian_col_major(n, lda);
    let x1 = (0..n)
        .flat_map(|i| [0.20 + 0.01 * (i as f64), -0.03 + 0.005 * (i as f64)])
        .collect::<Vec<_>>();
    let y1 = (0..n)
        .flat_map(|i| [-0.10 + 0.02 * (i as f64), 0.04 - 0.003 * (i as f64)])
        .collect::<Vec<_>>();
    let x2 = (0..n)
        .flat_map(|i| [-0.30 + 0.03 * (i as f64), 0.06 - 0.004 * (i as f64)])
        .collect::<Vec<_>>();
    let y2 = (0..n)
        .flat_map(|i| [0.40 - 0.01 * (i as f64), -0.02 + 0.002 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    zher2(
        CoralTriangular::UpperTriangular,
        n,
        [1.20, 0.30],
        &x1,
        1,
        &y1,
        1,
        &mut a_coral,
        lda,
    );
    zher2(
        CoralTriangular::UpperTriangular,
        n,
        [-0.70, -0.20],
        &x2,
        1,
        &y2,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        [1.20, 0.30],
        x1.as_ptr(),
        1,
        y1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_zher2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        [-0.70, -0.20],
        x2.as_ptr(),
        1,
        y2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

fn run_upper_equals_lower() {
    let n   = 33;
    let lda = n;

    let a0 = make_hermitian_col_major(n, lda);
    let x  = (0..n)
        .flat_map(|i| [0.2 + 0.005 * (i as f64), -0.04 + 0.002 * (i as f64)])
        .collect::<Vec<_>>();
    let y  = (0..n)
        .flat_map(|i| [0.1 + 0.004 * (i as f64),  0.03 - 0.001 * (i as f64)])
        .collect::<Vec<_>>();

    let mut a_upper = a0.clone();
    zher2(
        CoralTriangular::UpperTriangular,
        n,
        [0.77, 0.11],
        &x,
        1,
        &y,
        1,
        &mut a_upper,
        lda,
    );

    let mut a_lower = a0.clone();
    zher2(
        CoralTriangular::LowerTriangular,
        n,
        [0.77, 0.11],
        &x,
        1,
        &y,
        1,
        &mut a_lower,
        lda,
    );

    assert_upper_lower_equal_c(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

fn run_quick_return_n_zero() {
    let n   = 0;
    let lda = 1;

    let a0 = vec![0.0; 2 * lda];
    let x  = vec![0.0; 2];
    let y  = vec![0.0; 2];

    let mut a_coral = a0.clone();
    zher2(
        CoralTriangular::UpperTriangular,
        n,
        [0.55, -0.10],
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_zher2_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        [0.55, -0.10],
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
fn zher2_upper_suites() {
    run_small_and_large(Tri::Upper);
    run_alpha_zero_does_nothing();
    run_accumulate_twice();
    run_quick_return_n_zero();
}

#[test]
fn zher2_lower_suites() {
    run_small_and_large(Tri::Lower);
    run_lower_padded_strided();
}

#[test]
fn zher2_upper_equals_lower() {
    run_upper_equals_lower();
}
