use blas_src as _;
use cblas_sys::{ cblas_cher, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::level2::cher::cher;
use coral::enums::CoralTriangular;

fn cblas_cher_wrapper(
    uplo  : CBLAS_UPLO,
    n     : i32,
    alpha : f32,
    x     : *const f32,
    incx  : i32,
    a     : *mut f32,
    lda   : i32,
) {
    unsafe {
        cblas_cher(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            n,
            alpha,
            x       as *const [f32; 2],
            incx,
            a       as *mut   [f32; 2],
            lda,
        );
    }
}

fn make_hermitian_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
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
            a[idx_ji + 1] = -im;
        }
    }
    a
}

fn make_strided_cvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f32; 2],
) -> Vec<f32> {
    let mut v = vec![0.0; 2 * ((len_logical - 1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }
    v
}

fn assert_allclose_c(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol, 
            "mismatch at {i}: {x} vs {y} delta={diff}, tol={tol})"
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
    updated  : &[f32],
    baseline : &[f32],
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

const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-6;

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
    alpha : f32,
    xgen  : impl Fn(usize) -> [f32; 2],
    check_triangle : bool,
) {
    let (c_tri, uplo, is_upper) = flags_for(tri);

    let a0 = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(
        n,
        incx,
        xgen,
    );

    let mut a_coral = a0.clone();
    cher(
        c_tri,
        n,
        alpha,
        &x,
        incx,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_cher_wrapper(
        uplo,
        n as i32,
        alpha,
        x.as_ptr(),
        incx as i32,
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
                6,
                6,
                1,
                0.9,
                |i| [0.2 + 0.1 * (i as f32), -0.05 + 0.02 * (i as f32)],
                true,
            );
            run_case(
                Tri::Upper,
                512,
                512,
                1,
                -0.37,
                |i| [0.03 + 0.001 * (i as f32), -0.02 + 0.0007 * (i as f32)],
                false,
            );
        }
        Tri::Lower => {
            run_case(
                Tri::Lower,
                7,
                7,
                1,
                -0.8,
                |i| [-0.1 + 0.03 * (i as f32), 0.07 - 0.01 * (i as f32)],
                true,
            );
            run_case(
                Tri::Lower,
                256,
                256,
                1,
                0.93,
                |i| [-0.02 + 0.0005 * (i as f32), 0.01 - 0.0003 * (i as f32)],
                false,
            );
        }
    }
}

fn run_lower_padded_strided() {
    let n    = 9;
    let lda  = n + 3;
    let incx = 2;
    run_case(
        Tri::Lower,
        n,
        lda,
        incx,
        -0.85,
        |i| [0.05 + 0.03 * (i as f32), -0.02 + 0.01 * (i as f32)],
        true,
    );
}

fn run_alpha_zero_upper() {
    let n   = 64;
    let lda = n + 5;
    let a0  = make_hermitian_col_major(n, lda);
    let x   = (0..n)
        .flat_map(|i| [0.11 + 0.01 * (i as f32), -0.07 + 0.004 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher(
        CoralTriangular::UpperTriangular,
        n,
        0.0,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_cher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        0.0,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    assert_allclose_c(&a_coral, &a0, RTOL, ATOL);
}

fn run_accumulate_twice_upper() {
    let n    = 33;
    let lda  = n;
    let a0   = make_hermitian_col_major(n, lda);
    let x1   = (0..n)
        .flat_map(|i| [0.20 + 0.01 * (i as f32), -0.03 + 0.005 * (i as f32)])
        .collect::<Vec<_>>();
    let x2   = (0..n)
        .flat_map(|i| [-0.30 + 0.03 * (i as f32), 0.06 - 0.004 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    cher(
        CoralTriangular::UpperTriangular,
        n,
        1.2,
        &x1,
        1,
        &mut a_coral,
        lda,
    );
    cher(
        CoralTriangular::UpperTriangular,
        n,
        -0.7,
        &x2,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_cher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        1.2,
        x1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_cher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        -0.7,
        x2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

fn run_upper_equals_lower() {
    let n   = 25;
    let lda = n;
    let a0  = make_hermitian_col_major(n, lda);
    let x   = (0..n)
        .flat_map(|i| [0.2 + 0.005 * (i as f32), -0.04 + 0.002 * (i as f32)])
        .collect::<Vec<_>>();

    let mut a_upper = a0.clone();
    cher(
        CoralTriangular::UpperTriangular,
        n,
        0.77,
        &x,
        1,
        &mut a_upper,
        lda,
    );

    let mut a_lower = a0.clone();
    cher(
        CoralTriangular::LowerTriangular,
        n,
        0.77,
        &x,
        1,
        &mut a_lower,
        lda,
    );

    assert_upper_lower_equal_c(&a_upper, &a_lower, n, lda, RTOL, ATOL);
}

fn run_quick_return_n_zero() {
    let n   = 0;
    let lda = 1;
    let a0  = vec![0.0; 2 * lda];
    let x   = vec![0.0; 2];

    let mut a_coral = a0.clone();
    cher(
        CoralTriangular::UpperTriangular,
        n,
        0.55,
        &x,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_cher_wrapper(
        CBLAS_UPLO::CblasUpper,
        n as i32,
        0.55,
        x.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn upper_suites() {
    run_small_and_large(Tri::Upper);
    run_alpha_zero_upper();
    run_accumulate_twice_upper();
    run_quick_return_n_zero();
}

#[test]
fn lower_suites() {
    run_small_and_large(Tri::Lower);
    run_lower_padded_strided();
}

#[test]
fn upper_equals_lower() {
    run_upper_equals_lower();
}

