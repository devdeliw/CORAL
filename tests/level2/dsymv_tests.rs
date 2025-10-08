use blas_src as _;
use cblas_sys::{ cblas_dsymv, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::enums::CoralTriangular;
use coral::level2::dsymv::dsymv;

#[inline(always)]
fn to_cblas_uplo(
    tri : CoralTriangular,
) -> CBLAS_UPLO {
    match tri {
        CoralTriangular::UpperTriangular => CBLAS_UPLO::CblasUpper,
        CoralTriangular::LowerTriangular => CBLAS_UPLO::CblasLower,
    }
}

#[inline(always)]
fn cblas_dsymv_ref(
    tri   : CoralTriangular,
    n     : i32,
    alpha : f64,
    a     : *const f64,
    lda   : i32,
    x     : *const f64,
    incx  : i32,
    beta  : f64,
    y     : *mut f64,
    incy  : i32,
) {
    unsafe {
        cblas_dsymv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas_uplo(tri),
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
}

fn make_symmetric_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    assert!(lda >= n);
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

fn make_upper_stored_lower_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    assert!(lda >= n);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            if i <= j {
                let lo = i.min(j) as f64;
                let hi = i.max(j) as f64;
                a[i + j * lda] = 0.2 + 0.3 * lo + 0.15 * hi;
            } else {
                a[i + j * lda] = f64::NAN;
            }
        }
    }
    a
}

fn make_lower_stored_upper_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    assert!(lda >= n);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            if i >= j {
                let lo = i.min(j) as f64;
                let hi = i.max(j) as f64;
                a[i + j * lda] = 0.17 + 0.45 * lo + 0.08 * hi;
            } else {
                a[i + j * lda] = f64::NAN;
            }
        }
    }
    a
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f64,
) -> Vec<f64> {
    assert!(len_logical > 0 && inc > 0);
    let mut v = vec![0.0; (len_logical - 1) * inc + 1];

    let mut idx = 0;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn copy_logical_strided(
    src         : &[f64],
    inc         : usize,
    len_logical : usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(len_logical);

    let mut idx = 0;
    for _ in 0..len_logical {
        out.push(src[idx]);
        idx += inc;
    }
    out
}

fn assert_allclose(
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

const RTOL: f64 = 1e-13;
const ATOL: f64 = 1e-13;

fn run_case(
    tri   : CoralTriangular,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : f64,
    beta  : f64,
) {
    let a  = make_symmetric_col_major(n, lda);
    let x  = make_strided_vec(n, incx, |k| 0.2 + 0.1 * (k as f64));
    let y0 = make_strided_vec(n, incy, |k| -0.3 + 0.05 * (k as f64));

    let mut y_coral = y0.clone();
    dsymv(
        tri,
        n,
        alpha,
        &a,
        lda,
        &x,
        incx,
        beta,
        &mut y_coral,
        incy,
    );

    let mut y_ref = y0.clone();
    cblas_dsymv_ref(
        tri,
        n as i32,
        alpha,
        a.as_ptr(),
        lda as i32,
        x.as_ptr(),
        incx as i32,
        beta,
        y_ref.as_mut_ptr(),
        incy as i32,
    );

    let y_coral_logical = copy_logical_strided(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided(&y_ref,   incy, n);
    assert_allclose(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

fn run_all_uplos(
    n            : usize,
    lda          : usize,
    stride_pairs : &[(usize, usize)],
) {
    let uplos = [
        CoralTriangular::UpperTriangular,
        CoralTriangular::LowerTriangular
    ];

    // alpha = 0, beta=0, mixed cases
    let coeffs: &[(f64, f64)] = &[
        (1.0,   0.0),
        (0.0,   0.7),
        (0.75, -0.25),
        (-0.6,  0.4),
        (1.1,   0.0),
        (0.85,  0.1),
    ];

    for &(incx, incy) in stride_pairs {
        for &tri in &uplos {
            for &(alpha, beta) in coeffs {
                run_case(tri, n, lda, incx, incy, alpha, beta);
            }
        }
    }
}

#[test]
fn dsymv_small_all_uplos() {
    run_all_uplos(6, 6, &[(1, 1)]);
}

#[test]
fn dsymv_large_all_uplos() {
    run_all_uplos(768, 768, &[(1, 1)]);
}

#[test]
fn dsymv_padded_all_uplos() {
    run_all_uplos(300, 300 + 5, &[(1, 1)]);
}

#[test]
fn dsymv_strided_all_uplos() {
    run_all_uplos(640, 640, &[(2, 3), (3, 2), (2, 1)]);
}

#[test]
fn dsymv_upper_equals_lower() {
    let n   = 513;
    let lda = n;

    let a   = make_symmetric_col_major(n, lda);
    let x   = (0..n).map(|k| -0.07 + 0.013 * (k as f64)).collect::<Vec<_>>();
    let y0  = (0..n).map(|k|  0.02 - 0.004 * (k as f64)).collect::<Vec<_>>();

    let alpha = 1.123;
    let beta  = -0.321;

    let mut y_u = y0.clone();
    let mut y_l = y0.clone();

    dsymv(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &a,
        lda,
        &x,
        1,
        beta,
        &mut y_u,
        1,
    );

    dsymv(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &a,
        lda,
        &x,
        1,
        beta,
        &mut y_l,
        1,
    );

    assert_allclose(&y_u, &y_l, RTOL, ATOL);
}

#[test]
fn dsymv_upper_respects_triangle() {
    let n   = 200;
    let lda = n + 3;

    let alpha = 0.7;
    let beta  = 0.25;

    let a  = make_upper_stored_lower_garbage(n, lda);
    let x  = (0..n).map(|k| 0.03 * (k as f64) - 0.5).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.1 + 0.002 * (k as f64)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    dsymv(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &a,
        lda,
        &x,
        1,
        beta,
        &mut y_coral,
        1,
    );

    let mut y_ref = y0.clone();
    cblas_dsymv_ref(
        CoralTriangular::UpperTriangular,
        n as i32,
        alpha,
        a.as_ptr(),
        lda as i32,
        x.as_ptr(),
        1,
        beta,
        y_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn dsymv_lower_respects_triangle() {
    let n   = 180;
    let lda = n;

    let alpha = -0.9;
    let beta  = 0.6;

    let a  = make_lower_stored_upper_garbage(n, lda);
    let x  = (0..n).map(|k| 0.2 - 0.001 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.15 + 0.004 * (k as f64)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    dsymv(
        CoralTriangular::LowerTriangular,
        n,
        alpha,
        &a,
        lda,
        &x,
        1,
        beta,
        &mut y_coral,
        1,
    );

    let mut y_ref = y0.clone();
    cblas_dsymv_ref(
        CoralTriangular::LowerTriangular,
        n as i32,
        alpha,
        a.as_ptr(),
        lda as i32,
        x.as_ptr(),
        1,
        beta,
        y_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&y_coral, &y_ref, RTOL, ATOL);
}

