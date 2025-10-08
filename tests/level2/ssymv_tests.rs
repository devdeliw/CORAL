use blas_src as _;
use cblas_sys::{ cblas_ssymv, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::enums::CoralTriangular;
use coral::level2::ssymv::ssymv;

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
fn cblas_ssymv_ref(
    tri   : CoralTriangular,
    n     : i32,
    alpha : f32,
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : f32,
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_ssymv(
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
) -> Vec<f32> {
    assert!(lda >= n);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f32;
            let hi  = i.max(j) as f32;
            let val = (0.1 as f32) + (0.5 as f32) * lo + (0.25 as f32) * hi;
            a[i + j * lda] = val;
            a[j + i * lda] = val;
        }
    }
    a
}

fn make_upper_stored_lower_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= n);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            if i <= j {
                let lo  = i.min(j) as f32;
                let hi  = i.max(j) as f32;
                a[i + j * lda] = (0.2 as f32) + (0.3 as f32) * lo + (0.15 as f32) * hi;
            } else {
                a[i + j * lda] = f32::NAN;
            }
        }
    }
    a
}

fn make_lower_stored_upper_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= n);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            if i >= j {
                let lo  = i.min(j) as f32;
                let hi  = i.max(j) as f32;
                a[i + j * lda] = (0.17 as f32) + (0.45 as f32) * lo + (0.08 as f32) * hi;
            } else {
                a[i + j * lda] = f32::NAN;
            }
        }
    }
    a
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f32,
) -> Vec<f32> {
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
    src         : &[f32],
    inc         : usize,
    len_logical : usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(len_logical);

    let mut idx = 0;
    for _ in 0..len_logical {
        out.push(src[idx]);
        idx += inc;
    }
    out
}

fn assert_allclose(
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
            "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})"
        );
    }
}

const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-5;

fn run_case(
    tri   : CoralTriangular,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : f32,
    beta  : f32,
) {
    let a  = make_symmetric_col_major(n, lda);
    let x  = make_strided_vec(n, incx, |k| (0.2 as f32) + (0.1 as f32) * (k as f32));
    let y0 = make_strided_vec(n, incy, |k| (-0.3 as f32) + (0.05 as f32) * (k as f32));

    let mut y_coral = y0.clone();
    ssymv(
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
    cblas_ssymv_ref(
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

    // alpha=0, beta=0, mixed 
    let coeffs: &[(f32, f32)] = &[
        ((1.0 as f32),   (0.0 as f32)),
        ((0.0 as f32),   (0.7 as f32)),
        ((0.75 as f32),  (-0.25 as f32)),
        ((-0.6 as f32),  (0.4 as f32)),
        ((1.1 as f32),   (0.0 as f32)),
        ((0.85 as f32),  (0.1 as f32)),
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
fn small_all_uplos() {
    run_all_uplos(6, 6, &[(1, 1)]);
}

#[test]
fn large_all_uplos() {
    run_all_uplos(768, 768, &[(1, 1)]);
}

#[test]
fn padded_all_uplos() {
    run_all_uplos(300, 300 + 5, &[(1, 1)]);
}

#[test]
fn strided_all_uplos() {
    run_all_uplos(640, 640, &[(2, 3), (3, 2), (2, 1)]);
}

#[test]
fn upper_equals_lower() {
    let n   = 513;
    let lda = n;

    let a   = make_symmetric_col_major(n, lda);
    let x   = (0..n).map(|k| (-0.07 as f32) + (0.013 as f32) * (k as f32)).collect::<Vec<_>>();
    let y0  = (0..n).map(|k| (0.02 as f32)  - (0.004 as f32) * (k as f32)).collect::<Vec<_>>();

    let alpha = 1.123;
    let beta  = -0.321;

    let mut y_u = y0.clone();
    let mut y_l = y0.clone();

    ssymv(
        CoralTriangular::UpperTriangular,
        n, 
        alpha, 
        &a, 
        lda,
        &x,
        1, 
        beta,
        &mut y_u, 
        1
        );
    ssymv(
        CoralTriangular::LowerTriangular,
        n, 
        alpha,
        &a, 
        lda, 
        &x,
        1, 
        beta,
        &mut y_l, 
        1
    );

    assert_allclose(&y_u, &y_l, RTOL, ATOL);
}

#[test]
fn upper_respects_triangle() {
    let n   = 200;
    let lda = n + 3;

    let alpha = 0.7;
    let beta  = 0.25;

    let a  = make_upper_stored_lower_garbage(n, lda);
    let x  = (0..n).map(|k| (0.03 as f32) * (k as f32) - (0.5 as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| (-0.1 as f32) + (0.002 as f32) * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    ssymv(
        CoralTriangular::UpperTriangular,
        n, 
        alpha,
        &a, 
        lda,
        &x, 
        1,
        beta,
        &mut y_coral, 
        1
    );

    let mut y_ref = y0.clone();
    cblas_ssymv_ref(
        CoralTriangular::UpperTriangular,
        n as i32, 
        alpha,
        a.as_ptr(),
        lda as i32,
        x.as_ptr(), 
        1, 
        beta,
        y_ref.as_mut_ptr(),
        1
    );

    assert_allclose(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_respects_triangle() {
    let n   = 180;
    let lda = n;

    let alpha = -0.9;
    let beta  = 0.6;

    let a  = make_lower_stored_upper_garbage(n, lda);
    let x  = (0..n).map(|k| (0.2 as f32) - (0.001 as f32) * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| (0.15 as f32) + (0.004 as f32) * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    ssymv(
        CoralTriangular::LowerTriangular,
        n, 
        alpha, 
        &a,
        lda,
        &x, 
        1,
        beta,
        &mut y_coral,
        1
    );

    let mut y_ref = y0.clone();
    cblas_ssymv_ref(
        CoralTriangular::LowerTriangular,
        n as i32,
        alpha, 
        a.as_ptr(),
        lda as i32,
        x.as_ptr(),
        1, 
        beta,
        y_ref.as_mut_ptr(),
        1
    );

    assert_allclose(&y_coral, &y_ref, RTOL, ATOL);
}

