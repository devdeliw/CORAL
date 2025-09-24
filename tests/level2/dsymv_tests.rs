use blas_src as _; 
use cblas_sys::{
    cblas_dsymv,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
};
use coral::level2::{
    enums::CoralTriangular,
    dsymv::dsymv,
};

// cblas wrappers
fn cblas_upper(
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
            CBLAS_UPLO::CblasUpper,
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

fn cblas_lower(
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
            CBLAS_UPLO::CblasLower,
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

// helpers
fn make_symmetric_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; lda * n];

    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f64;
            let hi  = i.max(j) as f64;
            let val = 0.1f64 + 0.5f64 * lo + 0.25f64 * hi;

            a[i + j * lda] = val; // (i, j)
            a[j + i * lda] = val; // (j, i) mirror
        }
    }
    a
}

fn make_upper_stored_lower_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; lda * n];
    // fill upper (i <= j) with symmetric values
    // lower with NaN sentinels
    for j in 0..n {
        for i in 0..n {
            if i <= j {
                let lo = i.min(j) as f64;
                let hi = i.max(j) as f64;
                a[i + j * lda] = 0.2f64 + 0.3f64 * lo + 0.15f64 * hi;
            } else {
                a[i + j * lda] = f64::NAN; // should be ignored 
            }
        }
    }
    a
}

fn make_lower_stored_upper_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; lda * n];
    // fill lower (i >= j) with symmetric values 
    // upper with NaN sentinels
    for j in 0..n {
        for i in 0..n {
            if i >= j {
                let lo = i.min(j) as f64;
                let hi = i.max(j) as f64;
                a[i + j * lda] = 0.17f64 + 0.45f64 * lo + 0.08f64 * hi;
            } else {
                a[i + j * lda] = f64::NAN; // should be ignored when using Lower
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
    let mut v = vec![0.0f64; (len_logical - 1) * inc + 1];

    let mut idx = 0usize;
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

    let mut idx = 0usize;
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
        let tol = atol + rtol * x.abs().max(y.abs());

        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

const RTOL: f64 = 5e-15;
const ATOL: f64 = 5e-15;

// tests
#[test]
fn upper_small() {
    let n   = 6usize;
    let lda = n;

    let alpha = 0.75f64;
    let beta  = -0.25f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.2f64 + 0.1f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.3f64 + 0.05f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_upper(
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
fn lower_small() {
    let n   = 6usize;
    let lda = n;

    let alpha = -0.6f64;
    let beta  = 0.4f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.1f64 - 0.07f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.03f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_lower(
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
fn upper_large() {
    let n   = 1024usize;
    let lda = n;

    let alpha = 1.25f64;
    let beta  = -0.5f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.2f64 + (k as f64) * 0.1f64).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.3f64 + (k as f64) * 0.05f64).collect::<Vec<_>>();

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
    cblas_upper(
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
fn lower_large() {
    let n   = 768usize;
    let lda = n;

    let alpha = -0.75f64;
    let beta  = 0.3f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.4f64 - (k as f64) * 0.07f64).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.1f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_lower(
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
fn upper_padded() {
    let n   = 256usize;
    let lda = n + 7;

    let alpha = 0.85f64;
    let beta  = 0.1f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.05f64 + 0.02f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.01f64 * (k as f64) - 0.2f64).collect::<Vec<_>>();

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
    cblas_upper(
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
fn lower_padded() {
    let n   = 300usize;
    let lda = n + 5;

    let alpha = 1.1f64;
    let beta  = -0.2f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.02f64 + 0.015f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.3f64 - 0.01f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_lower(
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
fn strided_upper() {
    let n   = 640usize;
    let lda = n;

    let alpha = 0.95f64;
    let beta  = -1.1f64;

    let incx = 2usize;
    let incy = 3usize;

    let a = make_symmetric_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| 0.05f64 + 0.03f64 * (k as f64));
    let y = make_strided_vec(n, incy, |k| -0.2f64 + 0.02f64 * (k as f64));

    let mut y_coral = y.clone();
    dsymv(
        CoralTriangular::UpperTriangular,
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

    let mut y_ref = y.clone();
    cblas_upper(
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

#[test]
fn strided_lower() {
    let n   = 512usize;
    let lda = n;

    let alpha = -0.4f64;
    let beta  = 0.9f64;

    let incx = 3usize;
    let incy = 2usize;

    let a = make_symmetric_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| 0.12f64 - 0.01f64 * (k as f64));
    let y = make_strided_vec(n, incy, |k| 0.2f64 + 0.005f64 * (k as f64));

    let mut y_coral = y.clone();
    dsymv(
        CoralTriangular::LowerTriangular,
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

    let mut y_ref = y.clone();
    cblas_lower(
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

#[test]
fn alpha_zero_scales_y() {
    let n   = 256usize;
    let lda = n;

    let alpha = 0.0f64;
    let beta  = -0.75f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.1f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.05f64 * (k as f64) - 0.4f64).collect::<Vec<_>>();

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
    cblas_upper(
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
fn beta_zero_overwrites_y() {
    let n   = 300usize;
    let lda = n + 5;

    let alpha = 1.1f64;
    let beta  = 0.0f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.02f64 + 0.015f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.3f64 - 0.01f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_lower(
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
fn upper_equals_lower() {
    let n   = 513usize;
    let lda = n;

    let alpha = 1.123f64;
    let beta  = -0.321f64;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.07f64 + 0.013f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k|  0.02f64 - 0.004f64 * (k as f64)).collect::<Vec<_>>();

    let mut y_upper = y0.clone();
    let mut y_lower = y0.clone();

    dsymv(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &a,
        lda,
        &x,
        1,
        beta,
        &mut y_upper,
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
        &mut y_lower,
        1,
    );

    assert_allclose(&y_upper, &y_lower, RTOL, ATOL);
}

#[test]
fn upper_respects_triangle() {
    // lower triangular is NaN garbage 
    let n   = 200usize;
    let lda = n + 3;

    let alpha = 0.7f64;
    let beta  = 0.25f64;

    let a  = make_upper_stored_lower_garbage(n, lda);
    let x  = (0..n).map(|k| 0.03f64 * (k as f64) - 0.5f64).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.1f64 + 0.002f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_upper(
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
fn lower_respects_triangle() {
    let n   = 180usize;
    let lda = n;

    let alpha = -0.9f64;
    let beta  = 0.6f64;

    let a  = make_lower_stored_upper_garbage(n, lda);
    let x  = (0..n).map(|k| 0.2f64 - 0.001f64 * (k as f64)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.15f64 + 0.004f64 * (k as f64)).collect::<Vec<_>>();

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
    cblas_lower(
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

