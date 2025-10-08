use blas_src as _; 
use cblas_sys::{
    cblas_ssymv,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
};

use coral::enums::CoralTriangular; 
use coral::level2::ssymv::ssymv;

// cblas wrappers
fn cblas_upper(
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
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];

    for j in 0..n {
        for i in 0..=j {
            let lo = i.min(j) as f32;
            let hi = i.max(j) as f32;
            let val = 0.1 + 0.5 * lo + 0.25 * hi;

            a[i + j * lda] = val; // (i, j)
            a[j + i * lda] = val; // (j, i) mirror
        }
    }
    a
}

fn make_upper_stored_lower_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    // fill upper (i <= j) with symmetric values
    // lower with NaN sentinels
    for j in 0..n {
        for i in 0..n {
            if i <= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                a[i + j * lda] = 0.2 + 0.3 * lo + 0.15 * hi;
            } else {
                a[i + j * lda] = f32::NAN; // should be ignored 
            }
        }
    }
    a
}

fn make_lower_stored_upper_garbage(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    // fill lower (i >= j) with symmetric values 
    // upper with NaN sentinels
    for j in 0..n {
        for i in 0..n {
            if i >= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                a[i + j * lda] = 0.17 + 0.45 * lo + 0.08 * hi;
            } else {
                a[i + j * lda] = f32::NAN; // should be ignored when using Lower
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
    let mut v = vec![0.0f32; (len_logical - 1) * inc + 1];

    let mut idx = 0usize;
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

    let mut idx = 0usize;
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
        let tol = atol + rtol * x.abs().max(y.abs());

        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-6;

// tests
#[test]
fn upper_small() {
    let n   = 6usize;
    let lda = n;

    let alpha = 0.75f32;
    let beta  = -0.25f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.2 + 0.1 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.3 + 0.05 * (k as f32)).collect::<Vec<_>>();

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

    let alpha = -0.6f32;
    let beta  = 0.4f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.1 - 0.07 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.03 * (k as f32)).collect::<Vec<_>>();

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
fn lower_large() {
    let n   = 768usize;
    let lda = n;

    let alpha = -0.75f32;
    let beta  = 0.3f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.4 - (k as f32) * 0.07).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.1 * (k as f32)).collect::<Vec<_>>();

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

    let alpha = 0.85f32;
    let beta  = 0.1f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.05 + 0.02 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.01 * (k as f32) - 0.2).collect::<Vec<_>>();

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

    let alpha = 1.1f32;
    let beta  = -0.2f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.02 + 0.015 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.3 - 0.01 * (k as f32)).collect::<Vec<_>>();

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

    let alpha = 0.95f32;
    let beta  = -1.1f32;

    let incx = 2usize;
    let incy = 3usize;

    let a = make_symmetric_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| 0.05 + 0.03 * (k as f32));
    let y = make_strided_vec(n, incy, |k| -0.2 + 0.02 * (k as f32));

    let mut y_coral = y.clone();
    ssymv(
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

    let alpha = -0.4f32;
    let beta  = 0.9f32;

    let incx = 3usize;
    let incy = 2usize;

    let a = make_symmetric_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| 0.12 - 0.01 * (k as f32));
    let y = make_strided_vec(n, incy, |k| 0.2 + 0.005 * (k as f32));

    let mut y_coral = y.clone();
    ssymv(
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

    let alpha = 0.0f32;
    let beta  = -0.75f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| 0.1 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.05 * (k as f32) - 0.4).collect::<Vec<_>>();

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

    let alpha = 1.1f32;
    let beta  = 0.0f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.02 + 0.015 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.3 - 0.01 * (k as f32)).collect::<Vec<_>>();

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

    let alpha = 1.123f32;
    let beta  = -0.321f32;

    let a  = make_symmetric_col_major(n, lda);
    let x  = (0..n).map(|k| -0.07 + 0.013 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k|  0.02 - 0.004 * (k as f32)).collect::<Vec<_>>();

    let mut y_upper = y0.clone();
    let mut y_lower = y0.clone();

    ssymv(
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

    ssymv(
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

    let alpha = 0.7f32;
    let beta  = 0.25f32;

    let a  = make_upper_stored_lower_garbage(n, lda);
    let x  = (0..n).map(|k| 0.03 * (k as f32) - 0.5).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| -0.1 + 0.002 * (k as f32)).collect::<Vec<_>>();

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

    let alpha = -0.9f32;
    let beta  = 0.6f32;

    let a  = make_lower_stored_upper_garbage(n, lda);
    let x  = (0..n).map(|k| 0.2 - 0.001 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.15 + 0.004 * (k as f32)).collect::<Vec<_>>();

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

