use blas_src as _; 
use cblas_sys::{
    cblas_sgemv,
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
};
use coral::level2::{
    enums::CoralTranspose,
    sgemv::sgemv,
};

// cblas wrappers
fn cblas_notranspose(
    m     : i32,
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
        cblas_sgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
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

fn cblas_transpose(
    m     : i32,
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
        cblas_sgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasTrans,
            m,
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
fn make_col_major_matrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];

    for j in 0..n {
        for i in 0..m {
            a[i + j * lda] = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;
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
const ATOL: f32 = 1e-5;

#[test]
fn notranspose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = 0.75f32;
    let beta  = -0.25f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..n).map(|k| 0.2 + 0.1 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..m).map(|k| -0.3 + 0.05 * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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
fn transpose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = -0.6f32;
    let beta  = 0.4f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..m).map(|k| 0.1 - 0.07 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.03 * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::Transpose,
        m,
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
    cblas_transpose(
        m as i32,
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
fn notranspose_large() {
    let m   = 1024usize;
    let n   = 512usize;
    let lda = m;

    let alpha = 1.25f32;
    let beta  = -0.5f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..n).map(|k| 0.2 + (k as f32) * 0.1).collect::<Vec<_>>();
    let y0 = (0..m).map(|k| -0.3 + (k as f32) * 0.05).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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
fn transpose_large() {
    let m   = 1024usize;
    let n   = 512usize;
    let lda = m;

    let alpha = -0.75f32;
    let beta  = 0.3f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..m).map(|k| 0.4 - (k as f32) * 0.07).collect::<Vec<_>>();
    let y0 = (0..n).map(|k| 0.1 * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::Transpose,
        m,
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
    cblas_transpose(
        m as i32,
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
fn notranspose_padded() {
    let m   = 256usize;
    let n   = 1024usize;
    let lda = m + 7;

    let alpha = 0.85f32;
    let beta  = 0.1f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..n).map(|k| -0.05 + 0.02 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..m).map(|k| 0.01 * (k as f32) - 0.2).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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
fn strided_notranspose() {
    let m   = 1024usize;
    let n   = 512usize;
    let lda = m;

    let alpha = 0.95f32;
    let beta  = -1.1f32;

    let incx = 2usize;
    let incy = 3usize;

    let a = make_col_major_matrix(m, n, lda);
    let x = make_strided_vec(n, incx, |k| 0.05 + 0.03 * (k as f32));
    let y = make_strided_vec(m, incy, |k| -0.2 + 0.02 * (k as f32));

    let mut y_coral = y.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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

    let y_coral_logical = copy_logical_strided(&y_coral, incy, m);
    let y_ref_logical   = copy_logical_strided(&y_ref,   incy, m);

    assert_allclose(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_transpose() {
    let m   = 640usize;
    let n   = 320usize;
    let lda = m;

    let alpha = -0.4f32;
    let beta  = 0.9f32;

    let incx = 3usize;
    let incy = 2usize;

    let a = make_col_major_matrix(m, n, lda);
    let x = make_strided_vec(m, incx, |k| 0.12 - 0.01 * (k as f32));
    let y = make_strided_vec(n, incy, |k| 0.2 + 0.005 * (k as f32));

    let mut y_coral = y.clone();
    sgemv(
        CoralTranspose::Transpose,
        m,
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
    cblas_transpose(
        m as i32,
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
    let m   = 256usize;
    let n   = 128usize;
    let lda = m;

    let alpha = 0.0f32;
    let beta  = -0.75f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..n).map(|k| 0.1 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..m).map(|k| 0.05 * (k as f32) - 0.4).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = 1.1f32;
    let beta  = 0.0f32;

    let a  = make_col_major_matrix(m, n, lda);
    let x  = (0..n).map(|k| -0.02 + 0.015 * (k as f32)).collect::<Vec<_>>();
    let y0 = (0..m).map(|k| 0.3 - 0.01 * (k as f32)).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    sgemv(
        CoralTranspose::NoTranspose,
        m,
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
    cblas_notranspose(
        m as i32,
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

