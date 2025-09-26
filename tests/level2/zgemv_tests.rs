use blas_src as _;
use cblas_sys::{
    cblas_zgemv,
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
};
use coral::level2::{
    enums::CoralTranspose,
    zgemv::zgemv,
};

// cblas wrappers
fn cblas_notranspose(
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    a     : *const f64,
    lda   : i32,
    x     : *const f64,
    incx  : i32,
    beta  : [f64; 2],
    y     : *mut f64,
    incy  : i32,
) {
    unsafe {
        cblas_zgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            a               as *const [f64; 2],
            lda,
            x               as *const [f64; 2],
            incx,
            beta.as_ptr()   as *const [f64; 2],
            y               as *mut [f64; 2],
            incy,
        );
    }
}

fn cblas_transpose(
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    a     : *const f64,
    lda   : i32,
    x     : *const f64,
    incx  : i32,
    beta  : [f64; 2],
    y     : *mut f64,
    incy  : i32,
) {
    unsafe {
        cblas_zgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            a               as *const [f64; 2],
            lda,
            x               as *const [f64; 2],
            incx,
            beta.as_ptr()   as *const [f64; 2],
            y               as *mut [f64; 2],
            incy,
        );
    }
}

fn cblas_conjtranspose(
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    a     : *const f64,
    lda   : i32,
    x     : *const f64,
    incx  : i32,
    beta  : [f64; 2],
    y     : *mut f64,
    incy  : i32,
) {
    unsafe {
        cblas_zgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasConjTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            a               as *const [f64; 2],
            lda,
            x               as *const [f64; 2],
            incx,
            beta.as_ptr()   as *const [f64; 2],
            y               as *mut [f64; 2],
            incy,
        );
    }
}

// helpers
fn make_col_major_zmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            a[idx]     = 0.1 + (i as f64) * 0.5 + (j as f64) * 0.25;
            a[idx + 1] = -0.2 + (i as f64) * 0.3 - (j as f64) * 0.15;
        }
    }
    a
}

fn make_strided_zvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f64; 2],
) -> Vec<f64> {
    let mut v = vec![0.0f64; 2 * ((len_logical - 1) * inc + 1)];

    let mut idx = 0usize;
    for k in 0..len_logical {
        let z = f(k);
        v[2 * idx]     = z[0];
        v[2 * idx + 1] = z[1];
        idx += inc;
    }
    v
}

fn copy_logical_strided_z(
    src         : &[f64],
    inc         : usize,
    len_logical : usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * len_logical);

    let mut idx = 0usize;
    for _ in 0..len_logical {
        out.push(src[2 * idx]);
        out.push(src[2 * idx + 1]);
        idx += inc;
    }
    out
}

fn assert_allclose_z(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);

    for i in (0..a.len()).step_by(2) {
        let xr = a[i];     let xi = a[i + 1];
        let yr = b[i];     let yi = b[i + 1];

        let dr = (xr - yr).abs();
        let di = (xi - yi).abs();

        let tr = atol + rtol * xr.abs().max(yr.abs());
        let ti = atol + rtol * xi.abs().max(yi.abs());

        assert!(dr <= tr, "re mismatch at pair {}: {xr} vs {yr} (|Δ|={dr}, tol={tr})", i / 2);
        assert!(di <= ti, "im mismatch at pair {}: {xi} vs {yi} (|Δ|={di}, tol={ti})", i / 2);
    }
}

const RTOL: f64 = 1e-11;
const ATOL: f64 = 1e-12;

#[test]
fn notranspose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [0.75f64, -0.10f64];
    let beta  = [-0.25f64, 0.05f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..n)
        .flat_map(|k| [0.2 + 0.1 * (k as f64), -0.05 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..m)
        .flat_map(|k| [-0.3 + 0.05 * (k as f64), 0.02 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn transpose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [-0.6f64, 0.2f64];
    let beta  = [0.4f64, -0.1f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..m)
        .flat_map(|k| [0.1 - 0.07 * (k as f64), 0.03 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..n)
        .flat_map(|k| [0.03 * (k as f64), -0.02 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn conjtranspose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [0.3f64, 0.7f64];
    let beta  = [0.1f64, -0.2f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..m)
        .flat_map(|k| [-0.08 * (k as f64), 0.09 + 0.01 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..n)
        .flat_map(|k| [0.02 * (k as f64), 0.01 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
        CoralTranspose::ConjugateTranspose,
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
    cblas_conjtranspose(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn notranspose_large() {
    let m   = 512usize;
    let n   = 256usize;
    let lda = m;

    let alpha = [1.25f64, -0.5f64];
    let beta  = [-0.5f64, 0.25f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..n).
        flat_map(|k| [0.2 + (k as f64) * 0.1, -0.01 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..m)
        .flat_map(|k| [-0.3 + (k as f64) * 0.05, 0.005 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn transpose_large() {
    let m   = 640usize;
    let n   = 320usize;
    let lda = m;

    let alpha = [-0.75f64, 0.3f64];
    let beta  = [0.3f64, -0.15f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..m)
        .flat_map(|k| [0.4 - (k as f64) * 0.07, -0.02 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..n)
        .flat_map(|k| [0.1 * (k as f64), 0.01 - 0.003 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn conjtranspose_large() {
    let m   = 384usize;
    let n   = 768usize;
    let lda = m + 3;

    let alpha = [0.85f64, 0.15f64];
    let beta  = [0.1f64, -0.05f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..m)
        .flat_map(|k| [0.05 - 0.02 * (k as f64), 0.04 + 0.01 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..n)
        .flat_map(|k| [-0.2 + 0.02 * (k as f64), 0.03 - 0.005 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
        CoralTranspose::ConjugateTranspose,
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
    cblas_conjtranspose(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn notranspose_padded() {
    let m   = 256usize;
    let n   = 512usize;
    let lda = m + 7;

    let alpha = [0.85f64, -0.2f64];
    let beta  = [0.1f64, 0.0f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..n)
        .flat_map(|k| [-0.05 + 0.02 * (k as f64), 0.01 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..m)
        .flat_map(|k| [0.01 * (k as f64) - 0.2, -0.02 + 0.003 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn strided_notranspose() {
    let m   = 512usize;
    let n   = 256usize;
    let lda = m;

    let alpha = [0.95f64, 0.1f64];
    let beta  = [-1.1f64, 0.2f64];

    let incx = 2usize;
    let incy = 3usize;

    let a = make_col_major_zmatrix(m, n, lda);
    let x = make_strided_zvec(n, incx, |k| [0.05 + 0.03 * (k as f64), 0.02 - 0.01 * (k as f64)]);
    let y = make_strided_zvec(m, incy, |k| [-0.2 + 0.02 * (k as f64), 0.01 * (k as f64)]);

    let mut y_coral = y.clone();
    zgemv(
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

    let y_coral_logical = copy_logical_strided_z(&y_coral, incy, m);
    let y_ref_logical   = copy_logical_strided_z(&y_ref,   incy, m);

    assert_allclose_z(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_transpose() {
    let m   = 640usize;
    let n   = 320usize;
    let lda = m;

    let alpha = [-0.4f64, 0.6f64];
    let beta  = [0.9f64, -0.75f64];

    let incx = 3usize;
    let incy = 2usize;

    let a = make_col_major_zmatrix(m, n, lda);
    let x = make_strided_zvec(m, incx, |k| [0.12 - 0.01 * (k as f64), 0.02 * (k as f64)]);
    let y = make_strided_zvec(n, incy, |k| [0.2 + 0.005 * (k as f64), -0.015 * (k as f64)]);

    let mut y_coral = y.clone();
    zgemv(
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

    let y_coral_logical = copy_logical_strided_z(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_z(&y_ref,   incy, n);
    assert_allclose_z(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_conjtranspose() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.7f64, -0.2f64];
    let beta  = [0.0f64, 0.0f64];

    let incx = 2usize;
    let incy = 4usize;

    let a = make_col_major_zmatrix(m, n, lda);
    let x = make_strided_zvec(m, incx, |k| [0.1 * (k as f64), 0.03 - 0.02 * (k as f64)]);
    let y = make_strided_zvec(n, incy, |k| [0.3 - 0.01 * (k as f64), -0.02 + 0.004 * (k as f64)]);

    let mut y_coral = y.clone();
    zgemv(
        CoralTranspose::ConjugateTranspose,
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
    cblas_conjtranspose(
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

    let y_coral_logical = copy_logical_strided_z(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_z(&y_ref,   incy, n);

    assert_allclose_z(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn alpha_zero_scales_y() {
    let m   = 192usize;
    let n   = 96usize;
    let lda = m;

    let alpha = [0.0f64, 0.0f64];
    let beta  = [-0.75f64, 0.2f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [0.1 * (k as f64), 0.02 * (k as f64)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [0.05 * (k as f64) - 0.4, -0.03 + 0.01 * (k as f64)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn beta_zero_overwrites_y() {
    let m   = 200usize;
    let n   = 160usize;
    let lda = m + 3;

    let alpha = [1.1f64, -0.6f64];
    let beta  = [0.0f64, 0.0f64];

    let a  = make_col_major_zmatrix(m, n, lda);
    let x  = (0..n)
        .flat_map(|k| [-0.02 + 0.015 * (k as f64), 0.01 - 0.005 * (k as f64)])
        .collect::<Vec<_>>();
    let y0 = (0..m)
        .flat_map(|k| [0.3 - 0.01 * (k as f64), -0.02 + 0.004 * (k as f64)])
        .collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    zgemv(
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

    assert_allclose_z(&y_coral, &y_ref, RTOL, ATOL);
}

