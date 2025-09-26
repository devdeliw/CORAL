use blas_src as _;
use cblas_sys::{
    cblas_cgemv,
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
};
use coral::level2::{
    enums::CoralTranspose,
    cgemv::cgemv,
};

// cblas wrappers
fn cblas_notranspose(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : [f32; 2],
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_cgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f32; 2], 
            a               as *const [f32; 2],
            lda,
            x               as *const [f32; 2],
            incx,
            beta.as_ptr()   as *const [f32; 2],
            y as *mut [f32; 2],
            incy,
        );
    }
}

fn cblas_transpose(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : [f32; 2],
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_cgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            a               as *const [f32; 2],
            lda,
            x               as *const [f32; 2],
            incx,
            beta.as_ptr()   as *const [f32; 2],
            y               as *mut [f32; 2],
            incy,
        );
    }
}

fn cblas_conjtranspose(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : [f32; 2],
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_cgemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasConjTrans,
            m,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            a               as *const [f32; 2],
            lda,
            x               as *const [f32; 2],
            incx,
            beta.as_ptr()   as *const [f32; 2],
            y               as *mut [f32; 2],
            incy,
        );
    }
}

// helpers
fn make_col_major_cmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            // deterministic but nontrivial
            a[idx]     = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;   // re
            a[idx + 1] = -0.2 + (i as f32) * 0.3 - (j as f32) * 0.15;  // im
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
        let z = f(k);
        v[2 * idx]     = z[0];
        v[2 * idx + 1] = z[1];
        idx += inc;
    }
    v
}

fn copy_logical_strided_c(
    src         : &[f32],
    inc         : usize,
    len_logical : usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * len_logical);

    let mut idx = 0usize;
    for _ in 0..len_logical {
        out.push(src[2 * idx]);
        out.push(src[2 * idx + 1]);
        idx += inc;
    }
    out
}

fn assert_allclose_c(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
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

const RTOL: f32 = 5e-3;
const ATOL: f32 = 1e-6;

#[test]
fn notranspose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [0.75f32, -0.10f32];
    let beta  = [-0.25f32, 0.05f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [0.2 + 0.1 * (k as f32), -0.05 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [-0.3 + 0.05 * (k as f32), 0.02 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn transpose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [-0.6f32, 0.2f32];
    let beta  = [0.4f32, -0.1f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..m).flat_map(|k| [0.1 - 0.07 * (k as f32), 0.03 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..n).flat_map(|k| [0.03 * (k as f32), -0.02 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn conjtranspose_small() {
    let m   = 5usize;
    let n   = 4usize;
    let lda = m;

    let alpha = [0.3f32, 0.7f32];
    let beta  = [0.1f32, -0.2f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..m).flat_map(|k| [-0.08 * (k as f32), 0.09 + 0.01 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..n).flat_map(|k| [0.02 * (k as f32), 0.01 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn notranspose_large() {
    let m   = 512usize;
    let n   = 256usize;
    let lda = m;

    let alpha = [1.25f32, -0.5f32];
    let beta  = [-0.5f32, 0.25f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [0.2 + (k as f32) * 0.1, -0.01 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [-0.3 + (k as f32) * 0.05, 0.005 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn transpose_large() {
    let m   = 640usize;
    let n   = 320usize;
    let lda = m;

    let alpha = [-0.75f32, 0.3f32];
    let beta  = [0.3f32, -0.15f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..m).flat_map(|k| [0.4 - (k as f32) * 0.07, -0.02 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..n).flat_map(|k| [0.1 * (k as f32), 0.01 - 0.003 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn conjtranspose_large() {
    let m   = 384usize;
    let n   = 768usize;
    let lda = m + 3;

    let alpha = [0.85f32, 0.15f32];
    let beta  = [0.1f32, -0.05f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..m).flat_map(|k| [0.05 - 0.02 * (k as f32), 0.04 + 0.01 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..n).flat_map(|k| [-0.2 + 0.02 * (k as f32), 0.03 - 0.005 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn notranspose_padded() {
    let m   = 256usize;
    let n   = 512usize;
    let lda = m + 7;

    let alpha = [0.85f32, -0.2f32];
    let beta  = [0.1f32, 0.0f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [-0.05 + 0.02 * (k as f32), 0.01 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [0.01 * (k as f32) - 0.2, -0.02 + 0.003 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn strided_notranspose() {
    let m   = 512usize;
    let n   = 256usize;
    let lda = m;

    let alpha = [0.95f32, 0.1f32];
    let beta  = [-1.1f32, 0.2f32];

    let incx = 2usize;
    let incy = 3usize;

    let a = make_col_major_cmatrix(m, n, lda);
    let x = make_strided_cvec(n, incx, |k| [0.05 + 0.03 * (k as f32), 0.02 - 0.01 * (k as f32)]);
    let y = make_strided_cvec(m, incy, |k| [-0.2 + 0.02 * (k as f32), 0.01 * (k as f32)]);

    let mut y_coral = y.clone();
    cgemv(
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

    let y_coral_logical = copy_logical_strided_c(&y_coral, incy, m);
    let y_ref_logical   = copy_logical_strided_c(&y_ref,   incy, m);

    assert_allclose_c(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_transpose() {
    let m   = 640usize;
    let n   = 320usize;
    let lda = m;

    let alpha = [-0.4f32, 0.6f32];
    let beta  = [0.9f32, -0.75f32];

    let incx = 3usize;
    let incy = 2usize;

    let a = make_col_major_cmatrix(m, n, lda);
    let x = make_strided_cvec(m, incx, |k| [0.12 - 0.01 * (k as f32), 0.02 * (k as f32)]);
    let y = make_strided_cvec(n, incy, |k| [0.2 + 0.005 * (k as f32), -0.015 * (k as f32)]);

    let mut y_coral = y.clone();
    cgemv(
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

    let y_coral_logical = copy_logical_strided_c(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_c(&y_ref,   incy, n);
    assert_allclose_c(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_conjtranspose() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.7f32, -0.2f32];
    let beta  = [0.0f32, 0.0f32];

    let incx = 2usize;
    let incy = 4usize;

    let a = make_col_major_cmatrix(m, n, lda);
    let x = make_strided_cvec(m, incx, |k| [0.1 * (k as f32), 0.03 - 0.02 * (k as f32)]);
    let y = make_strided_cvec(n, incy, |k| [0.3 - 0.01 * (k as f32), -0.02 + 0.004 * (k as f32)]);

    let mut y_coral = y.clone();
    cgemv(
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

    let y_coral_logical = copy_logical_strided_c(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_c(&y_ref,   incy, n);

    assert_allclose_c(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn alpha_zero_scales_y() {
    let m   = 192usize;
    let n   = 96usize;
    let lda = m;

    let alpha = [0.0f32, 0.0f32];
    let beta  = [-0.75f32, 0.2f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [0.1 * (k as f32), 0.02 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [0.05 * (k as f32) - 0.4, -0.03 + 0.01 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn beta_zero_overwrites_y() {
    let m   = 200usize;
    let n   = 160usize;
    let lda = m + 3;

    let alpha = [1.1f32, -0.6f32];
    let beta  = [0.0f32, 0.0f32];

    let a  = make_col_major_cmatrix(m, n, lda);
    let x  = (0..n).flat_map(|k| [-0.02 + 0.015 * (k as f32), 0.01 - 0.005 * (k as f32)]).collect::<Vec<_>>();
    let y0 = (0..m).flat_map(|k| [0.3 - 0.01 * (k as f32), -0.02 + 0.004 * (k as f32)]).collect::<Vec<_>>();

    let mut y_coral = y0.clone();
    cgemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

