use blas_src as _;
use cblas_sys::{ cblas_chemv, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::level2::{
    chemv::chemv,
    enums::CoralTriangular,
};

// wrappers 
fn cblas_upper(
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
        cblas_chemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_UPLO::CblasUpper,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            a               as *const [f32; 2],
            lda,
            x               as *const [f32; 2],
            incx,
            beta.as_ptr()   as *const [f32; 2],
            y               as *mut   [f32; 2],
            incy,
        );
    }
}

fn cblas_lower(
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
        cblas_chemv(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_UPLO::CblasLower,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            a               as *const [f32; 2],
            lda,
            x               as *const [f32; 2],
            incx,
            beta.as_ptr()   as *const [f32; 2],
            y               as *mut   [f32; 2],
            incy,
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

fn make_upper_stored_lower_garbage_c(n: usize, lda: usize) -> Vec<f32> {
    // upper contains valid; lower contains NaN
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);
            if i <= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                let re = 0.20 + 0.30 * lo + 0.15 * hi;
                let im = if i == j { 0.0 } else { 0.05 + 0.01 * (hi - lo) };
                a[idx]     = re;
                a[idx + 1] = im;
            } else {
                a[idx]     = f32::NAN;
                a[idx + 1] = f32::NAN;
            }
        }
    }
    a
}

fn make_lower_stored_upper_garbage_c(n: usize, lda: usize) -> Vec<f32> {
    // lower contains valid; upper contains NaN
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);
            if i >= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                let re = 0.17 + 0.45 * lo + 0.08 * hi;
                let im = if i == j { 0.0 } else { -0.03 - 0.02 * (hi - lo) }; 
                a[idx]     = re;
                a[idx + 1] = im;
            } else {
                a[idx]     = f32::NAN;
                a[idx + 1] = f32::NAN;
            }
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

fn copy_logical_strided_c(
    src         : &[f32],
    inc         : usize,                
    len_logical : usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; 2 * len_logical];
    let mut idx = 0usize;
    for k in 0..len_logical {
        out[2 * k]     = src[2 * idx];
        out[2 * k + 1] = src[2 * idx + 1];
        idx += inc;
    }
    out
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

const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-6;

// tests 

#[test]
fn upper_small() {
    let n   = 6usize;
    let lda = n;

    let alpha = [0.75f32, -0.10f32];
    let beta  = [-0.25f32, 0.15f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.2 + 0.1 * (k as f32), -0.3 + 0.05 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [-0.3 + 0.05 * (k as f32), 0.1 - 0.02 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_small() {
    let n   = 6usize;
    let lda = n;

    let alpha = [-0.6f32, 0.2f32];
    let beta  = [0.4f32, -0.1f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.1 - 0.07 * (k as f32), 0.03 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.03 * (k as f32), -0.015 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn upper_large() {
    let n   = 768usize;
    let lda = n;

    let alpha = [1.25f32, 0.0f32];
    let beta  = [-0.5f32, 0.0f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.2 + (k as f32) * 0.1, 0.05 - 0.02 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [-0.3 + (k as f32) * 0.05, 0.02 * (k as f32) - 0.1]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_large() {
    let n   = 640usize;
    let lda = n;

    let alpha = [-0.75f32, 0.35f32];
    let beta  = [0.3f32, -0.2f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.4 - (k as f32) * 0.07, -0.03 + 0.004 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.1 * (k as f32), 0.02 - 0.0005 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn upper_padded() {
    let n   = 256usize;
    let lda = n + 7;

    let alpha = [0.85f32, -0.1f32];
    let beta  = [0.1f32, 0.05f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [-0.05 + 0.02 * (k as f32), 0.06 - 0.01 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.01 * (k as f32) - 0.2, -0.03 + 0.005 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_padded() {
    let n   = 300usize;
    let lda = n + 5;

    let alpha = [1.1f32, 0.25f32];
    let beta  = [0.0f32, -0.2f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [-0.02 + 0.015 * (k as f32), 0.01 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.3 - 0.01 * (k as f32), -0.02 + 0.003 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn strided_upper() {
    let n   = 384usize;
    let lda = n;

    let alpha = [0.95f32, -1.1f32];
    let beta  = [-0.3f32, 0.7f32];

    let incx = 2usize;
    let incy = 3usize;

    let a = make_hermitian_col_major(n, lda);
    let x = make_strided_cvec(n, incx, |k| [0.05 + 0.03 * (k as f32), -0.02 + 0.004 * (k as f32)]);
    let y = make_strided_cvec(n, incy, |k| [-0.2 + 0.02 * (k as f32), 0.015 - 0.001 * (k as f32)]);

    let mut y_coral = y.clone();
    chemv(
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

    let y_coral_logical = copy_logical_strided_c(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_c(&y_ref,   incy, n);
    assert_allclose_c(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_lower() {
    let n   = 512usize;
    let lda = n;

    let alpha = [-0.4f32, 0.9f32];
    let beta  = [0.9f32, -0.4f32];

    let incx = 3usize;
    let incy = 2usize;

    let a = make_hermitian_col_major(n, lda);
    let x = make_strided_cvec(n, incx, |k| [0.12 - 0.01 * (k as f32), 0.02 * (k as f32) - 0.1]);
    let y = make_strided_cvec(n, incy, |k| [0.2 + 0.005 * (k as f32), -0.015 + 0.002 * (k as f32)]);

    let mut y_coral = y.clone();
    chemv(
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

    let y_coral_logical = copy_logical_strided_c(&y_coral, incy, n);
    let y_ref_logical   = copy_logical_strided_c(&y_ref,   incy, n);
    assert_allclose_c(&y_coral_logical, &y_ref_logical, RTOL, ATOL);
}

#[test]
fn alpha_zero_scales_y() {
    let n   = 200usize;
    let lda = n;

    let alpha = [0.0f32, 0.0f32];
    let beta  = [-0.75f32, 0.2f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.1 * (k as f32), 0.01 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.05 * (k as f32) - 0.4, -0.02 + 0.003 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn beta_zero_overwrites_y() {
    let n   = 192usize;
    let lda = n + 3;

    let alpha = [1.1f32, -0.2f32];
    let beta  = [0.0f32, 0.0f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [-0.02 + 0.015 * (k as f32), 0.02 * (k as f32) - 0.1]);
    let y0 = make_strided_cvec(n, 1, |k| [0.3 - 0.01 * (k as f32), -0.04 + 0.006 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn upper_equals_lower() {
    let n   = 257usize;
    let lda = n;

    let alpha = [1.123f32, -0.321f32];
    let beta  = [-0.321f32, 0.777f32];

    let a  = make_hermitian_col_major(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [-0.07 + 0.013 * (k as f32), 0.015 - 0.002 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.02 - 0.004 * (k as f32), -0.01 + 0.001 * (k as f32)]);

    let mut y_upper = y0.clone();
    let mut y_lower = y0.clone();

    chemv(
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

    chemv(
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

    assert_allclose_c(&y_upper, &y_lower, RTOL, ATOL);
}

#[test]
fn upper_respects_triangle() {
    // lower triangular is NaN garbage
    let n   = 200usize;
    let lda = n + 3;

    let alpha = [0.7f32, 0.1f32];
    let beta  = [0.25f32, -0.05f32];

    let a  = make_upper_stored_lower_garbage_c(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.03 * (k as f32) - 0.5, 0.02 - 0.01 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [-0.1 + 0.002 * (k as f32), 0.05 - 0.003 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_respects_triangle() {
    // upper triangular is NaN garbage
    let n   = 180usize;
    let lda = n;

    let alpha = [-0.9f32, 0.2f32];
    let beta  = [0.6f32, -0.1f32];

    let a  = make_lower_stored_upper_garbage_c(n, lda);
    let x  = make_strided_cvec(n, 1, |k| [0.2 - 0.001 * (k as f32), -0.03 + 0.004 * (k as f32)]);
    let y0 = make_strided_cvec(n, 1, |k| [0.15 + 0.004 * (k as f32), -0.02 + 0.003 * (k as f32)]);

    let mut y_coral = y0.clone();
    chemv(
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

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

