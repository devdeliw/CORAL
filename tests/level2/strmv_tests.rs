use cblas_sys::{
    CBLAS_TRANSPOSE,
    CBLAS_LAYOUT,
    CBLAS_UPLO,
    CBLAS_DIAG,
    cblas_strmv,
};

use coral::level2::{
    enums::{
        CoralDiagonal,
        CoralTranspose,
        CoralTriangular,
    },
    strmv::strmv,
};

// cblas wrapper
fn cblas_trmv(
    uplo  : CBLAS_UPLO,
    trans : CBLAS_TRANSPOSE,
    diag  : CBLAS_DIAG,
    n     : i32,
    a     : *const f32,
    lda   : i32,
    x     : *mut f32,
    incx  : i32,
) {
    unsafe {
        cblas_strmv(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            trans,
            diag,
            n,
            a,
            lda,
            x,
            incx,
        );
    }
}

// helpers
fn make_upper_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..=j {
            a[i + j * lda] = 0.2 + (i as f32) * 0.3 + (j as f32) * 0.15;
        }
    }
    a
}

fn make_lower_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in j..n {
            a[i + j * lda] = -0.1 + (i as f32) * 0.25 - (j as f32) * 0.2;
        }
    }
    a
}

// both triangles populated to exercise lda > n 
fn make_padded_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * lda] = if i >= j {
                // lower triangle
                0.1 + (i as f32) * 0.2 + (j as f32) * 0.1
            } else {
                // upper triangle
                0.1 + (i as f32) * 0.1 + (j as f32) * 0.2
            };
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
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-5;

// tests

#[test]
fn upper_notranspose_small() {
    let n   = 6usize;
    let lda = n;

    let a  = make_upper_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.1 + 0.2 * (k as f32)).collect::<Vec<_>>();

    // coral
    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    // cblas
    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn upper_transpose_small() {
    let n   = 7usize;
    let lda = n;

    let a  = make_upper_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.3 - 0.05 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn lower_notranspose_small() {
    let n   = 5usize;
    let lda = n;

    let a  = make_lower_col_major(n, lda);
    let x0 = (0..n).map(|k| -0.2 + 0.1 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn lower_transpose_small() {
    let n   = 6usize;
    let lda = n;

    let a  = make_lower_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.15 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn upper_notranspose_large() {
    let n   = 1024usize;
    let lda = n;

    let a  = make_upper_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.02 + 0.003 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn strided_upper_notranspose() {
    let n    = 640usize;
    let lda  = n;
    let incx = 3usize;

    let a = make_upper_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| 0.05 + 0.01 * (k as f32));

    let mut x_coral = x.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        incx as i32,
    );

    let x_coral_logical = copy_logical_strided(&x_coral, incx, n);
    let x_ref_logical   = copy_logical_strided(&x_ref,   incx, n);
    assert_allclose(&x_coral_logical, &x_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_lower_transpose() {
    let n    = 512usize;
    let lda  = n;
    let incx = 2usize;

    let a = make_lower_col_major(n, lda);
    let x = make_strided_vec(n, incx, |k| -0.02 + 0.004 * (k as f32));

    let mut x_coral = x.clone();
    strmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        incx as i32,
    );

    let x_coral_logical = copy_logical_strided(&x_coral, incx, n);
    let x_ref_logical   = copy_logical_strided(&x_ref,   incx, n);
    assert_allclose(&x_coral_logical, &x_ref_logical, RTOL, ATOL);
}

#[test]
fn unitdiag_upper_notranspose() {
    let n   = 32usize;
    let lda = n;

    let mut a = make_upper_col_major(n, lda);
    // funky diagonal to ensure unit diagonal semantics 
    for i in 0..n {
        a[i + i * lda] = 7.5 + (i as f32) * 0.1;
    }
    let x0 = (0..n).map(|k| 0.2 + 0.01 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::UnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn unitdiag_lower_transpose() {
    let n   = 48usize;
    let lda = n;

    let mut a = make_lower_col_major(n, lda);
    // funky diagonal 
    for i in 0..n {
        a[i + i * lda] = -9.0 + (i as f32) * 0.05;
    }
    let x0 = (0..n).map(|k| -0.1 + 0.02 * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::UnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn paddedlda_lower_transpose() {
    let n   = 127usize;
    let lda = n + 5; // padded leading dimension

    let a  = make_padded_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.3 - 0.006 * (k as f32)).collect::<Vec<_>>();

    // coral
    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    // cblas
    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let n   = 0usize;
    let lda = 1usize; // useless

    let a  = vec![0.0f32; lda]; 
    let x0 = vec![0.0f32; 1];   

    let mut x_coral = x0.clone();
    strmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_trmv(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}

