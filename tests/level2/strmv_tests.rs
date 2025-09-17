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
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
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
fn make_upper_col_major(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..=j {
            a[i + j * lda] = 0.2 + (i as f32) * 0.3 + (j as f32) * 0.15;
        }
    }
    a
}

// build padded triangular matrix
fn make_padded_col_major(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * lda] = if i >= j {
                // lower triangle
                0.1 + (i as f32) * 0.2 + (j as f32) * 0.1
            } else {
                // upper part
                0.1 + (i as f32) * 0.1 + (j as f32) * 0.2
            };
        }
    }
    a
}

fn make_lower_col_major(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in j..n {
            a[i + j * lda] = -0.1 + (i as f32) * 0.25 - (j as f32) * 0.2;
        }
    }
    a
}

fn make_strided_vec(len_logical: usize, inc: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut v = vec![0.0f32; (len_logical - 1) * inc + 1];
    let mut idx = 0usize;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn copy_logical_strided(src: &[f32], inc: usize, len_logical: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len_logical);
    let mut idx = 0usize;
    for _ in 0..len_logical {
        out.push(src[idx]);
        idx += inc;
    }
    out
}

fn assert_allclose(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "mismatch at {}: got {} vs {} (diff = {}, tol = {})",
            i, x, y, diff, tol
        );
    }
} 

// tests
#[test]
fn upper_notranspose() {
    let n   = 6usize;
    let lda = n;
    let a   = make_upper_col_major(n, lda);
    let x0  = (0..n).map(|k| 0.1 + 0.2 * k as f32).collect::<Vec<_>>();

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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

#[test]
fn upper_transpose() {
    let n   = 7usize;
    let lda = n;
    let a   = make_upper_col_major(n, lda);
    let x0  = (0..n).map(|k| 0.3 - 0.05 * k as f32).collect::<Vec<_>>();

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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

#[test]
fn lower_notranspose() {
    let n   = 5usize;
    let lda = n;
    let a   = make_lower_col_major(n, lda);
    let x0  = (0..n).map(|k| -0.2 + 0.1 * k as f32).collect::<Vec<_>>();

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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

#[test]
fn lower_transpose() {
    let n   = 6usize;
    let lda = n;
    let a   = make_lower_col_major(n, lda);
    let x0  = (0..n).map(|k| 0.15 * (k as f32)).collect::<Vec<_>>();

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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

#[test]
fn strided() {
    let n   = 8usize;
    let lda = n;
    let a   = make_upper_col_major(n, lda);

    let incx = 3usize;
    let x = make_strided_vec(n, incx, |k| 0.05 + 0.04 * k as f32);

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
    let x_ref_logical = copy_logical_strided(&x_ref, incx, n);
    assert_allclose(&x_coral_logical, &x_ref_logical, 1e-4);
}

#[test]
fn unitdiag_upper() {
    let n   = 6usize;
    let lda = n;
    let a   = make_upper_col_major(n, lda);
    let x0  = (0..n).map(|k| 0.2 + 0.1 * k as f32).collect::<Vec<_>>();

    // coral
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

    // cblas
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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

#[test]
fn paddedlda() {
    let n   = 5usize;
    let lda = n + 3; // padded leading dimension
    let a   = make_padded_col_major(n, lda);
    let x0  = (0..n).map(|k| 0.3 - 0.06 * k as f32).collect::<Vec<_>>();

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

    assert_allclose(&x_coral, &x_ref, 1e-4);
}

