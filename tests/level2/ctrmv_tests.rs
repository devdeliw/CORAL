use blas_src as _;
use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
    CBLAS_DIAG,
    CBLAS_UPLO,
    cblas_ctrmv,
};

use coral::level2::{
    enums::{
        CoralDiagonal,
        CoralTranspose,
        CoralTriangular,
    },
    ctrmv::ctrmv,
};


// wrapper
fn cblas_ctrmv_wrapper(
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
        cblas_ctrmv(
            CBLAS_LAYOUT::CblasColMajor,
            uplo,
            trans,
            diag,
            n,
            a   as *const [f32; 2],
            lda,
            x   as *mut [f32; 2],
            incx,
        );
    }
}

// helpers 

fn make_upper_c32(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..=j {
            let idx = 2 * (i + j * lda);
            if i == j {
                // mildly varying diagonal
                a[idx]     = 1.0 + 0.02 * (i as f32);
                a[idx + 1] = 0.01 - 0.005 * (i as f32);
            } else {
                // small complex off-diagonal decaying with distance
                let d = (j - i) as f32;
                a[idx]     = 0.01 / (1.0 + d);
                a[idx + 1] = -0.008 / (1.0 + d);
            }
        }
    }
    a
}

fn make_lower_c32(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in j..n {
            let idx = 2 * (i + j * lda);
            if i == j {
                a[idx]     = 1.0 + 0.02 * (i as f32);
                a[idx + 1] = -0.012 + 0.004 * (i as f32);
            } else {
                let d = (i - j) as f32;
                a[idx]     = 0.01 / (1.0 + d);
                a[idx + 1] = 0.007 / (1.0 + d);
            }
        }
    }
    a
}

// well-conditioned upper for UNIT-DIAG tests
fn make_upper_unit_c32(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..=j {
            let idx = 2 * (i + j * lda);
            if i == j {
                // ignored by DIAG=UNIT
                a[idx]     = 2.5 + 0.05 * (i as f32);
                a[idx + 1] = -1.7 + 0.01 * (i as f32);
            } else {
                let d = (j - i) as f32;
                a[idx]     = 5e-4 * (1.0 + d);
                a[idx + 1] = -3e-4 * (1.0 + d);
            }
        }
    }
    a
}

// well-conditioned lower for UNIT-DIAG tests
fn make_lower_unit_c32(n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in j..n {
            let idx = 2 * (i + j * lda);
            if i == j {
                // ignored by DIAG=UNIT
                a[idx]     = -2.0 + 0.04 * (i as f32);
                a[idx + 1] =  1.2 - 0.02 * (i as f32);
            } else {
                let d = (i - j) as f32;
                a[idx]     = 6e-4 * (1.0 + d);
                a[idx + 1] = 4e-4 * (1.0 + d);
            }
        }
    }
    a
}

fn make_strided_vec_c32(
    len_logical: usize,
    inc: usize,
    f: impl Fn(usize) -> (f32, f32),
) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * (len_logical.saturating_sub(1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let (re, im) = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }
    v
}

fn copy_logical_strided_c32(src: &[f32], inc: usize, len_logical: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * len_logical);
    let mut idx = 0usize;
    for _ in 0..len_logical {
        out.push(src[2 * idx]);
        out.push(src[2 * idx + 1]);
        idx += inc;
    }
    out
}

fn assert_allclose_c32(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len() % 2, 0, "complex interleaved length must be even");
    for i in (0..a.len()).step_by(2) {
        for c in 0..2 {
            let x = a[i + c];
            let y = b[i + c];
            let diff = (x - y).abs();
            let tol  = atol + rtol * x.abs().max(y.abs());
            assert!(
                diff <= tol,
                "mismatch at complex idx {} [{}]: {} vs {} (|Δ|={}, tol={})",
                i / 2,
                if c == 0 { "re" } else { "im" },
                x, y, diff, tol
            );
        }
    }
}

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-5;

// tests

#[test]
fn upper_notranspose_small() {
    let n   = 6usize;
    let lda = n;
    let a   = make_upper_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [0.1 + 0.2 * (k as f32), -0.05 + 0.03 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn upper_transpose_small() {
    let n   = 7usize;
    let lda = n;
    let a   = make_upper_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [0.3 - 0.05 * (k as f32), 0.02 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn upper_conjtranspose_small() {
    let n   = 5usize;
    let lda = n;
    let a   = make_upper_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [-0.1 + 0.06 * (k as f32), 0.05 - 0.01 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::ConjugateTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn lower_notranspose_small() {
    let n   = 6usize;
    let lda = n;
    let a   = make_lower_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [0.02 * (k as f32), -0.2 + 0.07 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn lower_transpose_small() {
    let n   = 7usize;
    let lda = n;
    let a   = make_lower_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [0.11 - 0.03 * (k as f32), -0.04 + 0.02 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn lower_conjtranspose_small() {
    let n   = 5usize;
    let lda = n;
    let a   = make_lower_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [-0.15 + 0.04 * (k as f32), 0.03 + 0.01 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::ConjugateTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn upper_notranspose_large() {
    let n   = 768usize;
    let lda = n;
    let a   = make_upper_c32(n, lda);
    let x0  = (0..n)
        .flat_map(|k| [0.02 + 0.003 * (k as f32), -0.01 + 0.0007 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn strided_upper_conjtranspose() {
    let n    = 384usize;
    let lda  = n;
    let a    = make_upper_c32(n, lda);

    let incx = 3usize;
    let x    = make_strided_vec_c32(
        n, 
        incx, 
        |k| (0.05 + 0.01 * (k as f32), -0.02 + 0.004 * (k as f32))
    );

    let mut x_coral = x.clone();
    ctrmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::ConjugateTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        incx as i32,
    );

    let x_coral_logical = copy_logical_strided_c32(&x_coral, incx, n);
    let x_ref_logical   = copy_logical_strided_c32(&x_ref,   incx, n);
    assert_allclose_c32(&x_coral_logical, &x_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_lower_notranspose() {
    let n    = 320usize;
    let lda  = n;
    let a    = make_lower_c32(n, lda);

    let incx = 2usize;
    let x    = make_strided_vec_c32(
        n,
        incx, 
        |k| (-0.03 + 0.001 * (k as f32), 0.02 - 0.0005 * (k as f32))
    );

    let mut x_coral = x.clone();
    ctrmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        incx as i32,
    );

    let x_coral_logical = copy_logical_strided_c32(&x_coral, incx, n);
    let x_ref_logical   = copy_logical_strided_c32(&x_ref,   incx, n);
    assert_allclose_c32(&x_coral_logical, &x_ref_logical, RTOL, ATOL);
}

#[test]
fn strided_upper_transpose() {
    let n    = 256usize;
    let lda  = n;
    let a    = make_upper_c32(n, lda);

    let incx = 4usize;
    let x    = make_strided_vec_c32(
        n,
        incx, |k| (0.04 - 0.002 * (k as f32), -0.01 + 0.001 * (k as f32))
    );

    let mut x_coral = x.clone();
    ctrmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        incx as i32,
    );

    let x_coral_logical = copy_logical_strided_c32(&x_coral, incx, n);
    let x_ref_logical   = copy_logical_strided_c32(&x_ref,   incx, n);
    assert_allclose_c32(&x_coral_logical, &x_ref_logical, RTOL, ATOL);
}

#[test]
fn unitdiag_upper_conjtranspose() {
    let n   = 40usize;
    let lda = n;

    let mut a = make_upper_unit_c32(n, lda);
    // funky diagonal (ignored by unit)
    for i in 0..n {
        let idx = 2 * (i + i * lda);
        a[idx]     = 7.5 + (i as f32) * 0.1;
        a[idx + 1] = -3.2 + (i as f32) * 0.07;
    }

    let x0 = (0..n)
        .flat_map(|k| [0.2 + 0.01 * (k as f32), -0.03 + 0.002 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::ConjugateTranspose,
        CoralDiagonal::UnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn unitdiag_lower_notranspose() {
    let n   = 48usize;
    let lda = n;

    let mut a = make_lower_unit_c32(n, lda);
    // funky diagonal (ignored by unit)
    for i in 0..n {
        let idx = 2 * (i + i * lda);
        a[idx]     = -9.0 + (i as f32) * 0.05;
        a[idx + 1] =  2.1 - (i as f32) * 0.03;
    }

    let x0 = (0..n)
        .flat_map(|k| [-0.1 + 0.02 * (k as f32), 0.05 - 0.004 * (k as f32)])
        .collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    ctrmv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::UnitDiagonal,
        n,
        &a,
        lda,
        &mut x_coral,
        1,
    );

    let mut x_ref = x0.clone();
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let n   = 0usize;
    let lda = 1usize; // arbitrary when n == 0

    let a  = vec![0.0f32; 2 * lda]; // dummy
    let x0 = vec![0.0f32; 2];       // dummy complex

    let mut x_coral = x0.clone();
    ctrmv(
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
    cblas_ctrmv_wrapper(
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

