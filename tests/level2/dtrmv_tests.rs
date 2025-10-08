use blas_src as _;
use cblas_sys::{ CBLAS_TRANSPOSE, CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_DIAG, cblas_dtrmv };
use coral::enums::{ CoralDiagonal, CoralTranspose, CoralTriangular };
use coral::level2::dtrmv::dtrmv;

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
fn to_cblas_trans(
    t : CoralTranspose,
) -> CBLAS_TRANSPOSE {
    match t {
        CoralTranspose::NoTranspose          => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose            => CBLAS_TRANSPOSE::CblasTrans,
        CoralTranspose::ConjugateTranspose   => CBLAS_TRANSPOSE::CblasConjTrans,
    }
}

#[inline(always)]
fn to_cblas_diag(
    d : CoralDiagonal,
) -> CBLAS_DIAG {
    match d {
        CoralDiagonal::UnitDiagonal     => CBLAS_DIAG::CblasUnit,
        CoralDiagonal::NonUnitDiagonal  => CBLAS_DIAG::CblasNonUnit,
    }
}

#[inline(always)]
fn cblas_dtrmv_ref(
    tri   : CoralTriangular,
    trans : CoralTranspose,
    diag  : CoralDiagonal,
    n     : i32,
    a     : *const f64,
    lda   : i32,
    x     : *mut f64,
    incx  : i32,
) {
    unsafe {
        cblas_dtrmv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas_uplo(tri),
            to_cblas_trans(trans),
            to_cblas_diag(diag),
            n,
            a,
            lda,
            x,
            incx,
        );
    }
}

fn make_upper_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..=j {
            a[i + j * lda] = 0.2 + (i as f64) * 0.3 + (j as f64) * 0.15;
        }
    }
    a
}

fn make_lower_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in j..n {
            a[i + j * lda] = -0.1 + (i as f64) * 0.25 - (j as f64) * 0.2;
        }
    }
    a
}

// both triangles populated to exercise lda > n
fn make_padded_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * lda] = if i >= j {
                // lower triangle
                0.1 + (i as f64) * 0.2 + (j as f64) * 0.1
            } else {
                // upper triangle
                0.1 + (i as f64) * 0.1 + (j as f64) * 0.2
            };
        }
    }
    a
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v = vec![0.0; (len_logical - 1) * inc + 1];
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
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})"
        );
    }
}

const RTOL: f64 = 1e-13;
const ATOL: f64 = 1e-13;

fn build_matrix(
    tri  : CoralTriangular,
    diag : CoralDiagonal,
    n    : usize,
    lda  : usize,
) -> Vec<f64> {
    let mut a = match tri {
        CoralTriangular::UpperTriangular => make_upper_col_major(n, lda),
        CoralTriangular::LowerTriangular => make_lower_col_major(n, lda),
    };
    if let CoralDiagonal::UnitDiagonal = diag {
        // funky diagonal to ensure unit diagonal semantics
        for i in 0..n {
            a[i + i * lda] = if matches!(tri, CoralTriangular::UpperTriangular) {
                7.5 + (i as f64) * 0.1
            } else {
                -9.0 + (i as f64) * 0.05
            };
        }
    }
    a
}

fn run_case(
    tri   : CoralTriangular,
    trans : CoralTranspose,
    diag  : CoralDiagonal,
    n     : usize,
    lda   : usize,
    incx  : usize,
) {
    let a  = build_matrix(tri, diag, n, lda);
    let x0 = make_strided_vec(n, incx, |k| 0.1 + 0.2 * (k as f64));

    let mut x_coral = x0.clone();
    dtrmv(
        tri,
        trans,
        diag,
        n,
        &a,
        lda,
        &mut x_coral,
        incx,
    );

    let mut x_ref = x0.clone();
    cblas_dtrmv_ref(
        tri,
        trans,
        diag,
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

fn run_all(
    n           : usize,
    lda         : usize,
    stride_list : &[usize],
) {
    let tris   = [CoralTriangular::UpperTriangular, CoralTriangular::LowerTriangular];
    let transs = [
        CoralTranspose::NoTranspose,
        CoralTranspose::Transpose,
        CoralTranspose::ConjugateTranspose,
    ];
    let diags  = [CoralDiagonal::NonUnitDiagonal, CoralDiagonal::UnitDiagonal];

    for &incx in stride_list {
        for &tri in &tris {
            for &trans in &transs {
                for &diag in &diags {
                    run_case(tri, trans, diag, n, lda, incx);
                }
            }
        }
    }
}


#[test]
fn small_all() {
    run_all(7, 7, &[1]);
}

#[test]
fn large_all() {
    run_all(1024, 1024, &[1]);
}

#[test]
fn strided_all() {
    run_all(640, 640, &[2, 3]);
}

#[test]
fn padded_lower_transpose() {
    let n   = 127;
    let lda = n + 5; // padded leading dimension

    let a  = make_padded_col_major(n, lda);
    let x0 = (0..n).map(|k| 0.3 - 0.006 * (k as f64)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    dtrmv(
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
    cblas_dtrmv_ref(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
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
    let n   = 0;
    let lda = 1; // arbitrary when n == 0

    let a  = vec![0.0; lda];
    let x0 = vec![0.0; 1];

    let mut x_coral = x0.clone();
    dtrmv(
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
    cblas_dtrmv_ref(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose(&x_coral, &x_ref, RTOL, ATOL);
}


