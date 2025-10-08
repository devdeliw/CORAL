use blas_src as _;
use cblas_sys::{ CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_DIAG, CBLAS_UPLO, cblas_strsv };
use coral::enums::{ CoralDiagonal, CoralTranspose, CoralTriangular };
use coral::level2::strsv::strsv;

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
        CoralTranspose::NoTranspose        => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose          => CBLAS_TRANSPOSE::CblasTrans,
        CoralTranspose::ConjugateTranspose => CBLAS_TRANSPOSE::CblasTrans, 
    }
}

#[inline(always)]
fn to_cblas_diag(
    d : CoralDiagonal,
) -> CBLAS_DIAG {
    match d {
        CoralDiagonal::UnitDiagonal    => CBLAS_DIAG::CblasUnit,
        CoralDiagonal::NonUnitDiagonal => CBLAS_DIAG::CblasNonUnit,
    }
}

#[inline(always)]
fn cblas_strsv_ref(
    tri   : CoralTriangular,
    trans : CoralTranspose,
    diag  : CoralDiagonal,
    n     : i32,
    a     : *const f32,
    lda   : i32,
    x     : *mut f32,
    incx  : i32,
) {
    unsafe {
        cblas_strsv(
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


fn make_upper(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a: Vec<f32> = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..=j {
            let d    = (j - i) as f32;
            let diag = 1.0 + (0.02 as f32) * (i as f32);
            // gently decays away from the diagonal
            let off  = (0.01 as f32) / ((1.0 as f32) + d);
            a[i + j * lda] = if i == j { diag } else { off };
        }
    }
    a
}

fn make_lower(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a: Vec<f32> = vec![0.0; lda * n];
    for j in 0..n {
        for i in j..n {
            let d    = (i - j) as f32;
            let diag = 1.0 + (0.02 as f32) * (i as f32);
            // gently decays away from the diagonal
            let off  = (0.01 as f32) / ((1.0 as f32) + d);
            a[i + j * lda] = if i == j { diag } else { off };
        }
    }
    a
}

// both triangles populated to exercise lda > n
fn make_padded(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a: Vec<f32> = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                a[i + j * lda] = 1.0 + (0.02 as f32) * (i as f32);
            } else if i > j {
                // lower triangle
                let d = (i - j) as f32;
                a[i + j * lda] = (0.01 as f32) / ((1.0 as f32) + d);
            } else {
                // upper triangle
                let d = (j - i) as f32;
                a[i + j * lda] = (0.01 as f32) / ((1.0 as f32) + d);
            }
        }
    }
    a
}

// well-conditioned upper for unit diag tests
// strong unit diagonal; strictly-upper 
// increases mildly with distance
fn make_upper_unit(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a: Vec<f32> = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..=j {
            if i == j {
                // ignored by unit diag 
                a[i + j * lda] = 1.0 + (0.05 as f32) * (i as f32);
            } else {
                // small, increasing mildly with distance
                let d = (j - i) as f32;
                a[i + j * lda] = (5e-4 as f32) * ((1.0 as f32) + d);
            }
        }
    }
    a
}

// well-conditioned lower for unit-diag tests
// strong unit diagonal; strictly-lower 
// increases mildly with distance
fn make_lower_unit(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a: Vec<f32> = vec![0.0; lda * n];
    for j in 0..n {
        for i in j..n {
            if i == j {
                // ignored by unit diag 
                a[i + j * lda] = 1.0 - (0.03 as f32) * (i as f32);
            } else {
                // small, increasing mildly with distance
                let d = (i - j) as f32;
                a[i + j * lda] = (6e-4 as f32) * ((1.0 as f32) + d);
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
    let mut v: Vec<f32> = vec![0.0; len_logical.saturating_sub(1) * inc + 1];
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
            "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})"
        );
    }
}

const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-4;

fn build_matrix(
    tri  : CoralTriangular,
    diag : CoralDiagonal,
    n    : usize,
    lda  : usize,
) -> Vec<f32> {
    match (tri, diag) {
        (CoralTriangular::UpperTriangular, CoralDiagonal::NonUnitDiagonal) => make_upper(n, lda),
        (CoralTriangular::LowerTriangular, CoralDiagonal::NonUnitDiagonal) => make_lower(n, lda),
        (CoralTriangular::UpperTriangular, CoralDiagonal::UnitDiagonal)    => {
            let mut a = make_upper_unit(n, lda);
            // funky diagonal (ignored by unit)
            for i in 0..n {
                a[i + i * lda] = 7.5 + (i as f32) * (0.1 as f32);
            }
            a
        }
        (CoralTriangular::LowerTriangular, CoralDiagonal::UnitDiagonal)    => {
            let mut a = make_lower_unit(n, lda);
            // funky diagonal (ignored by unit)
            for i in 0..n {
                a[i + i * lda] = -9.0 + (i as f32) * (0.05 as f32);
            }
            a
        }
    }
}

fn run_case(
    tri   : CoralTriangular,
    trans : CoralTranspose,
    diag  : CoralDiagonal,
    n     : usize,
    lda   : usize,
    incx  : usize,
    x_gen : impl Fn(usize) -> f32,
) {
    let a  = build_matrix(tri, diag, n, lda);
    let x0 = make_strided_vec(n, incx, x_gen);

    let mut x_coral = x0.clone();
    strsv(
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
    cblas_strsv_ref(
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
    let transs = [CoralTranspose::NoTranspose, CoralTranspose::Transpose];
    let diags  = [CoralDiagonal::NonUnitDiagonal, CoralDiagonal::UnitDiagonal];

    for &incx in stride_list {
        for &tri in &tris {
            for &trans in &transs {
                for &diag in &diags {
                    run_case(
                        tri,
                        trans,
                        diag,
                        n,
                        lda,
                        incx,
                        |k| (0.05 as f32) + (0.01 as f32) * (k as f32),
                    );
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
    run_all(640, 640, &[2, 3, 4]);
}

#[test]
fn padded_all() {
    run_all(127, 127 + 5, &[1]);
}

#[test]
fn lower_transpose_padded() {
    // both triangles populated to exercise lda > n 
    // and triangle selection
    let n   = 127;
    let lda = n + 5;

    let a  = make_padded(n, lda);
    let x0 = (0..n).map(|k| (0.3 as f32) - (0.006 as f32) * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strsv(
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
    cblas_strsv_ref(
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
fn unitdiag_upper_notranspose() {
    let n   = 40;
    let lda = n;

    let a  = build_matrix(CoralTriangular::UpperTriangular, CoralDiagonal::UnitDiagonal, n, lda);
    let x0 = (0..n).map(|k| (0.2 as f32) + (0.01 as f32) * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strsv(
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
    cblas_strsv_ref(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::UnitDiagonal,
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
    let n   = 48;
    let lda = n;

    let a  = build_matrix(CoralTriangular::LowerTriangular, CoralDiagonal::UnitDiagonal, n, lda);
    let x0 = (0..n).map(|k| -(0.1 as f32) + (0.02 as f32) * (k as f32)).collect::<Vec<_>>();

    let mut x_coral = x0.clone();
    strsv(
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
    cblas_strsv_ref(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::UnitDiagonal,
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

    let a  = vec![0.0; lda]; // dummy
    let x0 = vec![1.23; 1];

    let mut x_coral = x0.clone();
    strsv(
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
    cblas_strsv_ref(
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

