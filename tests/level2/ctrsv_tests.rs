use blas_src as _;
use cblas_sys::{ CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_DIAG, CBLAS_UPLO, cblas_ctrsv };
use coral::enums::{ CoralDiagonal, CoralTranspose, CoralTriangular };
use coral::level2::ctrsv::ctrsv;

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
        CoralTranspose::ConjugateTranspose => CBLAS_TRANSPOSE::CblasConjTrans,
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
fn cblas_ctrsv_ref(
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
        cblas_ctrsv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas_uplo(tri),
            to_cblas_trans(trans),
            to_cblas_diag(diag),
            n,
            a   as *const [f32; 2],
            lda,
            x   as *mut   [f32; 2],
            incx,
        );
    }
}

fn make_padded_c32(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);
            if i == j {
                // mildly varying diagonal
                a[idx]     = 1.0 + 0.02 * (i as f32);
                a[idx + 1] = 0.01 - 0.005 * (i as f32);
            } else if i < j {
                // small complex off-diagonal decaying with distance (upper)
                let d = (j - i) as f32;
                a[idx]     = 0.01 / (1.0 + d);
                a[idx + 1] = -(0.008 as f32) / (1.0 + d);
            } else {
                // small complex off-diagonal decaying with distance (lower)
                let d = (i - j) as f32;
                a[idx]     = 0.01 / (1.0 + d);
                a[idx + 1] = (0.007 as f32) / (1.0 + d);
            }
        }
    }
    a
}

fn make_upper_c32(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
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
                a[idx + 1] = -(0.008 as f32) / (1.0 + d);
            }
        }
    }
    a
}

fn make_lower_c32(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
    for j in 0..n {
        for i in j..n {
            let idx = 2 * (i + j * lda);
            if i == j {
                // mildly varying diagonal
                a[idx]     = 1.0 + 0.02 * (i as f32);
                a[idx + 1] = -0.012 + 0.004 * (i as f32);
            } else {
                // small complex off-diagonal decaying with distance
                let d = (i - j) as f32;
                a[idx]     = 0.01 / (1.0 + d);
                a[idx + 1] = (0.007 as f32) / (1.0 + d);
            }
        }
    }
    a
}

// well-conditioned upper for unit diag tests
fn make_upper_unit_c32(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
    for j in 0..n {
        for i in 0..=j {
            let idx = 2 * (i + j * lda);
            if i == j {
                // ignored by unit diag
                a[idx]     = 2.5 + 0.05 * (i as f32);
                a[idx + 1] = -1.7 + 0.01 * (i as f32);
            } else {
                // small complex off-diagonal decaying with distance
                let d = (j - i) as f32;
                a[idx]     = (5e-4 as f32) * (1.0 + d);
                a[idx + 1] = -(3e-4 as f32) * (1.0 + d);
            }
        }
    }
    a
}

// well-conditioned lower for unit diag tests
fn make_lower_unit_c32(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0; 2 * lda * n];
    for j in 0..n {
        for i in j..n {
            let idx = 2 * (i + j * lda);
            if i == j {
                // ignored by unit diag
                a[idx]     = -2.0 + 0.04 * (i as f32);
                a[idx + 1] =  1.2 - 0.02 * (i as f32);
            } else {
                // small complex off-diagonal decaying with distance
                let d = (i - j) as f32;
                a[idx]     = (6e-4 as f32) * (1.0 + d);
                a[idx + 1] = (4e-4 as f32) * (1.0 + d);
            }
        }
    }
    a
}

fn make_strided_vec_c32(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> (f32, f32),
) -> Vec<f32> {
    let mut v = vec![0.0; 2 * (len_logical.saturating_sub(1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let (re, im) = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }
    v
}

fn copy_logical_strided_c32(
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

fn assert_allclose_c32(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
) {
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
                "mismatch at complex idx {} [{}]: {} vs {} (delta={}, tol={})",
                i / 2,
                if c == 0 { "re" } else { "im" },
                x, y, diff, tol
            );
        }
    }
}

const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-3;

fn build_matrix(
    tri  : CoralTriangular,
    diag : CoralDiagonal,
    n    : usize,
    lda  : usize,
) -> Vec<f32> {
    match (tri, diag) {
        (CoralTriangular::UpperTriangular, CoralDiagonal::NonUnitDiagonal) => make_upper_c32(n, lda),
        (CoralTriangular::LowerTriangular, CoralDiagonal::NonUnitDiagonal) => make_lower_c32(n, lda),
        (CoralTriangular::UpperTriangular, CoralDiagonal::UnitDiagonal)    => {
            let mut a = make_upper_unit_c32(n, lda);
            // funky diagonal (ignored by unit)
            for i in 0..n {
                let idx = 2 * (i + i * lda);
                a[idx]     = 7.5 + (i as f32) * 0.1;
                a[idx + 1] = -3.2 + (i as f32) * 0.07;
            }
            a
        }
        (CoralTriangular::LowerTriangular, CoralDiagonal::UnitDiagonal)    => {
            let mut a = make_lower_unit_c32(n, lda);
            // funky diagonal (ignored by unit)
            for i in 0..n {
                let idx = 2 * (i + i * lda);
                a[idx]     = -9.0 + (i as f32) * 0.05;
                a[idx + 1] =  4.2 - (i as f32) * 0.03;
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
) {
    let a  = build_matrix(tri, diag, n, lda);
    let x0 = make_strided_vec_c32(
        n,
        incx,
        |k| (0.1 + 0.2 * (k as f32), -0.05 + 0.03 * (k as f32)),
    );

    let mut x_coral = x0.clone();
    ctrsv(
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
    cblas_ctrsv_ref(
        tri,
        trans,
        diag,
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
    run_all(768, 768, &[1]);
}

#[test]
fn strided_all() {
    run_all(384, 384, &[2, 3, 4]);
}

#[test]
fn padded_all() {
    run_all(256, 256 + 7, &[1]);
}

#[test]
fn lower_transpose_padded() {
    // both triangles populated to exercise lda > n
    // and triangle selection
    let n   = 127;
    let lda = n + 7;

    let a  = make_padded_c32(n, lda);
    let x0 = make_strided_vec_c32(
        n,
        1,
        |k| (0.3 - 0.006 * (k as f32), -0.12 + 0.003 * (k as f32)),
    );

    let mut x_coral = x0.clone();
    ctrsv(
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
    cblas_ctrsv_ref(
        CoralTriangular::LowerTriangular,
        CoralTranspose::Transpose,
        CoralDiagonal::NonUnitDiagonal,
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
    let n   = 0;
    let lda = 1; // arbitrary when n == 0

    let a  = vec![0.0; 2 * lda];
    let x0 = vec![1.23, -4.56]; // one complex number

    let mut x_coral = x0.clone();
    ctrsv(
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
    cblas_ctrsv_ref(
        CoralTriangular::UpperTriangular,
        CoralTranspose::ConjugateTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n as i32,
        a.as_ptr(),
        lda as i32,
        x_ref.as_mut_ptr(),
        1,
    );

    assert_allclose_c32(&x_coral, &x_ref, RTOL, ATOL);
}

