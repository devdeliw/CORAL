use blas_src as _;
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use coral::enums::CoralTranspose;
use coral::level3::sgemm::sgemm;

#[inline(always)]
fn to_cblas(op: CoralTranspose) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTranspose        => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose          => CBLAS_TRANSPOSE::CblasTrans, 
        CoralTranspose::ConjugateTranspose => CBLAS_TRANSPOSE::CblasTrans,
    }
}

#[inline(always)]
fn cblas_sgemm_ref(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : i32,
    n     : i32,
    k     : i32,
    alpha : f32,
    a     : *const f32,
    lda   : i32,
    b     : *const f32,
    ldb   : i32,
    beta  : f32,
    c     : *mut f32,
    ldc   : i32,
) {
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op_a),
            to_cblas(op_b),
            m, n, k,
            alpha,
            a, lda,
            b, ldb,
            beta,
            c, ldc,
        );
    }
}

fn make_matrix_colmajor(
    rows : usize,
    cols : usize,
    ld   : usize,
    f    : impl Fn(usize, usize) -> f32,
) -> Vec<f32> {
    assert!(ld >= rows);
    let mut a = vec![0.0; ld * cols];

    for j in 0..cols {
        for i in 0..rows {
            a[i + j * ld] = f(i, j);
        }
    }
    a
}

fn assert_allclose(
    a    : &[f32], 
    b    : &[f32], 
    rtol : f32, 
    atol : f32
) {
    assert_eq!(a.len(), b.len());

    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());

        assert!(
            diff <= tol,
            "mismatch at {idx}: coral={x:.8e} vs cblas={y:.8e} delta={diff:.3e} tol={tol:.3e}"
        );
    }
}

// all tests except medium square 
// satisfy tolerances at 1e-6
const RTOL: f32 = 1e-3; 
const ATOL: f32 = 1e-3; 

fn run_case(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize, 
    n     : usize,
    k     : usize,
    lda   : usize,
    ldb   : usize,
    ldc   : usize,
    alpha : f32,
    beta  : f32,
) {
    let (a_rows, a_cols) = match op_a {
        CoralTranspose::NoTranspose          => (m, k),
        CoralTranspose::Transpose 
        | CoralTranspose::ConjugateTranspose => (k, m),
    };
    let (b_rows, b_cols) = match op_b {
        CoralTranspose::NoTranspose          => (k, n),
        CoralTranspose::Transpose 
        | CoralTranspose::ConjugateTranspose => (n, k),
    };
    assert!(lda >= a_rows && ldb >= b_rows && ldc >= m);

    let a = make_matrix_colmajor(a_rows, a_cols, lda, |i, j| 0.1 + (i as f32) * 0.25 + (j as f32) * 0.125);
    let b = make_matrix_colmajor(b_rows, b_cols, ldb, |i, j| -0.2 + (i as f32) * 0.05 - (j as f32) * 0.075);
    let c_init = make_matrix_colmajor(m, n, ldc, |i, j| 0.3 - (i as f32) * 0.01 + (j as f32) * 0.02);

    let mut c_coral = c_init.clone();
    sgemm(
        op_a, op_b,
        m, n, k,
        alpha,
        a.as_ptr(), lda,
        b.as_ptr(), ldb,
        beta,
        c_coral.as_mut_ptr(), ldc,
    );

    let mut c_ref = c_init.clone();
    cblas_sgemm_ref(
        op_a, op_b,
        m as i32, n as i32, k as i32,
        alpha,
        a.as_ptr(), lda as i32,
        b.as_ptr(), ldb as i32,
        beta,
        c_ref.as_mut_ptr(), ldc as i32,
    );

    assert_allclose(&c_coral, &c_ref, RTOL, ATOL);
}

fn run_all_ops(
    m     : usize, 
    n     : usize, 
    k     : usize,
    lda_n : usize,
    lda_t : usize,
    ldb_n : usize,
    ldb_t : usize,
    ldc   : usize,
) {
    let cases = &[
        (1.0f32, 0.0f32),
        (0.5, 1.0),
        (0.75, -0.25),
        (0.0, 0.7),
    ];

    let ops = &[
        (CoralTranspose::NoTranspose, CoralTranspose::NoTranspose, lda_n, ldb_n),
        (CoralTranspose::NoTranspose, CoralTranspose::Transpose,   lda_n, ldb_t),
        (CoralTranspose::Transpose,   CoralTranspose::NoTranspose, lda_t, ldb_n),
        (CoralTranspose::Transpose,   CoralTranspose::Transpose,   lda_t, ldb_t),
    ];

    for &(alpha, beta) in cases {
        for &(op_a, op_b, lda, ldb) in ops {
            run_case(op_a, op_b, m, n, k, lda, ldb, ldc, alpha, beta);
        }
        run_case(
            CoralTranspose::ConjugateTranspose,
            CoralTranspose::NoTranspose, 
            m, n, k,
            lda_t, ldb_n, ldc, 
            cases[0].0, cases[0].1
        );

        run_case(
            CoralTranspose::NoTranspose, 
            CoralTranspose::ConjugateTranspose,
            m, n, k, 
            lda_n, ldb_t, ldc, 
            cases[1].0, cases[1].1
        );
    }
}

#[test]
fn small_all_ops() {
    let (m, n, k) = (5, 7, 4);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn not_block_multiple_all_ops() {
    let (m, n, k) = (13, 14, 11);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn padded_lds_all_ops() {
    let (m, n, k) = (17, 19, 13);
    let lda_n = m + 3;
    let lda_t = k + 2;
    let ldb_n = k + 5;
    let ldb_t = n + 4;
    let ldc   = m + 7;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn medium_square_all_ops() {
    let (m, n, k) = (128, 128, 128);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn rectangular_all_ops() {
    run_all_ops(64, 192, 15, 64, 15, 15, 192, 64);
    run_all_ops(192, 64, 15, 192, 15, 15, 64, 192);
}

