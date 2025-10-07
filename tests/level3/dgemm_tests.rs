use blas_src as _; 
use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE}; 
use coral::level3::dgemm_nn::dgemm_nn; 

// cblas wrapper 
#[inline(always)]
fn cblas_dgemm_nn(
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    unsafe {
        cblas_dgemm(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

// helpers 
fn make_matrix_colmajor(
    m: usize,
    n: usize,
    ld: usize,
    f: impl Fn(usize, usize) -> f64
) -> Vec<f64> {
    assert!(ld >= m);
    let mut a = vec![0.0f64; ld * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * ld] = f(i, j);
        }
    }
    a
}

fn assert_allclose(
    a: &[f64], 
    b: &[f64],
    rtol: f64, 
    atol: f64
) {
    assert_eq!(a.len(), b.len());
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {idx}: coral={x:.16e} vs cblas={y:.16e} delta={diff:.3e} tol={tol:.3e}"
        );
    }
}

const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-12;

fn run_case(
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    alpha: f64,
    beta: f64,
) {
    let a = make_matrix_colmajor(m, k, lda, |i, j| 0.1 + (i as f64) * 0.25 + (j as f64) * 0.125);
    let b = make_matrix_colmajor(k, n, ldb, |i, j| -0.2 + (i as f64) * 0.05 - (j as f64) * 0.075);

    // C non-zero to exercise beta
    let c_init = make_matrix_colmajor(m, n, ldc, |i, j| 0.3 - (i as f64) * 0.01 + (j as f64) * 0.02);

    let mut c_coral = c_init.clone();
    dgemm_nn(
        m,
        n,
        k,
        alpha,
        a.as_ptr(),
        lda,
        b.as_ptr(),
        ldb,
        beta,
        c_coral.as_mut_ptr(),
        ldc,
    );

    let mut c_ref = c_init.clone();
    cblas_dgemm_nn(
        m as i32,
        n as i32,
        k as i32,
        alpha,
        a.as_ptr(),
        lda as i32,
        b.as_ptr(),
        ldb as i32,
        beta,
        c_ref.as_mut_ptr(),
        ldc as i32,
    );

    assert_allclose(&c_coral, &c_ref, RTOL, ATOL);
}

#[test]
fn nn_small() {
    let (m, n, k) = (5, 7, 4);
    let lda = m;
    let ldb = k;
    let ldc = m;

    // generic alpha/beta
    run_case(m, n, k, lda, ldb, ldc, 0.75, -0.25);

    // beta = 0 fast-path
    run_case(m, n, k, lda, ldb, ldc, 1.0, 0.0);

    // beta = 1 fast-path
    run_case(m, n, k, lda, ldb, ldc, 0.5, 1.0);
}

#[test]
fn nn_not_block_multiple() {
    // sizes that are not multiples of MR, NR, KC
    let (m, n, k) = (13, 14, 11);
    let lda = m;
    let ldb = k;
    let ldc = m;

    run_case(m, n, k, lda, ldb, ldc, 1.25, -0.5);
}

#[test]
fn nn_padded() {
    let (m, n, k) = (17, 19, 13);
    let lda = m + 3; 
    let ldb = k + 5; 
    let ldc = m + 7; 

    run_case(m, n, k, lda, ldb, ldc, -0.3, 0.4);
}

#[test]
fn nn_medium_square() {
    let (m, n, k) = (128, 128, 128);
    let lda = m;
    let ldb = k;
    let ldc = m;

    run_case(m, n, k, lda, ldb, ldc, 0.9, 0.1);
}

#[test]
fn nn_rectangular() {
    run_case(64, 192, 15, 64, 15, 64, 1.0, 0.0);
    run_case(192, 64, 15, 192, 15, 192, 0.7, 0.3);
}
