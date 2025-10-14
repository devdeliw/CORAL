use blas_src as _;
use cblas_sys::{cblas_cgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use coral::enums::CoralTranspose;
use coral::level3::cgemm::cgemm;

#[inline(always)]
fn to_cblas(op: CoralTranspose) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTranspose        => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose          => CBLAS_TRANSPOSE::CblasTrans,
        CoralTranspose::ConjugateTranspose => CBLAS_TRANSPOSE::CblasConjTrans,
    }
}

#[inline(always)]
fn cblas_cgemm_ref(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : i32,
    n     : i32,
    k     : i32,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : i32,
    b     : *const f32,
    ldb   : i32,
    beta  : [f32; 2],
    c     : *mut f32,
    ldc   : i32,
) {
    unsafe {
        cblas_cgemm(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op_a),
            to_cblas(op_b),
            m, n, k,
            &alpha as *const [f32; 2],
            a      as *const [f32; 2], 
            lda,
            b      as *const [f32; 2], 
            ldb,
            &beta  as *const [f32; 2],
            c      as *mut [f32; 2], 
            ldc,
        );
    }
}

fn make_matrix_colmajor_c32(
    rows : usize,
    cols : usize,
    ld   : usize,
    f    : impl Fn(usize, usize) -> [f32; 2],
) -> Vec<f32> {
    assert!(ld >= rows);
    let mut a = vec![0.0; 2 * ld * cols];
    for j in 0..cols {
        for i in 0..rows {
            let [re, im] = f(i, j);
            let idx = 2 * (i + j * ld);

            a[idx + 0] = re;
            a[idx + 1] = im;
        }
    }
    a
}

#[inline(always)]
fn opch(op: CoralTranspose) -> char {
    match op {
        CoralTranspose::NoTranspose        => 'N',
        CoralTranspose::Transpose          => 'T',
        CoralTranspose::ConjugateTranspose => 'C',
    }
}

fn assert_allclose(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
    ctx  : &str,
    ldc  : usize,
) {
    assert_eq!(a.len(), b.len());

    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        if diff > tol {
            let elem = idx / 2;
            let part = if (idx & 1) == 0 { "re" } else { "im" };
            let j    = elem / ldc;
            let i    = elem % ldc;

            panic!(
                "[{}] mismatch at (i={}, j={}, part={}): coral={:.8e} vs cblas={:.8e} \
                 delta={:.3e} tol={:.3e} (ldc={})",
                ctx, i, j, part, x, y, diff, tol, ldc
            );
        }
    }
}

// just to accomodate both openblas and accelerate 
// rtol = atol = 1e-3 works for openblas
const RTOL: f32 = 3e-3;
const ATOL: f32 = 2e-3;

fn run_case(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    lda   : usize,
    ldb   : usize,
    ldc   : usize,
    alpha : [f32; 2],
    beta  : [f32; 2],
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

    let a = make_matrix_colmajor_c32(a_rows, a_cols, lda, |i, j| {
        [
            0.1 + (i as f32) * 0.25 + (j as f32) * 0.125,
            -0.05 + (i as f32) * 0.20 - (j as f32) * 0.075
        ]
    });
    let b = make_matrix_colmajor_c32(b_rows, b_cols, ldb, |i, j| {
        [
            -0.2 + (i as f32) * 0.05 - (j as f32) * 0.075,
            0.15 - (i as f32) * 0.03 + (j as f32) * 0.02
        ]
    });
    let c_init = make_matrix_colmajor_c32(m, n, ldc, |i, j| {
        [
            0.3 - (i as f32) * 0.01 + (j as f32) * 0.02,
            -0.1 + (i as f32) * 0.015 - (j as f32) * 0.025
        ]
    });

    let mut c_coral = c_init.clone();
    cgemm(
        op_a, op_b,
        m, n, k,
        alpha,
        a.as_ptr(), lda,
        b.as_ptr(), ldb,
        beta,
        c_coral.as_mut_ptr(), ldc,
    );

    let mut c_ref = c_init.clone();
    cblas_cgemm_ref(
        op_a, op_b,
        m as i32, n as i32, k as i32,
        alpha,
        a.as_ptr(), lda as i32,
        b.as_ptr(), ldb as i32,
        beta,
        c_ref.as_mut_ptr(), ldc as i32,
    );

    let ctx = format!(
        "op_a={} op_b={} m={} n={} k={} lda={} ldb={} ldc={} \
         alpha=({:+.3e},{:+.3e}) beta=({:+.3e},{:+.3e})",
        opch(op_a), opch(op_b), m, n, k, lda, ldb, ldc,
        alpha[0], alpha[1], beta[0], beta[1]
    );

    assert_allclose(&c_coral, &c_ref, RTOL, ATOL, &ctx, ldc);
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
    let cases: &[[f32; 4]] = &[
        [1.0, 0.0, 0.0, 0.0],   
        [0.5, 0.25, 1.0, 0.0],  
        [0.75, -0.25, -0.5, 0.3],
        [0.0, 0.0, 0.7, -0.2],  
    ];

    let ops = &[
        (CoralTranspose::NoTranspose,        CoralTranspose::NoTranspose,        lda_n, ldb_n),
        (CoralTranspose::NoTranspose,        CoralTranspose::Transpose,          lda_n, ldb_t),
        (CoralTranspose::Transpose,          CoralTranspose::NoTranspose,        lda_t, ldb_n),
        (CoralTranspose::Transpose,          CoralTranspose::Transpose,          lda_t, ldb_t),
        (CoralTranspose::NoTranspose,        CoralTranspose::ConjugateTranspose, lda_n, ldb_t),
        (CoralTranspose::Transpose,          CoralTranspose::ConjugateTranspose, lda_t, ldb_t),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::NoTranspose,        lda_t, ldb_n),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::Transpose,          lda_t, ldb_t),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::ConjugateTranspose, lda_t, ldb_t),
    ];

    for case in cases {
        let alpha = [case[0], case[1]];
        let beta  = [case[2], case[3]];
        for &(op_a, op_b, lda, ldb) in ops {
            run_case(op_a, op_b, m, n, k, lda, ldb, ldc, alpha, beta);
        }
    }
}

#[test]
fn small_all_ops_c() {
    let (m, n, k) = (5, 7, 4);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn not_block_multiple_all_ops_c() {
    let (m, n, k) = (13, 14, 11);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn padded_lds_all_ops_c() {
    let (m, n, k) = (17, 19, 13);
    let lda_n = m + 3;
    let lda_t = k + 2;
    let ldb_n = k + 5;
    let ldb_t = n + 4;
    let ldc   = m + 7;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn medium_square_all_ops_c() {
    let (m, n, k) = (128, 128, 128);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc);
}

#[test]
fn rectangular_all_ops_c() {
    run_all_ops(64, 192, 15, 64, 15, 15, 192, 64);
    run_all_ops(192, 64, 15, 192, 15, 15, 64, 192);
}

