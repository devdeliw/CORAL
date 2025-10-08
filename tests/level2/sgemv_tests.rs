use blas_src as _;
use cblas_sys::{ cblas_sgemv, CBLAS_LAYOUT, CBLAS_TRANSPOSE };
use coral::enums::CoralTranspose;
use coral::level2::sgemv::sgemv;

#[inline(always)]
fn to_cblas(
    op : CoralTranspose,
) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTranspose        => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose          => CBLAS_TRANSPOSE::CblasTrans, 
        CoralTranspose::ConjugateTranspose => CBLAS_TRANSPOSE::CblasTrans,
    }
}

#[inline(always)]
fn cblas_sgemv_ref(
    op    : CoralTranspose,
    m     : i32,
    n     : i32,
    alpha : f32,
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : f32,
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op),
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
}

fn make_matrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= m);
    let mut a = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * lda] = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;
        }
    }

    a
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f32,
) -> Vec<f32> {
    assert!(len_logical > 0 && inc > 0);
    let mut v = vec![0.0; (len_logical - 1) * inc + 1];

    let mut idx = 0;
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

    let mut idx = 0;
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
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());

        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})"
        );
    }
}

const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-5;

#[inline(always)]
fn xy_lengths(
    op : CoralTranspose,
    m  : usize,
    n  : usize,
) -> (usize, usize) {
    match op {
        CoralTranspose::NoTranspose        => (n, m),
        CoralTranspose::Transpose          => (m, n), 
        CoralTranspose::ConjugateTranspose => (m, n),
    }
}

fn run_case(
    op    : CoralTranspose,
    m     : usize,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : f32,
    beta  : f32,
) {
    let a = make_matrix(m, n, lda);
    let (xlen, ylen) = xy_lengths(op, m, n);

    let x = make_strided_vec(
        xlen,
        incx,
        |k| 0.2 + 0.1 * (k as f32),
    );

    let y0 = make_strided_vec(
        ylen,
        incy,
        |k| -0.3 + 0.05 * (k as f32),
    );

    let mut y_coral = y0.clone();
    sgemv(
        op,
        m,
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

    let mut y_ref = y0.clone();
    cblas_sgemv_ref(
        op,
        m as i32,
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

    let yc = copy_logical_strided(&y_coral, incy, ylen);
    let yr = copy_logical_strided(&y_ref,   incy, ylen);
    assert_allclose(&yc, &yr, RTOL, ATOL);
}

fn run_all_ops(
    m            : usize,
    n            : usize,
    lda          : usize,
    stride_pairs : &[(usize, usize)],
) {
    let ops = [
        CoralTranspose::NoTranspose,
        CoralTranspose::Transpose,
    ];

    // alpha=0, beta=0, and mixed cases 
    let coeffs: &[(f32, f32)] = &[
        (1.0,   0.0),   
        (0.0,  -0.75),  
        (0.75, -0.25),
        (-0.6,  0.4),
        (1.1,   0.0),   
        (0.85,  0.1),
    ];

    for &(incx, incy) in stride_pairs {
        for &op in &ops {
            for &(alpha, beta) in coeffs {
                run_case(op, m, n, lda, incx, incy, alpha, beta);
            }
        }
    }
}

#[test]
fn small_all_ops() {
    run_all_ops(5, 4, 5, &[(1, 1)]);
}

#[test]
fn large_all_ops() {
    run_all_ops(1024, 512, 1024, &[(1, 1)]);
}

#[test]
fn padded_all_ops() {
    run_all_ops(256, 1024, 256 + 7, &[(1, 1)]);
}

#[test]
fn strided_all_ops() {
    run_all_ops(1024, 512, 1024, &[(2, 3), (3, 2), (2, 1)]);
}

#[test]
fn rectangular_all_ops() {
    run_all_ops(640, 320, 640, &[(1, 1)]);
}

