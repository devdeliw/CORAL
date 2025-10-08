use blas_src as _;
use cblas_sys::{cblas_cgemv, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use coral::enums::CoralTranspose;
use coral::level2::cgemv::cgemv;

#[inline(always)]
fn to_cblas(op: CoralTranspose) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTranspose          => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose            => CBLAS_TRANSPOSE::CblasTrans,
        CoralTranspose::ConjugateTranspose   => CBLAS_TRANSPOSE::CblasConjTrans,
    }
}

#[inline(always)]
fn cblas_cgemv_ref(
    op    : CoralTranspose,
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : i32,
    x     : *const f32,
    incx  : i32,
    beta  : [f32; 2],
    y     : *mut f32,
    incy  : i32,
) {
    unsafe {
        cblas_cgemv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op),
            m, n,
            alpha.as_ptr() as *const [f32; 2],
            a              as *const [f32; 2],
            lda,
            x              as *const [f32; 2],
            incx,
            beta.as_ptr()  as *const [f32; 2],
            y as *mut [f32; 2],
            incy,
        );
    }
}

fn make_cmatrix(
    m   : usize, 
    n   : usize, 
    lda : usize
) -> Vec<f32> {
    assert!(lda >= m);

    let mut a = vec![0.0; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            a[idx]     = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;   // re
            a[idx + 1] = -0.2 + (i as f32) * 0.3 - (j as f32) * 0.15;  // im
        }
    }
    a
}

fn make_strided_cvec(
    len_logical : usize, 
    inc         : usize, 
    f           : impl Fn(usize) -> [f32; 2]
) -> Vec<f32> {
    assert!(len_logical > 0 && inc > 0);

    let mut v = vec![0.0; 2 * ((len_logical - 1) * inc + 1)];
    let mut idx = 0;

    for k in 0..len_logical {
        let z = f(k);
        v[2 * idx]     = z[0];
        v[2 * idx + 1] = z[1];
        idx += inc;
    }

    v
}

fn copy_logical_strided_c(
    src         : &[f32], 
    inc         : usize,
    len_logical : usize
) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * len_logical);
    let mut idx = 0;

    for _ in 0..len_logical {
        out.push(src[2 * idx]);
        out.push(src[2 * idx + 1]);
        idx += inc;
    }

    out
}

fn assert_allclose_c(
    a    : &[f32], 
    b    : &[f32],
    rtol : f32,
    atol : f32
) {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);

    for i in (0..a.len()).step_by(2) {
        let (xr, xi) = (a[i], a[i + 1]);
        let (yr, yi) = (b[i], b[i + 1]);
        let dr = (xr - yr).abs();
        let di = (xi - yi).abs();
        let tr = atol + rtol * xr.abs().max(yr.abs());
        let ti = atol + rtol * xi.abs().max(yi.abs());

        assert!(dr <= tr, "re mismatch at pair {}: {xr} vs {yr} (delta={dr}, tol={tr})", i/2);
        assert!(di <= ti, "im mismatch at pair {}: {xi} vs {yi} (delta={di}, tol={ti})", i/2);
    }
}

// slightly looser
const RTOL: f32 = 5e-3;
const ATOL: f32 = 1e-6;

#[inline(always)]
fn xy_lengths(op: CoralTranspose, m: usize, n: usize) -> (usize, usize) {
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
    alpha : [f32; 2],
    beta  : [f32; 2],
) {
    assert!(lda >= m);
    let a = make_cmatrix(m, n, lda);
    let (xlen, ylen) = xy_lengths(op, m, n);

    // deterministic but nontrivial X/Y patterns
    let x = make_strided_cvec(xlen, incx, |k| [0.2 + 0.1 * (k as f32), -0.05 * (k as f32)]);
    let y0 = make_strided_cvec(ylen, incy, |k| [-0.3 + 0.05 * (k as f32), 0.02 * (k as f32)]);

    // coral
    let mut y_coral = y0.clone();
    cgemv(
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
        incy
    );

    // reference
    let mut y_ref = y0.clone();
    cblas_cgemv_ref(
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

    let yc = copy_logical_strided_c(&y_coral, incy, ylen);
    let yr = copy_logical_strided_c(&y_ref,   incy, ylen);
    assert_allclose_c(&yc, &yr, RTOL, ATOL);
}

fn run_all_ops(
    m     : usize,
    n     : usize,
    lda   : usize,
    stride_pairs: &[(usize, usize)],
) {
    let ops = [
        CoralTranspose::NoTranspose,
        CoralTranspose::Transpose,
        CoralTranspose::ConjugateTranspose,
    ];

    // includes alpha=0, beta=0, and mixed cases. 
    let coeffs: [([f32; 2], [f32; 2]); 6] = [
        ([1.0,  0.0], [0.0,  0.0]),   // pure product
        ([0.0,  0.0], [0.7, -0.1]),   // alpha=0 scale y
        ([0.75, -0.25], [0.5,  0.0]),
        ([-0.5,  0.2], [0.3, -0.15]),
        ([1.1, -0.6],  [0.0,  0.0]),  // beta=0 overwrite y
        ([0.3,  0.7],  [0.1, -0.2]),
    ];

    for &(incx, incy) in stride_pairs {
        for &op in &ops {
            for &(alpha, beta) in &coeffs {
                run_case(op, m, n, lda, incx, incy, alpha, beta);
            }
        }
    }
}



#[test]
fn small_all_ops() {
    let (m, n, lda) = (5, 4, 5);
    run_all_ops(m, n, lda, &[(1, 1)]);
}

#[test]
fn large_all_ops() {
    let (m, n, lda) = (384, 256, 384);
    run_all_ops(m, n, lda, &[(1, 1)]);
}

#[test]
fn padded_all_ops() {
    let (m, n, lda) = (256, 512, 256 + 7);
    run_all_ops(m, n, lda, &[(1, 1)]);
}

#[test]
fn strided_all_ops() {
    let (m, n, lda) = (300, 200, 300 + 5);
    
    // exercise nontrivial strides
    run_all_ops(m, n, lda, &[(2, 1), (1, 3), (2, 3)]);
}

#[test]
fn rectangular_all_ops() {
    let (m, n, lda) = (64, 192, 64);
    run_all_ops(m, n, lda, &[(1, 1)]);
}

