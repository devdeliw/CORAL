use blas_src as _;
use cblas_sys::{ cblas_zgemv, CBLAS_LAYOUT, CBLAS_TRANSPOSE };
use coral::level2::zgemv::zgemv;
use coral::enums::CoralTranspose;

#[inline(always)]
fn to_cblas(
    op : CoralTranspose,
) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTranspose          => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Transpose            => CBLAS_TRANSPOSE::CblasTrans,
        CoralTranspose::ConjugateTranspose   => CBLAS_TRANSPOSE::CblasConjTrans,
    }
}

#[inline(always)]
fn cblas_zgemv_ref(
    op    : CoralTranspose,
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    a     : *const f64,
    lda   : i32,
    x     : *const f64,
    incx  : i32,
    beta  : [f64; 2],
    y     : *mut f64,
    incy  : i32,
) {
    unsafe {
        cblas_zgemv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op),
            m,
            n,
            alpha.as_ptr() as *const [f64; 2],
            a              as *const [f64; 2],
            lda,
            x              as *const [f64; 2],
            incx,
            beta.as_ptr()  as *const [f64; 2],
            y as *mut [f64; 2],
            incy,
        );
    }
}

fn make_zmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    assert!(lda >= m);

    let mut a = vec![0.0; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            a[idx]     = 0.1 + (i as f64) * 0.5 + (j as f64) * 0.25;
            a[idx + 1] = -0.2 + (i as f64) * 0.3 - (j as f64) * 0.15;
        }
    }

    a
}

fn make_strided_zvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f64; 2],
) -> Vec<f64> {
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

fn copy_logical_strided_z(
    src         : &[f64],
    inc         : usize,
    len_logical : usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * len_logical);
    let mut idx = 0;

    for _ in 0..len_logical {
        out.push(src[2 * idx]);
        out.push(src[2 * idx + 1]);
        idx += inc;
    }

    out
}

fn assert_allclose_z(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);

    for i in (0..a.len()).step_by(2) {
        let xr = a[i];
        let xi = a[i + 1];
        let yr = b[i];
        let yi = b[i + 1];

        let dr = (xr - yr).abs();
        let di = (xi - yi).abs();

        let tr = atol + rtol * xr.abs().max(yr.abs());
        let ti = atol + rtol * xi.abs().max(yi.abs());

        assert!(
            dr <= tr, 
            "re mismatch at pair {}: {xr} vs {yr} (delta={dr}, tol={tr})", i / 2
        );
        assert!(
            di <= ti,
            "im mismatch at pair {}: {xi} vs {yi} (delta={di}, tol={ti})", i / 2
        );
    }
}

const RTOL: f64 = 1e-11;
const ATOL: f64 = 1e-12;

#[inline(always)]
fn xy_lengths(
    op : CoralTranspose,
    m  : usize,
    n  : usize,
) -> (usize, usize) {
    match op {
        CoralTranspose::NoTranspose          => (n, m),
        CoralTranspose::Transpose            => (m, n), 
        CoralTranspose::ConjugateTranspose   => (m, n),
    }
}


fn run_case(
    op    : CoralTranspose,
    m     : usize,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : [f64; 2],
    beta  : [f64; 2],
) {
    let a = make_zmatrix(m, n, lda);
    let (xlen, ylen) = xy_lengths(op, m, n);

    let x  = make_strided_zvec(
        xlen, 
        incx, 
        |k| [0.2 + 0.1 * (k as f64), 
        -0.05 * (k as f64)]
    );
    let y0 = make_strided_zvec(
        ylen, 
        incy, 
        |k| [-0.3 + 0.05 * (k as f64),
        0.02 * (k as f64)]
    );

    // coral
    let mut y_coral = y0.clone();
    zgemv(
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

    // reference
    let mut y_ref = y0.clone();
    cblas_zgemv_ref(
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

    let yc = copy_logical_strided_z(&y_coral, incy, ylen);
    let yr = copy_logical_strided_z(&y_ref,   incy, ylen);
    assert_allclose_z(&yc, &yr, RTOL, ATOL);
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
        CoralTranspose::ConjugateTranspose,
    ];

    // include alpha=0, beta=0, and mixed cases
    let coeffs: [([f64; 2], [f64; 2]); 6] = [
        ([1.0,   0.0],  [0.0,   0.0]),
        ([0.0,   0.0],  [-0.75,  0.2]),
        ([0.75, -0.10], [-0.25,  0.05]),
        ([-0.6,  0.2],  [0.4,   -0.1]),
        ([1.1,  -0.6],  [0.0,   0.0]),
        ([0.85, -0.2],  [0.1,   0.0]),
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
    run_all_ops(5, 4, 5, &[(1, 1)]);
}

#[test]
fn large_all_ops() {
    run_all_ops(384, 256, 384, &[(1, 1)]);
}

#[test]
fn padded_all_ops() {
    run_all_ops(256, 512, 256 + 7, &[(1, 1)]);
}

#[test]
fn strided_all_ops() {
    run_all_ops(512, 256, 512, &[(2, 3), (3, 2), (2, 1)]);
}

#[test]
fn rectangular_all_ops() {
    run_all_ops(640, 320, 640, &[(1, 1)]);
}

