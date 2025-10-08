use blas_src as _;
use cblas_sys::{ cblas_chemv, CBLAS_LAYOUT, CBLAS_UPLO };
use coral::level2::chemv::chemv;
use coral::enums::CoralTriangular;

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
fn cblas_chemv_ref(
    tri   : CoralTriangular,
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
        cblas_chemv(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas_uplo(tri),
            n,
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


fn make_hermitian_col_major(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= n);

    let mut a = vec![0.0; 2 * lda * n];

    for j in 0..n {
        for i in 0..=j {
            let lo  = i.min(j) as f32;
            let hi  = i.max(j) as f32;
            let re  = 0.10 + 0.25 * lo + 0.15 * hi;
            let im  = if i == j { 0.0 } else { 0.04 + 0.02 * (hi - lo) };

            let idx_ij = 2 * (i + j * lda);
            a[idx_ij]     = re;
            a[idx_ij + 1] = im;

            let idx_ji = 2 * (j + i * lda);
            a[idx_ji]     = re;
            a[idx_ji + 1] = -im; // conjugate
        }
    }

    a
}

// upper valid, lower = NaN garbage
fn make_upper_stored_lower_garbage_c(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= n);

    let mut a = vec![0.0; 2 * lda * n];

    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);

            if i <= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                let re = 0.20 + 0.30 * lo + 0.15 * hi;
                let im = if i == j { 0.0 } else { 0.05 + 0.01 * (hi - lo) };
                a[idx]     = re;
                a[idx + 1] = im;
            } else {
                a[idx]     = f32::NAN;
                a[idx + 1] = f32::NAN;
            }
        }
    }

    a
}

// lower valid, upper = NaN garbage
fn make_lower_stored_upper_garbage_c(
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= n);

    let mut a = vec![0.0; 2 * lda * n];

    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);

            if i >= j {
                let lo = i.min(j) as f32;
                let hi = i.max(j) as f32;
                let re = 0.17 + 0.45 * lo + 0.08 * hi;
                let im = if i == j { 0.0 } else { -0.03 - 0.02 * (hi - lo) };
                a[idx]     = re;
                a[idx + 1] = im;
            } else {
                a[idx]     = f32::NAN;
                a[idx + 1] = f32::NAN;
            }
        }
    }

    a
}

fn make_strided_cvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f32; 2],
) -> Vec<f32> {
    assert!(len_logical > 0 && inc > 0);
    let mut v = vec![0.0; 2 * ((len_logical - 1) * inc + 1)];

    let mut idx = 0;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
        idx += inc;
    }

    v
}

fn copy_logical_strided_c(
    src         : &[f32],
    inc         : usize,
    len_logical : usize,
) -> Vec<f32> {
    let mut out = vec![0.0; 2 * len_logical];

    let mut idx = 0;
    for k in 0..len_logical {
        out[2 * k]     = src[2 * idx];
        out[2 * k + 1] = src[2 * idx + 1];
        idx += inc;
    }

    out
}

fn assert_allclose_c(
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
const ATOL: f32 = 1e-6;


fn run_case(
    tri   : CoralTriangular,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : [f32; 2],
    beta  : [f32; 2],
) {
    let a  = make_hermitian_col_major(n, lda);

    let x  = make_strided_cvec(
        n, 
        incx, 
        |k| [0.2 + 0.1 * (k as f32),
        -0.3 + 0.05 * (k as f32)]
    );

    let y0 = make_strided_cvec(
        n, 
        incy, 
        |k| [-0.3 + 0.05 * (k as f32), 
        0.1 - 0.02 * (k as f32)]
    );

    // coral
    let mut y_coral = y0.clone();
    chemv(
        tri,
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
    cblas_chemv_ref(
        tri,
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

    let yc = copy_logical_strided_c(&y_coral, incy, n);
    let yr = copy_logical_strided_c(&y_ref,   incy, n);
    assert_allclose_c(&yc, &yr, RTOL, ATOL);
}

fn run_all_uplos(
    n            : usize,
    lda          : usize,
    stride_pairs : &[(usize, usize)],
) {
    let uplos = [CoralTriangular::UpperTriangular, CoralTriangular::LowerTriangular];

    // include alpha=0, beta=0, and mixed cases 
    let coeffs: [([f32; 2], [f32; 2]); 6] = [
        ([1.0,   0.0],  [0.0,   0.0]),   // pure product
        ([0.0,   0.0],  [0.7,  -0.1]),   // alpha = 0 (scale y)
        ([0.75, -0.10], [-0.25, 0.15]),
        ([-0.6,  0.2],  [0.4,  -0.1]),
        ([1.25,  0.0],  [-0.5,  0.0]),
        ([0.85, -0.1],  [0.1,   0.05]),
    ];

    for &(incx, incy) in stride_pairs {
        for &tri in &uplos {
            for &(alpha, beta) in &coeffs {
                run_case(tri, n, lda, incx, incy, alpha, beta);
            }
        }
    }
}


#[test]
fn small_all_uplos() {
    run_all_uplos(6, 6, &[(1, 1)]);
}

#[test]
fn large_all_uplos() {
    run_all_uplos(640, 640, &[(1, 1)]);
}

#[test]
fn padded_all_uplos() {
    run_all_uplos(300, 300 + 5, &[(1, 1)]);
}

#[test]
fn strided_all_uplos() {
    run_all_uplos(384, 384, &[(2, 3), (3, 2), (2, 1)]);
}

// upper and lower agree
#[test]
fn upper_equals_lower() {
    let n   = 257;
    let lda = n;

    let a   = make_hermitian_col_major(n, lda);

    let x   = make_strided_cvec(
        n, 
        1, 
        |k| [-0.07 + 0.013 * (k as f32), 
        0.015 - 0.002 * (k as f32)]
    );

    let y0  = make_strided_cvec(
        n, 
        1, 
        |k| [
            0.02 - 0.004 * (k as f32),
            -0.01 + 0.001 * (k as f32)
        ]
    );

    let alpha = [1.123, -0.321];
    let beta  = [-0.321, 0.777];

    let mut y_upper = y0.clone();
    chemv(
        CoralTriangular::UpperTriangular,
        n,
        alpha,
        &a, 
        lda, 
        &x,
        1,
        beta, 
        &mut y_upper, 
        1
    );

    let mut y_lower = y0.clone();
    chemv(
        CoralTriangular::LowerTriangular, 
        n, 
        alpha, 
        &a, 
        lda, 
        &x,
        1,
        beta,
        &mut y_lower,
        1
    );

    assert_allclose_c(&y_upper, &y_lower, RTOL, ATOL);
}

// kernels ignore non-selected triangle
#[test]
fn upper_respects_triangle() {
    let n   = 200;
    let lda = n + 3;

    let alpha = [0.7,  0.1];
    let beta  = [0.25, -0.05];

    let a  = make_upper_stored_lower_garbage_c(n, lda);

    let x  = make_strided_cvec(
        n, 
        1, 
        |k| [
            0.03 * (k as f32) - 0.5,
            0.02 - 0.01 * (k as f32)
        ]
    );

    let y0 = make_strided_cvec(
        n,
        1, 
        |k| [
            -0.1 + 0.002 * (k as f32), 
            0.05 - 0.003 * (k as f32)
        ]
    );

    let mut y_coral = y0.clone();
    chemv(
        CoralTriangular::UpperTriangular, 
        n, 
        alpha,
        &a, 
        lda,
        &x, 
        1,
        beta,
        &mut 
        y_coral,
        1
    );

    let mut y_ref = y0.clone();
    cblas_chemv_ref(
        CoralTriangular::UpperTriangular,
        n as i32, 
        alpha,
        a.as_ptr(), 
        lda as i32, 
        x.as_ptr(),
        1, beta,
        y_ref.as_mut_ptr(),
        1
    );

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

#[test]
fn lower_respects_triangle() {
    let n   = 180;
    let lda = n;

    let alpha = [-0.9, 0.2];
    let beta  = [0.6, -0.1];

    let a  = make_lower_stored_upper_garbage_c(n, lda);

    let x  = make_strided_cvec(
        n, 
        1, 
        |k| [
            0.2 - 0.001 * (k as f32),
            -0.03 + 0.004 * (k as f32)
        ]
    );

    let y0 = make_strided_cvec(
        n, 
        1, 
        |k| [
            0.15 + 0.004 * (k as f32), 
            -0.02 + 0.003 * (k as f32)
        ]
    );

    let mut y_coral = y0.clone();

    chemv(
        CoralTriangular::LowerTriangular, 
        n,
        alpha,
        &a, 
        lda, 
        &x,
        1,
        beta,
        &mut y_coral, 
        1
    );

    let mut y_ref = y0.clone();
    cblas_chemv_ref(
        CoralTriangular::LowerTriangular,
        n as i32,
        alpha,
        a.as_ptr(), 
        lda as i32, 
        x.as_ptr(), 
        1, 
        beta, 
        y_ref.as_mut_ptr(), 
        1
    );

    assert_allclose_c(&y_coral, &y_ref, RTOL, ATOL);
}

