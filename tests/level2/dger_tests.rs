use blas_src as _; 
use cblas_sys::{
    cblas_dger,
    CBLAS_LAYOUT,
};
use coral::level2::dger::dger;

fn cblas_dger_wrapper(
    m     : i32,
    n     : i32,
    alpha : f64,
    x     : *const f64,
    incx  : i32,
    y     : *const f64,
    incy  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_dger(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha,
            x,
            incx,
            y,
            incy,
            a,
            lda,
        );
    }
}

// helpers
fn make_matrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; lda * n];
    for j in 0..n {
        for i in 0..m {
            // deterministic pattern
            a[i + j * lda] = 0.1f64 + (i as f64) * 0.5f64 + (j as f64) * 0.25f64;
        }
    }
    a
}

fn make_strided_vec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v = vec![0.0f64; (len_logical - 1) * inc + 1];
    let mut idx = 0usize;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn assert_allclose(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})"
        );
    }
}

const RTOL: f64 = 1e-15;
const ATOL: f64 = 1e-15;

// tests

#[test]
fn contiguous_small() {
    let m   = 7usize;
    let n   = 5usize;
    let lda = m;

    let alpha = 1.25f64;

    let a0 = make_matrix(m, n, lda);
    let x  = (0..m).map(|i| 0.2f64  + 0.1f64  * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|j| -0.3f64 + 0.05f64 * (j as f64)).collect::<Vec<_>>();

    // coral
    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    // cblas
    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn contiguous_large_tall() {
    let m   = 1024usize;
    let n   = 768usize;
    let lda = m;

    let alpha = -0.37f64;

    let a0 = make_matrix(m, n, lda);
    let x  = (0..m).map(|i| 0.05f64 + 0.002f64 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|j| 0.4f64  - 0.003f64 * (j as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn contiguous_large_wide() {
    let m   = 512usize;
    let n   = 1024usize;
    let lda = m;

    let alpha = 0.93f64;

    let a0 = make_matrix(m, n, lda);
    let x  = (0..m).map(|i| -0.2f64 + 0.0015f64 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|j|  0.1f64 + 0.0020f64 * (j as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn strided_padded() {
    let m   = 9usize;
    let n   = 4usize;
    let lda = m + 3; // padded lda

    let alpha = -0.85f64;

    let a0 = make_matrix(m, n, lda);

    // non-unit strides
    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_vec(m, incx, |i| 0.05f64 + 0.03f64 * (i as f64));
    let y = make_strided_vec(n, incy, |j| 0.4f64  - 0.02f64 * (j as f64));

    // coral
    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        incx,
        &y,
        incy,
        &mut a_coral,
        lda,
    );

    // cblas
    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        incx as i32,
        y.as_ptr(),
        incy as i32,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn alpha_zero_keeps_a() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = 0.0f64;

    let a0 = make_matrix(m, n, lda);
    let x  = (0..m).map(|i| 0.11f64 + 0.01f64 * (i as f64)).collect::<Vec<_>>();
    let y  = (0..n).map(|j| 0.07f64 - 0.01f64 * (j as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn accumulate_twice() {
    let m   = 64usize;
    let n   = 48usize;
    let lda = m;

    let alpha1 = 1.2f64;
    let alpha2 = -0.7f64;

    let a0 = make_matrix(m, n, lda);
    let x1 = (0..m).map(|i| 0.2f64  + 0.01f64 * (i as f64)).collect::<Vec<_>>();
    let y1 = (0..n).map(|j| -0.1f64 + 0.02f64 * (j as f64)).collect::<Vec<_>>();
    let x2 = (0..m).map(|i| -0.3f64 + 0.03f64 * (i as f64)).collect::<Vec<_>>();
    let y2 = (0..n).map(|j|  0.4f64 - 0.01f64 * (j as f64)).collect::<Vec<_>>();

    // coral: two updates
    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha1,
        &x1,
        1,
        &y1,
        1,
        &mut a_coral,
        lda,
    );
    dger(
        m,
        n,
        alpha2,
        &x2,
        1,
        &y2,
        1,
        &mut a_coral,
        lda,
    );

    // cblas: two updates
    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha1,
        x1.as_ptr(),
        1,
        y1.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha2,
        x2.as_ptr(),
        1,
        y2.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn m_zero_quick_return() {
    let m   = 0usize;
    let n   = 5usize;
    let lda = 1usize; // arbitrary for m = 0

    let alpha = 0.77f64;

    let a0 = vec![0.0f64; lda * n];
    let x  = vec![1.0f64; 1];
    let y  = (0..n).map(|j| 0.2f64 + 0.1f64 * (j as f64)).collect::<Vec<_>>();

    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn n_zero_quick_return() {
    let m   = 6usize;
    let n   = 0usize;
    let lda = m;

    let alpha = -0.55f64;

    let a0 = make_matrix(m, 1.max(n), lda); // allocate non-empty
    let x  = (0..m).map(|i| 0.3f64 - 0.02f64 * (i as f64)).collect::<Vec<_>>();
    let y  = vec![0.0f64; 1]; // dummy

    let mut a_coral = a0.clone();
    dger(
        m,
        n,
        alpha,
        &x,
        1,
        &y,
        1,
        &mut a_coral,
        lda,
    );

    let mut a_ref = a0.clone();
    cblas_dger_wrapper(
        m as i32,
        n as i32,
        alpha,
        x.as_ptr(),
        1,
        y.as_ptr(),
        1,
        a_ref.as_mut_ptr(),
        lda as i32,
    );

    assert_allclose(&a_coral[..(lda * 1.max(n))], &a_ref[..(lda * 1.max(n))], RTOL, ATOL);
}

