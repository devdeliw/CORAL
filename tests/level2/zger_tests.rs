use blas_src as _; 
use cblas_sys::{
    cblas_zgeru,
    cblas_zgerc,
    CBLAS_LAYOUT,
};
use coral::level2::zgeru::zgeru;
use coral::level2::zgerc::zgerc;

fn cblas_zgeru_wrapper(
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    x     : *const f64,
    incx  : i32,
    y     : *const f64,
    incy  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_zgeru(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            x               as *const [f64; 2],
            incx,
            y               as *const [f64; 2],
            incy,
            a               as *mut [f64; 2],
            lda,
        );
    }
}

fn cblas_zgerc_wrapper(
    m     : i32,
    n     : i32,
    alpha : [f64; 2],
    x     : *const f64,
    incx  : i32,
    y     : *const f64,
    incy  : i32,
    a     : *mut f64,
    lda   : i32,
) {
    unsafe {
        cblas_zgerc(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha.as_ptr()  as *const [f64; 2],
            x               as *const [f64; 2],
            incx,
            y               as *const [f64; 2],
            incy,
            a               as *mut [f64; 2],
            lda,
        );
    }
}

// helpers
fn make_col_major_zmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f64> {
    let mut a = vec![0.0f64; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            // deterministic but nontrivial
            a[idx]     = 0.1 + (i as f64) * 0.5 + (j as f64) * 0.25;   // re
            a[idx + 1] = -0.2 + (i as f64) * 0.3 - (j as f64) * 0.15;  // im
        }
    }
    a
}

fn make_strided_zvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f64; 2],
) -> Vec<f64> {
    let mut v = vec![0.0f64; 2 * ((len_logical - 1) * inc + 1)];

    let mut idx = 0usize;
    for k in 0..len_logical {
        let z = f(k);
        v[2 * idx]     = z[0];
        v[2 * idx + 1] = z[1];
        idx += inc;
    }
    v
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
        let xr = a[i];     let xi = a[i + 1];
        let yr = b[i];     let yi = b[i + 1];

        let dr = (xr - yr).abs();
        let di = (xi - yi).abs();

        let tr = atol + rtol * xr.abs().max(yr.abs());
        let ti = atol + rtol * xi.abs().max(yi.abs());

        assert!(dr <= tr, "re mismatch at pair {}: {xr} vs {yr} (|Δ|={dr}, tol={tr})", i / 2);
        assert!(di <= ti, "im mismatch at pair {}: {xi} vs {yi} (|Δ|={di}, tol={ti})", i / 2);
    }
}

const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-11;

// tests (zgeru)

#[test]
fn zgeru_contiguous_small() {
    let m   = 7usize;
    let n   = 5usize;
    let lda = m;

    let alpha = [1.25f64, -0.40f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.2  + 0.1  * (i as f64), -0.3 + 0.05 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [-0.1 + 0.07 * (j as f64),  0.2 - 0.04 * (j as f64)]);

    // coral
    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_contiguous_large_tall() {
    let m   = 1024usize;
    let n   = 768usize;
    let lda = m;

    let alpha = [-0.37f64, 0.21f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.05 + 0.002 * (i as f64), -0.02 + 0.001 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [0.4  - 0.003 * (j as f64),  0.1  + 0.002 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_contiguous_large_wide() {
    let m   = 512usize;
    let n   = 1024usize;
    let lda = m;

    let alpha = [0.93f64, -0.61f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [-0.2 + 0.0015 * (i as f64), 0.3 - 0.0007 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [ 0.1 + 0.0020 * (j as f64), 0.2 + 0.0010 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_strided_padded() {
    let m   = 9usize;
    let n   = 4usize;
    let lda = m + 3;

    let alpha = [-0.85f64, 0.0f64];

    let a0 = make_col_major_zmatrix(m, n, lda);

    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_zvec(m, incx, |i| [0.05 + 0.03 * (i as f64), 0.04 - 0.01 * (i as f64)]);
    let y = make_strided_zvec(n, incy, |j| [0.4  - 0.02 * (j as f64), -0.3 + 0.05 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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

    let mut a_ref = a0.clone();
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_alpha_zero_keeps_a() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.0f64, 0.0f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.11 + 0.01 * (i as f64), -0.02 + 0.003 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [0.07 - 0.01 * (j as f64),  0.05 + 0.004 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_accumulate_twice() {
    let m   = 64usize;
    let n   = 48usize;
    let lda = m;

    let alpha1 = [1.2f64, -0.3f64];
    let alpha2 = [-0.7f64, 0.15f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x1 = make_strided_zvec(m, 1, |i| [0.2  + 0.01 * (i as f64), -0.1 + 0.02 * (i as f64)]);
    let y1 = make_strided_zvec(n, 1, |j| [-0.1 + 0.02 * (j as f64),  0.4 - 0.01 * (j as f64)]);
    let x2 = make_strided_zvec(m, 1, |i| [-0.3 + 0.03 * (i as f64),  0.25 - 0.02 * (i as f64)]);
    let y2 = make_strided_zvec(n, 1, |j| [ 0.4 - 0.01 * (j as f64), -0.2 + 0.03 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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
    zgeru(
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

    let mut a_ref = a0.clone();
    cblas_zgeru_wrapper(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_m_zero_quick_return() {
    let m   = 0usize;
    let n   = 5usize;
    let lda = 1usize;

    let alpha = [0.77f64, -0.18f64];

    let a0 = vec![0.0f64; 2 * lda * n];
    let x  = make_strided_zvec(1, 1, |_| [1.0, 0.0]);
    let y  = make_strided_zvec(n, 1, |j| [0.2 + 0.1 * (j as f64), -0.05 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgeru_n_zero_quick_return() {
    let m   = 6usize;
    let n   = 0usize;
    let lda = m;

    let alpha = [-0.55f64, 0.12f64];

    let a0 = make_col_major_zmatrix(m, 1.max(n), lda);
    let x  = make_strided_zvec(m, 1, |i| [0.3 - 0.02 * (i as f64), 0.15 + 0.01 * (i as f64)]);
    let y  = make_strided_zvec(1, 1, |_| [0.0, 0.0]);

    let mut a_coral = a0.clone();
    zgeru(
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
    cblas_zgeru_wrapper(
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

    assert_allclose_z(&a_coral[..(2 * lda * 1.max(n))], &a_ref[..(2 * lda * 1.max(n))], RTOL, ATOL);
}

// tests (zgerc)

#[test]
fn zgerc_contiguous_small() {
    let m   = 7usize;
    let n   = 5usize;
    let lda = m;

    let alpha = [0.85f64, 0.35f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.2  + 0.1  * (i as f64), -0.3 + 0.05 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [-0.1 + 0.07 * (j as f64),  0.2 - 0.04 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_contiguous_large_tall() {
    let m   = 1024usize;
    let n   = 768usize;
    let lda = m;

    let alpha = [-0.21f64, 0.44f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.05 + 0.002 * (i as f64), -0.02 + 0.001 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [0.4  - 0.003 * (j as f64),  0.1  + 0.002 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_contiguous_large_wide() {
    let m   = 512usize;
    let n   = 1024usize;
    let lda = m;

    let alpha = [0.37f64, -0.72f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [-0.2 + 0.0015 * (i as f64), 0.3 - 0.0007 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [ 0.1 + 0.0020 * (j as f64), 0.2 + 0.0010 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_strided_padded() {
    let m   = 9usize;
    let n   = 4usize;
    let lda = m + 3;

    let alpha = [0.65f64, 0.10f64];

    let a0 = make_col_major_zmatrix(m, n, lda);

    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_zvec(m, incx, |i| [0.05 + 0.03 * (i as f64), 0.04 - 0.01 * (i as f64)]);
    let y = make_strided_zvec(n, incy, |j| [0.4  - 0.02 * (j as f64), -0.3 + 0.05 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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

    let mut a_ref = a0.clone();
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_alpha_zero_keeps_a() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.0f64, 0.0f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x  = make_strided_zvec(m, 1, |i| [0.11 + 0.01 * (i as f64), -0.02 + 0.003 * (i as f64)]);
    let y  = make_strided_zvec(n, 1, |j| [0.07 - 0.01 * (j as f64),  0.05 + 0.004 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_accumulate_twice() {
    let m   = 64usize;
    let n   = 48usize;
    let lda = m;

    let alpha1 = [0.6f64,  0.4f64];
    let alpha2 = [-0.7f64, 0.2f64];

    let a0 = make_col_major_zmatrix(m, n, lda);
    let x1 = make_strided_zvec(m, 1, |i| [0.2  + 0.01 * (i as f64), -0.1 + 0.02 * (i as f64)]);
    let y1 = make_strided_zvec(n, 1, |j| [-0.1 + 0.02 * (j as f64),  0.4 - 0.01 * (j as f64)]);
    let x2 = make_strided_zvec(m, 1, |i| [-0.3 + 0.03 * (i as f64),  0.25 - 0.02 * (i as f64)]);
    let y2 = make_strided_zvec(n, 1, |j| [ 0.4 - 0.01 * (j as f64), -0.2 + 0.03 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    zgerc(
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

    let mut a_ref = a0.clone();
    cblas_zgerc_wrapper(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_m_zero_quick_return() {
    let m   = 0usize;
    let n   = 5usize;
    let lda = 1usize;

    let alpha = [0.77f64, 0.18f64];

    let a0 = vec![0.0f64; 2 * lda * n];
    let x  = make_strided_zvec(1, 1, |_| [1.0, 0.0]);
    let y  = make_strided_zvec(n, 1, |j| [0.2 + 0.1 * (j as f64), -0.05 * (j as f64)]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn zgerc_n_zero_quick_return() {
    let m   = 6usize;
    let n   = 0usize;
    let lda = m;

    let alpha = [-0.55f64, -0.12f64];

    let a0 = make_col_major_zmatrix(m, 1.max(n), lda);
    let x  = make_strided_zvec(m, 1, |i| [0.3 - 0.02 * (i as f64), 0.15 + 0.01 * (i as f64)]);
    let y  = make_strided_zvec(1, 1, |_| [0.0, 0.0]);

    let mut a_coral = a0.clone();
    zgerc(
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
    cblas_zgerc_wrapper(
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

    assert_allclose_z(&a_coral[..(2 * lda * 1.max(n))], &a_ref[..(2 * lda * 1.max(n))], RTOL, ATOL);
}

