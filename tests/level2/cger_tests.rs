use blas_src as _; 
use cblas_sys::{
    cblas_cgeru,
    cblas_cgerc,
    CBLAS_LAYOUT,
};
use coral::level2::cgeru::cgeru;
use coral::level2::cgerc::cgerc;

fn cblas_cgeru_wrapper(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    x     : *const f32,
    incx  : i32,
    y     : *const f32,
    incy  : i32,
    a     : *mut f32,
    lda   : i32,
) {
    unsafe {
        cblas_cgeru(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            x               as *const [f32; 2],
            incx,
            y               as *const [f32; 2],
            incy,
            a               as *mut [f32; 2],
            lda,
        );
    }
}

fn cblas_cgerc_wrapper(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    x     : *const f32,
    incx  : i32,
    y     : *const f32,
    incy  : i32,
    a     : *mut f32,
    lda   : i32,
) {
    unsafe {
        cblas_cgerc(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha.as_ptr()  as *const [f32; 2],
            x               as *const [f32; 2],
            incx,
            y               as *const [f32; 2],
            incy,
            a               as *mut [f32; 2],
            lda,
        );
    }
}

// helpers
fn make_cmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    let mut a = vec![0.0f32; 2 * lda * n];

    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            // deterministic but nontrivial
            a[idx]     = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;   // re
            a[idx + 1] = -0.2 + (i as f32) * 0.3 - (j as f32) * 0.15;  // im
        }
    }
    a
}

fn make_strided_cvec(
    len_logical : usize,
    inc         : usize,
    f           : impl Fn(usize) -> [f32; 2],
) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * ((len_logical - 1) * inc + 1)];

    let mut idx = 0usize;
    for k in 0..len_logical {
        let z = f(k);
        v[2 * idx]     = z[0];
        v[2 * idx + 1] = z[1];
        idx += inc;
    }
    v
}

fn assert_allclose_c(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
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

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-6;

// tests (cgeru)

#[test]
fn cgeru_contiguous_small() {
    let m   = 7usize;
    let n   = 5usize;
    let lda = m;

    let alpha = [1.25f32, -0.40f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.2  + 0.1  * (i as f32), -0.3 + 0.05 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [-0.1 + 0.07 * (j as f32),  0.2 - 0.04 * (j as f32)]);

    // coral
    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_contiguous_large_tall() {
    let m   = 1024usize;
    let n   = 768usize;
    let lda = m;

    let alpha = [-0.37f32, 0.21f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.05 + 0.002 * (i as f32), -0.02 + 0.001 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [0.4  - 0.003 * (j as f32),  0.1  + 0.002 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_contiguous_large_wide() {
    let m   = 512usize;
    let n   = 1024usize;
    let lda = m;

    let alpha = [0.93f32, -0.61f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [-0.2 + 0.0015 * (i as f32), 0.3 - 0.0007 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [ 0.1 + 0.0020 * (j as f32), 0.2 + 0.0010 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_strided_padded() {
    let m   = 9usize;
    let n   = 4usize;
    let lda = m + 3;

    let alpha = [-0.85f32, 0.0f32];

    let a0 = make_cmatrix(m, n, lda);

    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec(m, incx, |i| [0.05 + 0.03 * (i as f32), 0.04 - 0.01 * (i as f32)]);
    let y = make_strided_cvec(n, incy, |j| [0.4  - 0.02 * (j as f32), -0.3 + 0.05 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_alpha_zero_keeps_a() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.0f32, 0.0f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.11 + 0.01 * (i as f32), -0.02 + 0.003 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [0.07 - 0.01 * (j as f32),  0.05 + 0.004 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_accumulate_twice() {
    let m   = 64usize;
    let n   = 48usize;
    let lda = m;

    let alpha1 = [1.2f32, -0.3f32];
    let alpha2 = [-0.7f32, 0.15f32];

    let a0 = make_cmatrix(m, n, lda);
    let x1 = make_strided_cvec(m, 1, |i| [0.2  + 0.01 * (i as f32), -0.1 + 0.02 * (i as f32)]);
    let y1 = make_strided_cvec(n, 1, |j| [-0.1 + 0.02 * (j as f32),  0.4 - 0.01 * (j as f32)]);
    let x2 = make_strided_cvec(m, 1, |i| [-0.3 + 0.03 * (i as f32),  0.25 - 0.02 * (i as f32)]);
    let y2 = make_strided_cvec(n, 1, |j| [ 0.4 - 0.01 * (j as f32), -0.2 + 0.03 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cgeru(
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
    cblas_cgeru_wrapper(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_m_zero_quick_return() {
    let m   = 0usize;
    let n   = 5usize;
    let lda = 1usize;

    let alpha = [0.77f32, -0.18f32];

    let a0 = vec![0.0f32; 2 * lda * n];
    let x  = make_strided_cvec(1, 1, |_| [1.0, 0.0]);
    let y  = make_strided_cvec(n, 1, |j| [0.2 + 0.1 * (j as f32), -0.05 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgeru_n_zero_quick_return() {
    let m   = 6usize;
    let n   = 0usize;
    let lda = m;

    let alpha = [-0.55f32, 0.12f32];

    let a0 = make_cmatrix(m, 1.max(n), lda);
    let x  = make_strided_cvec(m, 1, |i| [0.3 - 0.02 * (i as f32), 0.15 + 0.01 * (i as f32)]);
    let y  = make_strided_cvec(1, 1, |_| [0.0, 0.0]);

    let mut a_coral = a0.clone();
    cgeru(
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
    cblas_cgeru_wrapper(
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

    assert_allclose_c(&a_coral[..(2 * lda * 1.max(n))], &a_ref[..(2 * lda * 1.max(n))], RTOL, ATOL);
}

// tests (cgerc)

#[test]
fn cgerc_contiguous_small() {
    let m   = 7usize;
    let n   = 5usize;
    let lda = m;

    let alpha = [0.85f32, 0.35f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.2  + 0.1  * (i as f32), -0.3 + 0.05 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [-0.1 + 0.07 * (j as f32),  0.2 - 0.04 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_contiguous_large_tall() {
    let m   = 1024usize;
    let n   = 768usize;
    let lda = m;

    let alpha = [-0.21f32, 0.44f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.05 + 0.002 * (i as f32), -0.02 + 0.001 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [0.4  - 0.003 * (j as f32),  0.1  + 0.002 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_contiguous_large_wide() {
    let m   = 512usize;
    let n   = 1024usize;
    let lda = m;

    let alpha = [0.37f32, -0.72f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [-0.2 + 0.0015 * (i as f32), 0.3 - 0.0007 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [ 0.1 + 0.0020 * (j as f32), 0.2 + 0.0010 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_strided_padded() {
    let m   = 9usize;
    let n   = 4usize;
    let lda = m + 3;

    let alpha = [0.65f32, 0.10f32];

    let a0 = make_cmatrix(m, n, lda);

    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec(m, incx, |i| [0.05 + 0.03 * (i as f32), 0.04 - 0.01 * (i as f32)]);
    let y = make_strided_cvec(n, incy, |j| [0.4  - 0.02 * (j as f32), -0.3 + 0.05 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_alpha_zero_keeps_a() {
    let m   = 300usize;
    let n   = 200usize;
    let lda = m + 5;

    let alpha = [0.0f32, 0.0f32];

    let a0 = make_cmatrix(m, n, lda);
    let x  = make_strided_cvec(m, 1, |i| [0.11 + 0.01 * (i as f32), -0.02 + 0.003 * (i as f32)]);
    let y  = make_strided_cvec(n, 1, |j| [0.07 - 0.01 * (j as f32),  0.05 + 0.004 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_accumulate_twice() {
    let m   = 64usize;
    let n   = 48usize;
    let lda = m;

    let alpha1 = [0.6f32,  0.4f32];
    let alpha2 = [-0.7f32, 0.2f32];

    let a0 = make_cmatrix(m, n, lda);
    let x1 = make_strided_cvec(m, 1, |i| [0.2  + 0.01 * (i as f32), -0.1 + 0.02 * (i as f32)]);
    let y1 = make_strided_cvec(n, 1, |j| [-0.1 + 0.02 * (j as f32),  0.4 - 0.01 * (j as f32)]);
    let x2 = make_strided_cvec(m, 1, |i| [-0.3 + 0.03 * (i as f32),  0.25 - 0.02 * (i as f32)]);
    let y2 = make_strided_cvec(n, 1, |j| [ 0.4 - 0.01 * (j as f32), -0.2 + 0.03 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cgerc(
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
    cblas_cgerc_wrapper(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_m_zero_quick_return() {
    let m   = 0usize;
    let n   = 5usize;
    let lda = 1usize;

    let alpha = [0.77f32, 0.18f32];

    let a0 = vec![0.0f32; 2 * lda * n];
    let x  = make_strided_cvec(1, 1, |_| [1.0, 0.0]);
    let y  = make_strided_cvec(n, 1, |j| [0.2 + 0.1 * (j as f32), -0.05 * (j as f32)]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

#[test]
fn cgerc_n_zero_quick_return() {
    let m   = 6usize;
    let n   = 0usize;
    let lda = m;

    let alpha = [-0.55f32, -0.12f32];

    let a0 = make_cmatrix(m, 1.max(n), lda);
    let x  = make_strided_cvec(m, 1, |i| [0.3 - 0.02 * (i as f32), 0.15 + 0.01 * (i as f32)]);
    let y  = make_strided_cvec(1, 1, |_| [0.0, 0.0]);

    let mut a_coral = a0.clone();
    cgerc(
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
    cblas_cgerc_wrapper(
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

    assert_allclose_c(&a_coral[..(2 * lda * 1.max(n))], &a_ref[..(2 * lda * 1.max(n))], RTOL, ATOL);
}

