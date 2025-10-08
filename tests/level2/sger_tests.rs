use blas_src as _;
use cblas_sys::{ cblas_sger, CBLAS_LAYOUT };
use coral::level2::sger::sger;

#[inline(always)]
fn cblas_sger_ref(
    m     : i32,
    n     : i32,
    alpha : f32,
    x     : *const f32,
    incx  : i32,
    y     : *const f32,
    incy  : i32,
    a     : *mut f32,
    lda   : i32,
) {
    unsafe {
        cblas_sger(
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

fn make_matrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= m);
    let mut a: Vec<f32> = vec![0.0; lda * n];

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
    let mut v: Vec<f32> = vec![0.0; (len_logical - 1) * inc + 1];

    let mut idx = 0;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }

    v
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
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
    }
}

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-6;

fn run_case(
    m     : usize,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : f32,
    xgen  : impl Fn(usize) -> f32,
    ygen  : impl Fn(usize) -> f32,
) {
    let a0 = make_matrix(m, n, lda);
    let x  = make_strided_vec(m, incx, xgen);
    let y  = make_strided_vec(n, incy, ygen);

    let mut a_coral = a0.clone();
    sger(
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
    cblas_sger_ref(
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

fn run_contiguous_sets() {
    run_case(
        7,
        5,
        7,
        1,
        1,
        1.25,
        |i| 0.2 + 0.1 * (i as f32),
        |j| -0.1 + 0.07 * (j as f32),
    );

    run_case(
        1024,
        768,
        1024,
        1,
        1,
        -0.37,
        |i| 0.05 + 0.002 * (i as f32),
        |j| 0.4  - 0.003 * (j as f32),
    );

    run_case(
        512,
        1024,
        512,
        1,
        1,
        0.93,
        |i| -0.2 + 0.0015 * (i as f32),
        |j|  0.1 + 0.0020 * (j as f32),
    );
}

fn run_strided_padded() {
    let m    = 9;
    let n    = 4;
    let lda  = m + 3;
    let incx = 2;
    let incy = 3;

    run_case(
        m,
        n,
        lda,
        incx,
        incy,
        -0.85,
        |i| 0.05 + 0.03 * (i as f32),
        |j| 0.4  - 0.02 * (j as f32),
    );
}

fn run_alpha_zero(
    m   : usize,
    n   : usize,
    lda : usize,
) {
    run_case(
        m,
        n,
        lda,
        1,
        1,
        0.0,
        |i| 0.11 + 0.01 * (i as f32),
        |j| 0.07 - 0.01 * (j as f32),
    );
}

fn run_accumulate_twice() {
    let m   = 64;
    let n   = 48;
    let lda = m;

    let a0 = make_matrix(m, n, lda);

    let alpha1 = 1.2;
    let alpha2 = -0.7;

    let x1 = make_strided_vec(
        m,
        1,
        |i| 0.2 + 0.01 * (i as f32),
    );
    let y1 = make_strided_vec(
        n,
        1,
        |j| -0.1 + 0.02 * (j as f32),
    );
    let x2 = make_strided_vec(
        m,
        1,
        |i| -0.3 + 0.03 * (i as f32),
    );
    let y2 = make_strided_vec(
        n,
        1,
        |j| 0.4 - 0.01 * (j as f32),
    );

    let mut a_coral = a0.clone();
    sger(
        m, n, alpha1, &x1, 1, &y1, 1, &mut a_coral, lda,
    );
    sger(
        m, n, alpha2, &x2, 1, &y2, 1, &mut a_coral, lda,
    );

    let mut a_ref = a0.clone();
    cblas_sger_ref(
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
    cblas_sger_ref(
       
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

fn run_quick_returns() {
    {
        let m   = 0;
        let n   = 5;
        let lda = 1;

        let a0: Vec<f32> = vec![0.0; lda * n];
        let x            = make_strided_vec(1, 1, |_| 1.0);
        let y            = make_strided_vec(n, 1, |j| 0.2 + 0.1 * (j as f32));

        let mut a_coral = a0.clone();
        sger(
            m,
            n,
            0.77,
            &x,
            1,
            &y,
            1,
            &mut a_coral,
            lda,
        );

        let mut a_ref = a0.clone();
        cblas_sger_ref(
            m as i32,
            n as i32,
            0.77,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a_ref.as_mut_ptr(),
            lda as i32,
        );

        assert_allclose(&a_coral, &a_ref, RTOL, ATOL);
    }

    {
        let m   = 6;
        let n   = 0;
        let lda = m;

        let a0 = make_matrix(m, 1.max(n), lda);
        let x  = make_strided_vec(m, 1, |i| 0.3 - 0.02 * (i as f32));
        let y  = make_strided_vec(1, 1, |_| 0.0);

        let mut a_coral = a0.clone();
        sger(
            m,
            n,
            -0.55,
            &x,
            1,
            &y,
            1,
            &mut a_coral,
            lda,
        );

        let mut a_ref = a0.clone();
        cblas_sger_ref(
            m as i32,
            n as i32,
            -0.55,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a_ref.as_mut_ptr(),
            lda as i32,
        );

        let used = lda * 1.max(n);
        assert_allclose(&a_coral[..used], &a_ref[..used], RTOL, ATOL);
    }
}

#[test]
fn sger_suites() {
    run_contiguous_sets();
    run_strided_padded();
    run_alpha_zero(300, 200, 300 + 5);
    run_accumulate_twice();
    run_quick_returns();
}

