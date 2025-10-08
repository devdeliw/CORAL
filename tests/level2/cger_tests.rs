use blas_src as _;
use cblas_sys::{ cblas_cgeru, cblas_cgerc, CBLAS_LAYOUT };
use coral::level2::{ cgeru::cgeru, cgerc::cgerc };

#[inline(always)]
fn cblas_cgeru_ref(
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
            a               as *mut   [f32; 2],
            lda,
        );
    }
}

#[inline(always)]
fn cblas_cgerc_ref(
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
            a               as *mut   [f32; 2],
            lda,
        );
    }
}

fn make_cmatrix(
    m   : usize,
    n   : usize,
    lda : usize,
) -> Vec<f32> {
    assert!(lda >= m);
    let mut a: Vec<f32> = vec![0.0; 2 * lda * n];
    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * lda);
            a[idx]     = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;  // re
            a[idx + 1] = -0.2 + (i as f32) * 0.3 - (j as f32) * 0.15; // im
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
    let mut v: Vec<f32> = vec![0.0; 2 * ((len_logical - 1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len_logical {
        let [re, im] = f(k);
        v[2 * idx]     = re;
        v[2 * idx + 1] = im;
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

        assert!(dr <= tr, "re mismatch at pair {}: {xr} vs {yr} (delta={dr}, tol={tr})", i / 2);
        assert!(di <= ti, "im mismatch at pair {}: {xi} vs {yi} (delta={di}, tol={ti})", i / 2);
    }
}

const RTOL: f32 = 1e-6;
const ATOL: f32 = 1e-6;

#[derive(Copy, Clone)]
enum GerKind { Unconj, Conj }

type CoralGerFn = fn(
    m     : usize,
    n     : usize,
    alpha : [f32; 2],
    x     : &[f32],
    incx  : usize,
    y     : &[f32],
    incy  : usize,
    a     : &mut [f32],
    lda   : usize,
);

type CblasGerFn = fn(
    m     : i32,
    n     : i32,
    alpha : [f32; 2],
    x     : *const f32,
    incx  : i32,
    y     : *const f32,
    incy  : i32,
    a     : *mut f32,
    lda   : i32,
);

fn coral_fn_for(
    kind : GerKind,
) -> CoralGerFn {
    match kind {
        GerKind::Unconj => |m, n, alpha, x, incx, y, incy, a, lda| {
            cgeru(
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                a,
                lda,
            )
        },
        GerKind::Conj => |m, n, alpha, x, incx, y, incy, a, lda| {
            cgerc(
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                a,
                lda,
            )
        },
    }
}

fn cblas_fn_for(
    kind : GerKind,
) -> CblasGerFn {
    match kind {
        GerKind::Unconj => cblas_cgeru_ref,
        GerKind::Conj   => cblas_cgerc_ref,
    }
}

fn run_case(
    kind  : GerKind,
    m     : usize,
    n     : usize,
    lda   : usize,
    incx  : usize,
    incy  : usize,
    alpha : [f32; 2],
    xgen  : impl Fn(usize) -> [f32; 2],
    ygen  : impl Fn(usize) -> [f32; 2],
) {
    let coral = coral_fn_for(kind);
    let cblas = cblas_fn_for(kind);

    let a0  = make_cmatrix(m, n, lda);
    let x   = make_strided_cvec(m, incx, xgen);
    let y   = make_strided_cvec(n, incy, ygen);

    let mut a_coral = a0.clone();
    coral(
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
    cblas(
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

fn run_alpha_zero(
    kind : GerKind,
    m    : usize,
    n    : usize,
    lda  : usize,
) {
    run_case(
        kind,
        m,
        n,
        lda,
        1,
        1,
        [0.0, 0.0],
        |i| [0.11 + 0.01 * (i as f32), -0.02 + 0.003 * (i as f32)],
        |j| [0.07 - 0.01 * (j as f32),  0.05 + 0.004 * (j as f32)],
    );
}

fn run_strided_padded(
    kind : GerKind,
) {
    let m    = 9;
    let n    = 4;
    let lda  = m + 3;
    let incx = 2;
    let incy = 3;

    let alpha = match kind {
        GerKind::Unconj => [-0.85,  0.00],
        GerKind::Conj   => [ 0.65,  0.10],
    };

    run_case(
        kind,
        m,
        n,
        lda,
        incx,
        incy,
        alpha,
        |i| [0.05 + 0.03 * (i as f32),  0.04 - 0.01 * (i as f32)],
        |j| [0.40 - 0.02 * (j as f32), -0.30 + 0.05 * (j as f32)],
    );
}

fn run_contiguous_sets(
    kind : GerKind,
) {
    // small
    run_case(
        kind,
        7,
        5,
        7,
        1,
        1,
        match kind { GerKind::Unconj => [ 1.25, -0.40], GerKind::Conj => [ 0.85,  0.35] },
        |i| [0.20 + 0.10 * (i as f32), -0.30 + 0.05 * (i as f32)],
        |j| [-0.10 + 0.07 * (j as f32), 0.20 - 0.04 * (j as f32)],
    );

    // large tall
    run_case(
        kind,
        1024,
        768,
        1024,
        1,
        1,
        match kind { 
            GerKind::Unconj => [-0.37,  0.21], 
            GerKind::Conj   => [-0.21,  0.44] 
        },
        |i| [0.05 + 0.002 * (i as f32), -0.02 + 0.001 * (i as f32)],
        |j| [0.40 - 0.003 * (j as f32),  0.10 + 0.002 * (j as f32)],
    );

    // large wide
    run_case(
        kind,
        512,
        1024,
        512,
        1,
        1,
        match kind { 
            GerKind::Unconj => [ 0.93, -0.61], 
            GerKind::Conj   => [ 0.37, -0.72] 
        },
        |i| [-0.20 + 0.0015 * (i as f32), 0.30 - 0.0007 * (i as f32)],
        |j| [ 0.10 + 0.0020 * (j as f32), 0.20 + 0.0010 * (j as f32)],
    );
}

fn run_accumulate_twice(
    kind : GerKind,
) {
    let m   = 64;
    let n   = 48;
    let lda = m;

    let a0  = make_cmatrix(m, n, lda);

    let (alpha1, alpha2) = match kind {
        GerKind::Unconj => ([ 1.20, -0.30], [-0.70, 0.15]),
        GerKind::Conj   => ([ 0.60,  0.40], [-0.70, 0.20]),
    };

    let x1 = make_strided_cvec(
        m,
        1,
        |i| [0.20 + 0.01 * (i as f32),
        -0.10 + 0.02 * (i as f32)],
    );
    let y1 = make_strided_cvec(
        n,
        1,
        |j| [-0.10 + 0.02 * (j as f32),
        0.40 - 0.01 * (j as f32)],
    );
    let x2 = make_strided_cvec(
        m,
        1,
        |i| [-0.30 + 0.03 * (i as f32), 
        0.25 - 0.02 * (i as f32)],
    );
    let y2 = make_strided_cvec(
        n,
        1,
        |j| [0.40 - 0.01 * (j as f32),
        -0.20 + 0.03 * (j as f32)],
    );

    let coral = coral_fn_for(kind);
    let cblas = cblas_fn_for(kind);

    let mut a_coral = a0.clone();

    coral(
        m, n, alpha1, &x1, 1, &y1, 1, &mut a_coral, lda,
    );
    coral(
        m, n, alpha2, &x2, 1, &y2, 1, &mut a_coral, lda,
    );

    let mut a_ref = a0.clone();
    cblas(
        m as i32, n as i32, alpha1, x1.as_ptr(), 1, y1.as_ptr(), 1, a_ref.as_mut_ptr(), lda as i32,
    );
    cblas(
        m as i32, n as i32, alpha2, x2.as_ptr(), 1, y2.as_ptr(), 1, a_ref.as_mut_ptr(), lda as i32,
    );

    assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
}

fn run_quick_returns(
    kind : GerKind,
) {
    // m == 0
    {
        let m   = 0;
        let n   = 5;
        let lda = 1;

        let a0 : Vec<f32> = vec![0.0; 2 * lda * n];
        let x  = make_strided_cvec(1, 1, |_| [1.0, 0.0]);
        let y  = make_strided_cvec(
            n,
            1,
            |j| [0.20 + 0.10 * (j as f32), -0.05 * (j as f32)],
        );

        let coral = coral_fn_for(kind);
        let cblas = cblas_fn_for(kind);

        let mut a_coral = a0.clone();
        coral(
            m,
            n,
            match kind {
                GerKind::Unconj => [ 0.77, -0.18],
                GerKind::Conj   => [ 0.77,  0.18]
            },
            &x,
            1,
            &y,
            1,
            &mut a_coral,
            lda,
        );

        let mut a_ref = a0.clone();
        cblas(
            m as i32,
            n as i32,
            match kind { 
                GerKind::Unconj => [ 0.77, -0.18], 
                GerKind::Conj   => [ 0.77,  0.18] 
            },
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a_ref.as_mut_ptr(),
            lda as i32,
        );

        assert_allclose_c(&a_coral, &a_ref, RTOL, ATOL);
    }

    // n == 0
    {
        let m   = 6;
        let n   = 0;
        let lda = m;

        let a0 = make_cmatrix(m, 1.max(n), lda);
        let x  = make_strided_cvec(
            m,
            1,
            |i| [0.30 - 0.02 * (i as f32),
            0.15 + 0.01 * (i as f32)],
        );
        let y  = make_strided_cvec(1, 1, |_| [0.0, 0.0]);

        let coral = coral_fn_for(kind);
        let cblas = cblas_fn_for(kind);

        let mut a_coral = a0.clone();
        coral(
            m,
            n,
            match kind {
                GerKind::Unconj => [-0.55,  0.12],
                GerKind::Conj   => [-0.55, -0.12] 
            },
            &x,
            1,
            &y,
            1,
            &mut a_coral,
            lda,
        );

        let mut a_ref = a0.clone();
        cblas(
            m as i32,
            n as i32,
            match kind {
                GerKind::Unconj => [-0.55,  0.12],
                GerKind::Conj   => [-0.55, -0.12] 
            },
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a_ref.as_mut_ptr(),
            lda as i32,
        );

        let used = 2 * lda * 1.max(n);
        assert_allclose_c(&a_coral[..used], &a_ref[..used], RTOL, ATOL);
    }
}


#[test]
fn geru_suites() {
    run_contiguous_sets(GerKind::Unconj);
    run_strided_padded(GerKind::Unconj);
    run_alpha_zero(GerKind::Unconj, 300, 200, 300 + 5);
    run_accumulate_twice(GerKind::Unconj);
    run_quick_returns(GerKind::Unconj);
}

#[test]
fn gerc_suites() {
    run_contiguous_sets(GerKind::Conj);
    run_strided_padded(GerKind::Conj);
    run_alpha_zero(GerKind::Conj, 300, 200, 300 + 5);
    run_accumulate_twice(GerKind::Conj);
    run_quick_returns(GerKind::Conj);
}

