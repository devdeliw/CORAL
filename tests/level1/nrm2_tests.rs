use rusty_blas::level1::{
    snrm2::snrm2,
    dnrm2::dnrm2,
    scnrm2::scnrm2,
    dznrm2::dznrm2,
};
use cblas_sys::{
    cblas_snrm2,
    cblas_dnrm2,
    cblas_scnrm2,
    cblas_dznrm2,
};


#[inline]
fn approx_eq_f32(a: f32, b: f32, rel: f32, abs: f32) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "nrm2 f32 mismatch: a={a}, b={b}, diff={diff}"
    );
}

#[inline]
fn approx_eq_f64(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "nrm2 f64 mismatch: a={a}, b={b}, diff={diff}"
    );
}


#[test]
fn snrm2_n0_is_zero() {
    let x: [f32; 0] = [];

    let rusty = snrm2(0, &x, 1);

    assert_eq!(rusty, 0.0);
}


#[test]
fn snrm2_matches_cblas() {
    let n = 2048usize;

    let x32: Vec<f32> = (0..n)
        .map(|i| ((i as f32) - 777.0) * 1e-3)
        .collect();

    let rusty = snrm2(n, &x32, 1);
    let cblas = unsafe {
        cblas_snrm2(
            n as i32,
            x32.as_ptr(),
            1,
        )
    };

    approx_eq_f32(rusty, cblas, 1e-6, 1e-6);
}

#[test]
fn dnrm2_matches_cblas() {
    let n = 2048usize;

    let x64: Vec<f64> = (0..n)
        .map(|i| ((i as f64) - 777.0) * 1e-6)
        .collect();

    let rusty = dnrm2(n, &x64, 1);
    let cblas = unsafe {
        cblas_dnrm2(
            n as i32,
            x64.as_ptr(),
            1,
        )
    };

    approx_eq_f64(rusty, cblas, 1e-12, 1e-12);
}

#[test]
fn snrm2_matches_cblas_stride() {
    let n      = 333usize;
    let stride = 3isize;

    let mut x32: Vec<f32> = (0..n * 3)
        .map(|_| 0.0)
        .collect();
    for i in 0..n {
        x32[i * 3] = (i as f32 * 0.05).sin() * 1e-2;
    }

    let rusty = snrm2(n, &x32, stride);
    let cblas = unsafe {
        cblas_snrm2(
            n as i32,
            x32.as_ptr(),
            stride as i32,
        )
    };

    approx_eq_f32(rusty, cblas, 1e-6, 1e-6);
}

#[test]
fn dnrm2_matches_cblas_stride() {
    let n      = 333usize;
    let stride = 3isize;

    let mut x64: Vec<f64> = (0..n * 3)
        .map(|_| 0.0)
        .collect();
    for i in 0..n {
        x64[i * 3] = (i as f64 * 0.05).cos() * 1e-6;
    }

    let rusty = dnrm2(n, &x64, stride);
    let cblas = unsafe {
        cblas_dnrm2(
            n as i32,
            x64.as_ptr(),
            stride as i32,
        )
    };

    approx_eq_f64(rusty, cblas, 1e-12, 1e-12);
}

#[test]
fn scnrm2_matches_cblas() {
    let n = 1024usize;

    let mut z32 = vec![0.0f32; 2 * n];
    for i in 0..n {
        z32[2 * i]     = (i as f32 * 0.2).sin();
        z32[2 * i + 1] = (i as f32 * 0.3).cos() * 0.7;
    }

    let rusty = scnrm2(n, &z32, 1);
    let cblas = unsafe {
        cblas_scnrm2(
            n as i32,
            z32.as_ptr().cast::<[f32; 2]>(),
            1,
        )
    };

    approx_eq_f32(rusty, cblas, 1e-5, 1e-6);
}

#[test]
fn dznrm2_matches_cblas() {
    let n = 1024usize;

    let mut z64 = vec![0.0f64; 2 * n];
    for i in 0..n {
        z64[2 * i]     = (i as f64 * 0.2).sin();
        z64[2 * i + 1] = (i as f64 * 0.3).cos() * 0.7;
    }

    let rusty = dznrm2(n, &z64, 1);
    let cblas = unsafe {
        cblas_dznrm2(
            n as i32,
            z64.as_ptr().cast::<[f64; 2]>(),
            1,
        )
    };

    approx_eq_f64(rusty, cblas, 1e-12, 1e-12);
}

#[test]
fn scnrm2_matches_cblas_stride() {
    let n        = 600usize;
    let inc_c32  = 2isize;
    let len_c32  = 1 + (n - 1) * (inc_c32 as usize);

    let mut z32 = vec![0.0f32; 2 * len_c32];
    for i in 0..n {
        let k = i * (inc_c32 as usize);
        z32[2 * k]     = (i as f32 * 0.11).sin();
        z32[2 * k + 1] = (i as f32 * 0.13).cos() * 0.5;
    }

    let rusty = scnrm2(n, &z32, inc_c32);
    let cblas = unsafe {
        cblas_scnrm2(
            n as i32,
            z32.as_ptr().cast::<[f32; 2]>(),
            inc_c32 as i32,
        )
    };

    approx_eq_f32(rusty, cblas, 1e-5, 1e-6);
}

#[test]
fn dznrm2_matches_cblas_stride() {
    let n        = 600usize;
    let inc_c64  = 3isize;
    let len_c64  = 1 + (n - 1) * (inc_c64 as usize);

    let mut z64 = vec![0.0f64; 2 * len_c64];
    for i in 0..n {
        let k = i * (inc_c64 as usize);
        z64[2 * k]     = (i as f64 * 0.07).sin();
        z64[2 * k + 1] = (i as f64 * 0.09).cos() * 0.25;
    }

    let rusty = dznrm2(n, &z64, inc_c64);
    let cblas = unsafe {
        cblas_dznrm2(
            n as i32,
            z64.as_ptr().cast::<[f64; 2]>(),
            inc_c64 as i32,
        )
    };

    approx_eq_f64(rusty, cblas, 1e-12, 1e-12);
}
