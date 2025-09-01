use rusty_blas::level1::{
    sasum::sasum,
    dasum::dasum,
    scasum::scasum,
    dzasum::dzasum,
};
use cblas_sys::{
    cblas_sasum,
    cblas_dasum,
    cblas_scasum,
    cblas_dzasum,
};


#[inline]
fn approx_eq_f32(a: f32, b: f32, rel: f32, abs: f32) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "asum f32 mismatch: a={a}, b={b}, diff={diff}"
    );
}

#[inline] 
fn approx_eq_f64(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "asum f64 mismatch: a={a}, b={b}, diff={diff}"
    );
}


#[test]
fn asum_edge_n0() {
    let x: [f32; 0] = [];
    let rusty1 = sasum(0, &x, 1);
    assert_eq!(rusty1, 0.0);
}


#[test]
fn sasum_matches_cblas() {
    let n     = 1024usize;
    let mut x: Vec<f32> = (0..n)
        .map(|i| (i as f32 - 512.0) * 0.00123)
        .collect();

    let rusty1 = sasum(n, &x, 1);
    let cblas1 = unsafe { cblas_sasum(n as i32, x.as_ptr(), 1) };

    for i in (0..n).step_by(7) { x[i] = -x[i]; }

    let rusty2 = sasum(n, &x, 1);
    let cblas2 = unsafe {
        cblas_sasum(
            n as i32, 
            x.as_ptr(), 
            1
        ) 
    };

    approx_eq_f32(rusty1, cblas1, 1e-6, 1e-6);
    approx_eq_f32(rusty2, cblas2, 1e-6, 1e-6);
}

#[test]
fn sasum_matches_cblas_stride() {
    let n      = 511usize;
    let stride = 2isize;

    let mut x: Vec<f32> = (0..n*2)
        .map(|_| 0.0)
        .collect();
    for i in 0..n {
        x[2*i] = (i as f32 * 0.37 - 42.0).sin();
    }

    let rusty1 = sasum(n, &x, stride);
    let cblas1 = unsafe { 
        cblas_sasum(
            n as i32, 
            x.as_ptr(), 
            stride as i32
        )
    };

    approx_eq_f32(rusty1, cblas1, 1e-6, 1e-6);
}

#[test]
fn dasum_matches_cblas_stride() {
    let n = 777usize;
    let x: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 1.1 - 100.0).cos() * 1e-3)
        .collect();

    let rusty1 = dasum(n, &x, 1);
    let cblas1 = unsafe { cblas_dasum(n as i32, x.as_ptr(), 1) };
 
    // test non unit stride 
    let stride = 3isize;
    let mut x: Vec<f64> = (0..n*3)
        .map(|_| 0.0)
        .collect();
    for i in 0..n {
        x[i*3] = ((i as f64) * 1.1 - 100.0).cos() * 1e-3;
    }

    let rusty2 = dasum(n, &x, stride);
    let cblas2 = unsafe {
        cblas_dasum(
            n as i32, 
            x.as_ptr(), 
            stride as i32)
    };

    approx_eq_f64(rusty1, cblas1, 1e-12, 1e-12);
    approx_eq_f64(rusty2, cblas2, 1e-12, 1e-12);
}

#[test]
fn scasum_matches_cblas_stride() {
    let n: usize = 512;
    let x_pairs: Vec<[f32; 2]> = (0..n)
        .map(|i| ((i as f32 * 0.07).sin(), (i as f32 * 0.11).cos()))
        .map(|(re, im)| [re, im])
        .collect();

    let rusty1 = {
        let x_flat: &[f32] = unsafe {
            core::slice::from_raw_parts(x_pairs.as_ptr() as *const f32, 2 * n)
        };
        scasum(n, x_flat, 1)
    };
    let cblas1 = unsafe { 
        cblas_scasum(
            n as i32, 
            x_pairs.as_ptr(),
            1)
    };

    // non unit stride 
    let m      = 257usize;
    let stride = 2isize;
    let mut x: Vec<[f32; 2]> = (0..m*2)
        .map(|_| [0.0, 0.0])
        .collect();
    for i in 0..m {
        x[2*i] = [(i as f32 * 0.17).sin(), (i as f32 * 0.21).cos()];
    }

    let rusty2 = {
        let x_flat: &[f32] = unsafe {
            core::slice::from_raw_parts(x.as_ptr() as *const f32, 2 * x.len())
        };
        scasum(m, x_flat, stride)
    };
    let cblas2 = unsafe { 
        cblas_scasum(
            m as i32,
            x.as_ptr(),
            stride as i32
        ) 
    };

    approx_eq_f32(rusty1, cblas1, 1e-6, 1e-6);
    approx_eq_f32(rusty2, cblas2, 1e-6, 1e-6);
}

#[test]
fn dzasum_matches_cblas_stride() {
    let n: usize = 400;
    let x_pairs: Vec<[f64; 2]> = (0..n)
        .map(|i| ((i as f64 * 0.05).sin(), (i as f64 * 0.09).cos()))
        .map(|(re, im)| [re, im])
        .collect();

    let rusty1 = {
        let x_flat: &[f64] = unsafe {
            core::slice::from_raw_parts(x_pairs.as_ptr() as *const f64, 2 * n)
        };
        dzasum(n, x_flat, 1)
    };
    let cblas1 = unsafe {
        cblas_dzasum(
            n as i32, 
            x_pairs.as_ptr(),
            1
        )
    };

    // non unit stride 
    let m      = 129usize;
    let stride = 3isize;
    let mut x: Vec<[f64; 2]> = (0..m*3)
        .map(|_| [0.0, 0.0])
        .collect();
    for i in 0..m {
        x[3*i] = [(i as f64 * 0.13).sin(), (i as f64 * 0.31).cos()];
    }

    let rusty2 = {
        let x_flat: &[f64] = unsafe {
            core::slice::from_raw_parts(x.as_ptr() as *const f64, 2 * x.len())
        };
        dzasum(m, x_flat, stride)
    };
    let cblas2 = unsafe {
        cblas_dzasum(
            m as i32, 
            x.as_ptr(), 
            stride as i32
        )
    };

    approx_eq_f64(rusty1, cblas1, 1e-12, 1e-12);
    approx_eq_f64(rusty2, cblas2, 1e-12, 1e-12);
}
