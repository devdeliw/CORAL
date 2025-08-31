use rusty_blas::level1::{
    sasum::sasum, 
    dasum::dasum, 
    scasum::scasum, 
    dzasum::dzasum, 
};
use cblas_sys::{cblas_sasum, cblas_dasum, cblas_scasum, cblas_dzasum};

fn approx_eq_f32(a: f32, b: f32, rel: f32, abs: f32) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "asum f32 mismatch: a={a}, b={b}, diff={diff}"
    );
}
fn approx_eq_f64(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "asum f64 mismatch: a={a}, b={b}, diff={diff}"
    );
}

#[test]
fn asum_f32_inc1_matches_cblas() {
    let n = 1024usize;
    let mut x: Vec<f32> = (0..n).map(|i| ((i as f32) - 512.0) * 0.00123).collect();

    let mine = sasum(n, &x, 1);
    let c = unsafe { cblas_sasum(n as i32, x.as_ptr(), 1) };
    approx_eq_f32(mine, c, 1e-6, 1e-6);

    for i in (0..n).step_by(7) { x[i] = -x[i]; }
    let mine2 = sasum(n, &x, 1);
    let c2 = unsafe { cblas_sasum(n as i32, x.as_ptr(), 1) };
    approx_eq_f32(mine2, c2, 1e-6, 1e-6);
}

#[test]
fn asum_f32_stride2_matches_cblas() {
    let n = 511usize;
    let stride = 2isize;

    let mut buf = vec![0.0f32; n * 2];
    for i in 0..n { buf[2*i] = (i as f32 * 0.37 - 42.0).sin(); }

    let mine = sasum(n, &buf, stride);
    let c = unsafe { cblas_sasum(n as i32, buf.as_ptr(), stride as i32) };
    approx_eq_f32(mine, c, 1e-6, 1e-6);
}

#[test]
fn asum_f64_inc1_and_stride_matches_cblas() {
    let n = 777usize;
    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.1 - 100.0).cos() * 1e-3).collect();

    let mine = dasum(n, &x, 1);
    let c = unsafe { cblas_dasum(n as i32, x.as_ptr(), 1) };
    approx_eq_f64(mine, c, 1e-12, 1e-12);

    let stride = 3isize;
    let mut buf = vec![0.0f64; n * 3];
    for i in 0..n { buf[i*3] = x[i]; }
    let mine_s = dasum(n, &buf, stride);
    let c_s = unsafe { cblas_dasum(n as i32, buf.as_ptr(), stride as i32) };
    approx_eq_f64(mine_s, c_s, 1e-12, 1e-12);
}

#[test]
fn scasum_inc1_and_stride_matches_cblas() {
    let n = 512usize;

    let x_pairs: Vec<[f32; 2]> = (0..n).map(|i| ((i as f32 * 0.07).sin(), (i as f32 * 0.11).cos()))
        .map(|(re, im)| [re, im]).collect();

    // inc = 1
    let mine_inc1 = {
        let x_flat: &[f32] = unsafe {
            core::slice::from_raw_parts(x_pairs.as_ptr() as *const f32, 2 * n)
        };
        scasum(n, x_flat, 1)
    };
    let c_inc1 = unsafe { cblas_scasum(n as i32, x_pairs.as_ptr(), 1) };
    approx_eq_f32(mine_inc1, c_inc1, 1e-6, 1e-6);

    // inc = 2
    let m = 257usize;
    let stride = 2isize;
    let mut buf2 = vec![[0.0f32; 2]; m * 2];
    for i in 0..m {
        buf2[2*i] = [ (i as f32 * 0.17).sin(), (i as f32 * 0.21).cos() ];
    }
    let mine_stride = {
        let x_flat: &[f32] = unsafe {
            core::slice::from_raw_parts(buf2.as_ptr() as *const f32, 2 * buf2.len())
        };
        scasum(m, x_flat, stride)
    };
    let c_stride = unsafe { cblas_scasum(m as i32, buf2.as_ptr(), stride as i32) };
    approx_eq_f32(mine_stride, c_stride, 1e-6, 1e-6);
}

#[test]
fn dzasum_inc1_and_stride_matches_cblas() {
    let n = 400usize;

    let x_pairs: Vec<[f64; 2]> = (0..n).map(|i| ((i as f64 * 0.05).sin(), (i as f64 * 0.09).cos()))
        .map(|(re, im)| [re, im]).collect();

    // inc = 1
    let mine_inc1 = {
        let x_flat: &[f64] = unsafe {
            core::slice::from_raw_parts(x_pairs.as_ptr() as *const f64, 2 * n)
        };
        dzasum(n, x_flat, 1)
    };
    let c_inc1 = unsafe { cblas_dzasum(n as i32, x_pairs.as_ptr(), 1) };
    approx_eq_f64(mine_inc1, c_inc1, 1e-12, 1e-12);

    // inc = 3
    let m = 129usize;
    let stride = 3isize;
    let mut buf3 = vec![[0.0f64; 2]; m * 3];
    for i in 0..m {
        buf3[3*i] = [ (i as f64 * 0.13).sin(), (i as f64 * 0.31).cos() ];
    }
    let mine_stride = {
        let x_flat: &[f64] = unsafe {
            core::slice::from_raw_parts(buf3.as_ptr() as *const f64, 2 * buf3.len())
        };
        dzasum(m, x_flat, stride)
    };
    let c_stride = unsafe { cblas_dzasum(m as i32, buf3.as_ptr(), stride as i32) };
    approx_eq_f64(mine_stride, c_stride, 1e-12, 1e-12);
}

#[test]
fn asum_edge_n0() {
    let x: [f32; 0] = [];
    let mine = sasum(0, &x, 1);
    assert_eq!(mine, 0.0);
}


#[test]
fn sasum_f32_neg_stride() {
    let x: Vec<f32> = vec![1.0, -2.0, 3.0, 1234.0];

    let n: usize   = 2;
    let incx: isize = -2;

    let got = sasum(n, &x, incx);

    let expected = (3.0 + 1.0) as f32;
    assert!(
        (got - expected).abs() < 1e-6,
        "sasum with negative stride: got {got}, expected {expected}"
    );
}

