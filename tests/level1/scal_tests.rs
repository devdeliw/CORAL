use rusty_blas::level1::{
    sscal::sscal,
    dscal::dscal,
    cscal::cscal,
    zscal::zscal,
    csscal::csscal,
    zdscal::zdscal,
};
use cblas_sys::{
    cblas_sscal,
    cblas_dscal,
    cblas_cscal,
    cblas_zscal,
    cblas_csscal,
    cblas_zdscal,
}; 

#[inline] 
fn slice_approx_eq_f32(a: &[f32], b: &[f32], rel: f32, abs: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= abs + rel * x.abs().max(y.abs()),
            "idx {i}: x={x}, y={y}, diff={diff}"
        );
    }
}
#[inline] 
fn slice_approx_eq_f64(a: &[f64], b: &[f64], rel: f64, abs: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= abs + rel * x.abs().max(y.abs()),
            "idx {i}: x={x}, y={y}, diff={diff}"
        );
    }
}

#[test]
fn sscal_matches_cblas() {
    let n = 4096usize;

    let alpha: f32 = 1.000123;
    let mut rusty  = vec![1.0f32; n];
    let mut cblas  = rusty.clone();

    sscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr(),
            1,
        );
    }

    slice_approx_eq_f32(&rusty, &cblas, 1e-6, 0.0);
}

#[test]
fn dscal_matches_cblas() {
    let n = 4096usize;

    let alpha: f64 = 0.99991;
    let mut rusty  = vec![1.0f64; n];
    let mut cblas  = rusty.clone();

    dscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr(),
            1,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

#[test]
fn sscal_matches_cblas_stride() {
    let n    = 1024usize;
    let alpha: f32 = -1.2345;

    let mut rusty = vec![0.0f32; n * 2];
    for i in 0..n {
        rusty[2 * i] = (i as f32 * 0.1).sin();
    }
    let mut cblas = rusty.clone();

    sscal(n, alpha, &mut rusty, 2);
    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr(),
            2,
        );
    }

    let used_rusty: Vec<f32> = (0..n).map(|i| rusty[2 * i]).collect();
    let used_cblas: Vec<f32> = (0..n).map(|i| cblas[2 * i]).collect();

    slice_approx_eq_f32(&used_rusty, &used_cblas, 1e-6, 0.0);
}

#[test]
fn cscal_matches_cblas() {
    let n = 1500usize;

    let alpha: [f32; 2] = [0.7, 0.3];
    let mut rusty       = vec![0.0f32; 2 * n];
    for i in 0..n {
        rusty[2 * i]     = (i as f32 * 0.2).sin();
        rusty[2 * i + 1] = (i as f32 * 0.3).cos();
    }
    let mut cblas = rusty.clone();

    cscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_cscal(
            n as i32,
            &alpha as *const [f32; 2],
            cblas.as_mut_ptr().cast::<[f32; 2]>(),
            1,
        );
    }

    slice_approx_eq_f32(&rusty, &cblas, 3e-5, 0.0);
}

#[test]
fn csscal_matches_cblas() {
    let n = 1500usize;

    let alpha: f32 = -1.1;
    let mut rusty  = vec![0.0f32; 2 * n];
    for i in 0..n {
        rusty[2 * i]     = (i as f32 * 0.4).sin();
        rusty[2 * i + 1] = (i as f32 * 0.5).cos();
    }
    let mut cblas = rusty.clone();

    csscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_csscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr().cast::<[f32; 2]>(),
            1,
        );
    }

    slice_approx_eq_f32(&rusty, &cblas, 3e-5, 0.0);
}

#[test]
fn zscal_matches_cblas() {
    let n = 900usize;

    let alpha: [f64; 2] = [0.6, -0.2];
    let mut rusty       = vec![0.0f64; 2 * n];
    for i in 0..n {
        rusty[2 * i]     = (i as f64 * 0.2).sin();
        rusty[2 * i + 1] = (i as f64 * 0.3).cos();
    }
    let mut cblas = rusty.clone();

    zscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_zscal(
            n as i32,
            &alpha as *const [f64; 2],
            cblas.as_mut_ptr().cast::<[f64; 2]>(),
            1,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

#[test]
fn zdscal_matches_cblas() {
    let n = 900usize;

    let alpha: f64 = 0.987654321;
    let mut rusty  = vec![0.0f64; 2 * n];
    for i in 0..n {
        rusty[2 * i]     = (i as f64 * 0.4).sin();
        rusty[2 * i + 1] = (i as f64 * 0.5).cos();
    }
    let mut cblas = rusty.clone();

    zdscal(n, alpha, &mut rusty, 1);
    unsafe {
        cblas_zdscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr().cast::<[f64; 2]>(),
            1,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

#[test]
fn dscal_matches_cblas_stride() {
    let n   = 777usize;
    let inc = 3isize;

    let mut rusty = vec![0.0f64; 1 + (n - 1) * (inc as usize)];
    for i in 0..n {
        rusty[i * (inc as usize)] = (i as f64 * 0.07).sin();
    }
    let mut cblas = rusty.clone();

    let alpha = -0.3333333333333f64;

    dscal(n, alpha, &mut rusty, inc);
    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr(),
            inc as i32,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

#[test]
fn cscal_matches_cblas_stride() {
    let n   = 512usize;
    let inc = 2isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: [f32; 2] = [0.3, 0.9];
    let mut rusty       = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        rusty[2 * k]     = (i as f32 * 0.13).sin();
        rusty[2 * k + 1] = (i as f32 * 0.31).cos();
    }
    let mut cblas = rusty.clone();

    cscal(n, alpha, &mut rusty, inc);
    unsafe {
        cblas_cscal(
            n as i32,
            &alpha,
            cblas.as_mut_ptr().cast::<[f32; 2]>(),
            inc as i32,
        );
    }

    slice_approx_eq_f32(&rusty, &cblas, 3e-5, 2e-7);
}

#[test]
fn zscal_matches_cblas_stride() {
    let n   = 333usize;
    let inc = 3isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: [f64; 2] = [0.8, -0.4];
    let mut rusty       = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        rusty[2 * k]     = (i as f64 * 0.17).sin();
        rusty[2 * k + 1] = (i as f64 * 0.19).cos();
    }
    let mut cblas = rusty.clone();

    zscal(n, alpha, &mut rusty, inc);
    unsafe {
        cblas_zscal(
            n as i32,
            &alpha,
            cblas.as_mut_ptr().cast::<[f64; 2]>(),
            inc as i32,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

#[test]
fn csscal_matches_cblas_stride() {
    let n   = 271usize;
    let inc = 3isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: f32 = -1.75;
    let mut rusty  = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        rusty[2 * k]     = (i as f32 * 0.07).sin();
        rusty[2 * k + 1] = (i as f32 * 0.05).cos();
    }
    let mut cblas = rusty.clone();

    csscal(n, alpha, &mut rusty, inc);
    unsafe {
        cblas_csscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr().cast::<[f32; 2]>(),
            inc as i32,
        );
    }

    slice_approx_eq_f32(&rusty, &cblas, 3e-5, 0.0);
}

#[test]
fn zdscal_matches_cblas_stride() {
    let n   = 289usize;
    let inc = 2isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: f64 = 1.23456789;
    let mut rusty  = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        rusty[2 * k]     = (i as f64 * 0.09).sin();
        rusty[2 * k + 1] = (i as f64 * 0.11).cos();
    }
    let mut cblas = rusty.clone();

    zdscal(n, alpha, &mut rusty, inc);
    unsafe {
        cblas_zdscal(
            n as i32,
            alpha,
            cblas.as_mut_ptr().cast::<[f64; 2]>(),
            inc as i32,
        );
    }

    slice_approx_eq_f64(&rusty, &cblas, 1e-12, 0.0);
}

