use rusty_blas::level1::{
    sscal::sscal, 
    dscal::dscal, 
    cscal::cscal, 
    zscal::zscal, 
    csscal::csscal, 
    zdscal::zdscal
};
use cblas_sys::{cblas_sscal, cblas_dscal, cblas_cscal, cblas_zscal, cblas_csscal, cblas_zdscal};

fn slice_approx_eq_f32(a: &[f32], b: &[f32], rel: f32, abs: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= abs + rel * x.abs().max(y.abs()), "idx {i}: x={x}, y={y}, diff={diff}");
    }
}
fn slice_approx_eq_f64(a: &[f64], b: &[f64], rel: f64, abs: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= abs + rel * x.abs().max(y.abs()), "idx {i}: x={x}, y={y}, diff={diff}");
    }
}

#[test]
fn scal_real_inc1_matches_cblas() {
    let n = 4096usize;

    let alpha_f32: f32 = 1.000123;
    let mut a = vec![1.0f32; n];
    let mut b = a.clone();
    sscal(n, alpha_f32, &mut a, 1);
    unsafe { cblas_sscal(n as i32, alpha_f32, b.as_mut_ptr(), 1); }
    slice_approx_eq_f32(&a, &b, 1e-6, 0.0);

    let alpha_f64: f64 = 0.99991;
    let mut c = vec![1.0f64; n];
    let mut d = c.clone();
    dscal(n, alpha_f64, &mut c, 1);
    unsafe { cblas_dscal(n as i32, alpha_f64, d.as_mut_ptr(), 1); }
    slice_approx_eq_f64(&c, &d, 1e-12, 0.0);
}

#[test]
fn scal_real_stride2_matches_cblas() {
    let n = 1024usize;
    let alpha_f32: f32 = -1.2345;
    let mut a = vec![0.0f32; n*2];
    for i in 0..n { a[2*i] = (i as f32 * 0.1).sin(); }
    let mut b = a.clone();

    sscal(n, alpha_f32, &mut a, 2);
    unsafe { cblas_sscal(n as i32, alpha_f32, b.as_mut_ptr(), 2); }

    let used_a: Vec<f32> = (0..n).map(|i| a[2*i]).collect();
    let used_b: Vec<f32> = (0..n).map(|i| b[2*i]).collect();
    slice_approx_eq_f32(&used_a, &used_b, 1e-6, 0.0);
}

#[test]
fn scal_complex_inc1_matches_cblas() {
    let n = 1500usize;

    let alpha_c: [f32; 2] = [0.7, 0.3];
    let mut z = vec![0.0f32; 2*n];
    for i in 0..n { z[2*i] = (i as f32 * 0.2).sin(); z[2*i+1] = (i as f32 * 0.3).cos(); }
    let mut z_ref = z.clone();

    cscal(n, alpha_c, &mut z, 1);
    unsafe { cblas_cscal(n as i32, &alpha_c as *const [f32; 2], z_ref.as_mut_ptr().cast::<[f32;2]>(), 1); }

    slice_approx_eq_f32(&z, &z_ref, 3e-5, 0.0);

    let alpha_r: f32 = -1.1;
    let mut w = vec![0.0f32; 2*n];
    for i in 0..n { w[2*i] = (i as f32 * 0.4).sin(); w[2*i+1] = (i as f32 * 0.5).cos(); }
    let mut w_ref = w.clone();

    csscal(n, alpha_r, &mut w, 1);
    unsafe { cblas_csscal(n as i32, alpha_r, w_ref.as_mut_ptr().cast::<[f32;2]>(), 1); }

    slice_approx_eq_f32(&w, &w_ref, 3e-5, 0.0);
}

#[test]
fn scal_complex64_inc1_matches_cblas() {
    let n = 900usize;

    let alpha_z: [f64; 2] = [0.6, -0.2];
    let mut z = vec![0.0f64; 2*n];
    for i in 0..n { z[2*i] = (i as f64 * 0.2).sin(); z[2*i+1] = (i as f64 * 0.3).cos(); }
    let mut z_ref = z.clone();

    zscal(n, alpha_z, &mut z, 1);
    unsafe { cblas_zscal(n as i32, &alpha_z as *const [f64; 2], z_ref.as_mut_ptr().cast::<[f64;2]>(), 1); }
    slice_approx_eq_f64(&z, &z_ref, 1e-12, 0.0);

    let alpha_rd: f64 = 0.987654321;
    let mut w = vec![0.0f64; 2*n];
    for i in 0..n { w[2*i] = (i as f64 * 0.4).sin(); w[2*i+1] = (i as f64 * 0.5).cos(); }
    let mut w_ref = w.clone();

    zdscal(n, alpha_rd, &mut w, 1);
    unsafe { cblas_zdscal(n as i32, alpha_rd, w_ref.as_mut_ptr().cast::<[f64;2]>(), 1); }
    slice_approx_eq_f64(&w, &w_ref, 1e-12, 0.0);
}

#[test]
fn dscal_stride3_matches_cblas() {
    let n = 777usize;
    let inc = 3isize;

    let mut x = vec![0.0f64; 1 + (n - 1) * (inc as usize)];
    for i in 0..n { x[i * (inc as usize)] = (i as f64 * 0.07).sin(); }
    let mut y = x.clone();

    let alpha = -0.3333333333333f64;
    dscal(n, alpha, &mut x, inc);
    unsafe { cblas_dscal(n as i32, alpha, y.as_mut_ptr(), inc as i32); }

    slice_approx_eq_f64(&x, &y, 1e-12, 0.0);
}

#[test]
fn cscal_nonunit_stride_matches_cblas() {
    let n = 512usize;
    let inc = 2isize; 
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: [f32; 2] = [0.3, 0.9];
    let mut z = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        z[2*k]   = (i as f32 * 0.13).sin();
        z[2*k+1] = (i as f32 * 0.31).cos();
    }
    let mut z_ref = z.clone();

    cscal(n, alpha, &mut z, inc);
    unsafe {
        cblas_cscal(n as i32, &alpha, z_ref.as_mut_ptr().cast::<[f32;2]>(), inc as i32);
    }
    slice_approx_eq_f32(&z, &z_ref, 3e-5, 2e-7);
}

#[test]
fn zscal_nonunit_stride_matches_cblas() {
    let n = 333usize;
    let inc = 3isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: [f64; 2] = [0.8, -0.4];
    let mut z = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        z[2*k]   = (i as f64 * 0.17).sin();
        z[2*k+1] = (i as f64 * 0.19).cos();
    }
    let mut z_ref = z.clone();

    zscal(n, alpha, &mut z, inc);
    unsafe { cblas_zscal(n as i32, &alpha, z_ref.as_mut_ptr().cast::<[f64;2]>(), inc as i32); }
    slice_approx_eq_f64(&z, &z_ref, 1e-12, 0.0);
}

#[test]
fn csscal_nonunit_stride_matches_cblas() {
    let n = 271usize;
    let inc = 3isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: f32 = -1.75;
    let mut z = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        z[2*k]   = (i as f32 * 0.07).sin();
        z[2*k+1] = (i as f32 * 0.05).cos();
    }
    let mut z_ref = z.clone();

    csscal(n, alpha, &mut z, inc);
    unsafe { cblas_csscal(n as i32, alpha, z_ref.as_mut_ptr().cast::<[f32;2]>(), inc as i32); }
    slice_approx_eq_f32(&z, &z_ref, 3e-5, 0.0);
}

#[test]
fn zdscal_nonunit_stride_matches_cblas() {
    let n = 289usize;
    let inc = 2isize;
    let len = 1 + (n - 1) * (inc as usize);

    let alpha: f64 = 1.23456789;
    let mut z = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        z[2*k]   = (i as f64 * 0.09).sin();
        z[2*k+1] = (i as f64 * 0.11).cos();
    }
    let mut z_ref = z.clone();

    zdscal(n, alpha, &mut z, inc);
    unsafe { cblas_zdscal(n as i32, alpha, z_ref.as_mut_ptr().cast::<[f64;2]>(), inc as i32); }
    slice_approx_eq_f64(&z, &z_ref, 1e-12, 0.0);
}

