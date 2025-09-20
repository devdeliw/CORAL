use coral::level1::{
    sscal::sscal,
    dscal::dscal,
    cscal::cscal,
    zscal::zscal,
};

use cblas_sys::{
    cblas_sscal,
    cblas_dscal,
    cblas_cscal,
    cblas_zscal,
};

// builders
fn make_strided_vec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut v = vec![0.0f32; (len - 1) * inc + 1];
    let mut idx = 0usize;

    for k in 0..len {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn make_strided_vec_f64(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v = vec![0.0f64; (len - 1) * inc + 1];
    let mut idx = 0usize;

    for k in 0..len {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn make_strided_cvec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> (f32, f32),
) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0usize;

    for k in 0..len {
        let (re, im) = f(k);
        let off = 2 * idx;
        v[off]     = re;
        v[off + 1] = im;
        idx += inc;
    }
    v
}

fn make_strided_cvec_f64(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> (f64, f64),
) -> Vec<f64> {
    let mut v = vec![0.0f64; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0usize;

    for k in 0..len {
        let (re, im) = f(k);
        let off = 2 * idx;
        v[off]     = re;
        v[off + 1] = im;
        idx += inc;
    }
    v
}

// checking vectors are equivalent
fn assert_allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(a.len(), b.len());

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})");
    }
}

fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(a.len(), b.len());

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})");
    }
}

const RTOL_F32: f32 = 1e-6;
const ATOL_F32: f32 = 1e-5;

const RTOL_F64: f64 = 1e-12;
const ATOL_F64: f64 = 1e-12;

#[test]
fn sscal_unit_stride() {
    let n     = 1024usize;
    let alpha = 3.1415926f32;

    let x = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * (k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    sscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}


// SSCAL //  
#[test]
fn sscal_strided() {
    let n     = 777usize;
    let alpha = -0.75f32;
    let incx  = 3usize;

    let x = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * (k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    sscal(n, alpha, &mut x_coral, incx);

    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sscal_alpha_zero() {
    let n     = 4096usize;
    let alpha = 0.0f32;

    let x = make_strided_vec_f32(n, 1, |k| -0.2 + 0.001 * (k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    sscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sscal_alpha_one() {
    let n     = 257usize;
    let alpha = 1.0f32;

    let x = make_strided_vec_f32(n, 2, |k| 0.01 * (k as f32) - 0.3);

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    sscal(n, alpha, &mut x_coral, 2);

    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            2,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sscal_n_zero() {
    let n     = 0usize;
    let alpha = 1.23f32;

    let mut x_coral = vec![1.0f32; 4];
    let mut x_ref   = vec![1.0f32; 4];

    sscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_sscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}


// DSCAL // 
#[test]
fn dscal_unit_stride() {
    let n     = 1536usize;
    let alpha = -2.5f64;

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * (k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    dscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dscal_strided() {
    let n     = 1023usize;
    let alpha = 1.0/3.0;

    let incx  = 4usize;

    let x = make_strided_vec_f64(n, incx, |k| 1.0 - 1e-3 * (k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    dscal(n, alpha, &mut x_coral, incx);

    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dscal_alpha_zero() {
    let n     = 512usize;
    let alpha = 0.0f64;

    let x = make_strided_vec_f64(n, 2, |k| 0.2 - 0.02 * (k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    dscal(n, alpha, &mut x_coral, 2);

    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            2,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dscal_alpha_one() {
    let n     = 129usize;
    let alpha = 1.0f64;

    let x = make_strided_vec_f64(n, 3, |k| -0.5 + 2e-3 * (k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    dscal(n, alpha, &mut x_coral, 3);

    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            3,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dscal_n_zero() {
    let n     = 0usize;
    let alpha = 9.0f64;

    let mut x_coral = vec![1.0f64; 6];
    let mut x_ref   = vec![1.0f64; 6];

    dscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_dscal(
            n as i32,
            alpha,
            x_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}


// CSCAL // 
#[test]
fn cscal_unit_stride() {
    let n     = 800usize;
    let alpha = [0.75f32, -0.20f32]; 

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    cscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cscal_strided() {
    let n     = 513usize;
    let alpha = [-1.1f32, 0.4f32];

    let incx  = 3usize;

    let x = make_strided_cvec_f32(n, incx, |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    cscal(n, alpha, &mut x_coral, incx);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            incx as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cscal_alpha_zero() {
    let n     = 256usize;
    let alpha = [0.0f32, 0.0f32];

    let x = make_strided_cvec_f32(n, 1, |k| (0.001 * k as f32, -0.002 * k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    cscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cscal_pure_real_alpha() {
    let n     = 300usize;
    let alpha = [2.0f32, 0.0f32];

    let x = make_strided_cvec_f32(n, 1, |k| (0.3 - 0.001 * k as f32, 0.25 + 0.002 * k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    cscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cscal_pure_imag_alpha() {
    let n     = 300usize;
    let alpha = [0.0f32, 1.0f32];

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.001 * k as f32, -0.2 + 0.002 * k as f32));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    cscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cscal_n_zero() {
    let n     = 0usize;
    let alpha = [1.0f32, -0.5f32];

    let mut x_coral = make_strided_cvec_f32(5, 2, |k| (k as f32 * 0.1, 1.0 - k as f32 * 0.05));
    let mut x_ref   = x_coral.clone();

    cscal(n, alpha, &mut x_coral, 2);

    unsafe {
        cblas_cscal(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x_ref.as_mut_ptr() as *mut [f32; 2],
            2,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
}


// ZSCAL // 
#[test]
fn zscal_unit_stride() {
    let n     = 640usize;
    let alpha = [1.25f64, -0.75f64];

    let x = make_strided_cvec_f64(n, 1, |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    zscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zscal_strided() {
    let n     = 511usize;
    let alpha = [-0.2f64, 0.9f64];

    let incx  = 2usize;

    let x = make_strided_cvec_f64(n, incx, |k| (0.001 * k as f64, -0.002 * k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    zscal(n, alpha, &mut x_coral, incx);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            incx as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zscal_alpha_zero() {
    let n     = 2048usize;
    let alpha = [0.0f64, 0.0f64];

    let x = make_strided_cvec_f64(n, 1, |k| (1.0 - 0.0001 * k as f64, -0.5 + 0.0002 * k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    zscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zscal_pure_real_alpha() {
    let n     = 333usize;
    let alpha = [2.0f64, 0.0f64];

    let x = make_strided_cvec_f64(n, 1, |k| (-0.15 + 0.0005 * k as f64, 0.07 - 0.0003 * k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    zscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zscal_pure_imag_alpha() {
    let n     = 320usize;
    let alpha = [0.0f64, 1.0f64];

    let x = make_strided_cvec_f64(n, 1, |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64));

    let mut x_coral = x.clone();
    let mut x_ref   = x.clone();

    zscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zscal_n_zero() {
    let n     = 0usize;
    let alpha = [0.5f64, -0.25f64];

    let mut x_coral = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));
    let mut x_ref   = x_coral.clone();

    zscal(n, alpha, &mut x_coral, 1);

    unsafe {
        cblas_zscal(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
}

