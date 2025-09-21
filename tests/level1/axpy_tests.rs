use blas_src as _; 
use coral::level1::{
    saxpy::saxpy,
    daxpy::daxpy,
    caxpy::caxpy,
    zaxpy::zaxpy,
};

use cblas_sys::{
    cblas_saxpy,
    cblas_daxpy,
    cblas_caxpy,
    cblas_zaxpy,
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
        v[off] = re;
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
        v[off] = re;
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
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})");
    }
}

fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(a.len(), b.len());

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (|Δ|={diff}, tol={tol})");
    }
}

const RTOL_F32: f32 = 1e-6;
const ATOL_F32: f32 = 1e-5;

const RTOL_F64: f64 = 1e-12;
const ATOL_F64: f64 = 1e-12;


// SAXPY //
#[test]
fn saxpy_unit_stride() {
    let n = 1024usize;
    let alpha = 3.1415926f32;

    let x = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * (k as f32));
    let y = make_strided_vec_f32(n, 1, |k| 0.10 + 0.06 * (k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    saxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn saxpy_strided() {
    let n = 777usize;
    let alpha = -0.75f32;

    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * (k as f32));
    let y = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * (k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    saxpy(n, alpha, &x, incx, &mut y_coral, incy);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn saxpy_alpha_zero() {
    let n = 4096usize;
    let alpha = 0.0f32;

    let x = make_strided_vec_f32(n, 1, |k| 0.01 * (k as f32));
    let y = make_strided_vec_f32(n, 1, |k| -0.2 + 0.001 * (k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    saxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn saxpy_n_zero() {
    let n = 0usize;
    let alpha = 1.23f32;

    let mut y_coral = vec![1.0f32; 4];
    let mut y_ref   = vec![1.0f32; 4];
    let x           = vec![2.0f32; 4];

    saxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// DAXPY //
#[test]
fn daxpy_unit_stride() {
    let n = 1536usize;
    let alpha = -2.5f64;

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * (k as f64));
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * (k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    daxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_daxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn daxpy_strided() {
    let n = 1023usize;
    let alpha = 0.333333333333f64;

    let incx = 4usize;
    let incy = 3usize;

    let x = make_strided_vec_f64(n, incx, |k| 1.0 - 1e-3 * (k as f64));
    let y = make_strided_vec_f64(n, incy, |k| -0.5 + 2e-3 * (k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    daxpy(n, alpha, &x, incx, &mut y_coral, incy);

    unsafe {
        cblas_daxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn daxpy_alpha_zero() {
    let n = 512usize;
    let alpha = 0.0f64;

    let x = make_strided_vec_f64(n, 2, |k| -0.1 + 0.01 * (k as f64));
    let y = make_strided_vec_f64(n, 2, |k| 0.2 - 0.02 * (k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    daxpy(n, alpha, &x, 2, &mut y_coral, 2);

    unsafe {
        cblas_daxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            2,
            y_ref.as_mut_ptr(),
            2,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn daxpy_n_zero() {
    let n = 0usize;
    let alpha = 9.0f64;

    let mut y_coral = vec![1.0f64; 6];
    let mut y_ref   = vec![1.0f64; 6];
    let x           = vec![2.0f64; 6];

    daxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_daxpy(
            n as i32,
            alpha,
            x.as_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}


// CAXPY // 
#[test]
fn caxpy_unit_stride() {
    let n = 800usize;
    let alpha = [0.75f32, -0.20f32]; 

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (-0.2 + 0.001 * k as f32, 0.3 - 0.003 * k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    caxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_caxpy(
            n as i32,
            alpha.as_ptr() as *const[f32; 2],
            x.as_ptr() as *const[f32; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn caxpy_strided() {
    let n = 513usize;
    let alpha = [-1.1f32, 0.4f32]; 

    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec_f32(n, incx, |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32));
    let y = make_strided_cvec_f32(n, incy, |k| (-0.01 * k as f32, 0.04 + 0.002 * k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    caxpy(n, alpha, &x, incx, &mut y_coral, incy);

    unsafe {
        cblas_caxpy(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x.as_ptr() as *const [f32; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            incy as i32,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn caxpy_alpha_zero() {
    let n = 256usize;
    let alpha = [0.0f32, 0.0f32];

    let x = make_strided_cvec_f32(n, 1, |k| (0.001 * k as f32, -0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (0.5 - 0.01 * k as f32, -0.25 + 0.02 * k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    caxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_caxpy(
            n as i32,
            alpha.as_ptr() as *const [f32; 2], 
            x.as_ptr() as *const [f32; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn caxpy_pure_imag_alpha() {
    let n = 300usize;
    let alpha = [0.0f32, 1.0f32];

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.001 * k as f32, -0.2 + 0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (-0.05 + 0.0015 * k as f32, 0.03 - 0.001 * k as f32));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    caxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_caxpy(
            n as i32,
            alpha.as_ptr() as *const [f32; 2],
            x.as_ptr() as *const [f32; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// ZAXPY // 
#[test]
fn zaxpy_unit_stride() {
    let n = 640usize;
    let alpha = [1.25f64, -0.75f64]; 

    let x = make_strided_cvec_f64(n, 1, |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (-0.2 + 0.0005 * k as f64, 0.3 + 0.0003 * k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    zaxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_zaxpy(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x.as_ptr() as *const [f64; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zaxpy_strided() {
    let n = 511usize;
    let alpha = [-0.2f64, 0.9f64];

    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_cvec_f64(n, incx, |k| (0.001 * k as f64, -0.002 * k as f64));
    let y = make_strided_cvec_f64(n, incy, |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    zaxpy(n, alpha, &x, incx, &mut y_coral, incy);

    unsafe {
        cblas_zaxpy(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x.as_ptr() as *const [f64; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            incy as i32,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zaxpy_alpha_zero() {
    let n = 2048usize;
    let alpha = [0.0f64, 0.0f64];

    let x = make_strided_cvec_f64(n, 1, |k| (0.01 * k as f64, -0.01 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (1.0 - 0.0001 * k as f64, -0.5 + 0.0002 * k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    zaxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_zaxpy(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x.as_ptr() as *const [f64; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zaxpy_pure_real_alpha() {
    let n = 333usize;
    let alpha = [2.0f64, 0.0f64];

    let x = make_strided_cvec_f64(n, 1, |k| (0.3 - 0.001 * k as f64, 0.25 + 0.002 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (-0.15 + 0.0005 * k as f64, 0.07 - 0.0003 * k as f64));

    let mut y_coral = y.clone();
    let mut y_ref   = y.clone();

    zaxpy(n, alpha, &x, 1, &mut y_coral, 1);

    unsafe {
        cblas_zaxpy(
            n as i32,
            alpha.as_ptr() as *const [f64; 2],
            x.as_ptr() as *const [f64; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

