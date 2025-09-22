use blas_src as _; 
use coral::level1::{
    sswap::sswap,
    dswap::dswap,
    cswap::cswap,
    zswap::zswap,
};

use cblas_sys::{
    cblas_sswap,
    cblas_dswap,
    cblas_cswap,
    cblas_zswap,
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

// swap should be exact 
const RTOL_F32: f32 = 0.0;
const ATOL_F32: f32 = 0.0;

const RTOL_F64: f64 = 0.0;
const ATOL_F64: f64 = 0.0;


// SSWAP // 
#[test]
fn sswap_contiguous() {
    let n = 1024usize;

    let x = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * (k as f32));
    let y = make_strided_vec_f32(n, 1, |k| -0.7 + 0.001 * (k as f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    sswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_sswap(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sswap_strided() {
    let n    = 777usize;
    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * (k as f32));
    let y = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * (k as f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    sswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_sswap(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sswap_n_zero() {
    let n = 0usize;

    let x = vec![2.0f32; 5];
    let y = vec![1.0f32; 7];

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    sswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_sswap(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn sswap_len1_strided() {
    let n    = 1usize;
    let incx = 5usize;
    let incy = 7usize;

    let x = make_strided_vec_f32(n, incx, |_| 3.14159f32);
    let y = make_strided_vec_f32(n, incy, |_| -2.71828f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    sswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_sswap(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// DSWAP // 
#[test]
fn dswap_contiguous() {
    let n = 1536usize;

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * (k as f64));
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * (k as f64));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    dswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_dswap(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dswap_strided() {
    let n    = 1023usize;
    let incx = 4usize;
    let incy = 3usize;

    let x = make_strided_vec_f64(n, incx, |k| 1.0 - 1e-3 * (k as f64));
    let y = make_strided_vec_f64(n, incy, |k| -0.5 + 2e-3 * (k as f64));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    dswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_dswap(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dswap_n_zero() {
    let n = 0usize;

    let x = vec![2.0f64; 6];
    let y = vec![1.0f64; 6];

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    dswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_dswap(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn dswap_len1_strided() {
    let n    = 1usize;
    let incx = 6usize;
    let incy = 5usize;

    let x = make_strided_vec_f64(n, incx, |_| -1234.56789f64);
    let y = make_strided_vec_f64(n, incy, |_|  9876.54321f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    dswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_dswap(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}


// CSWAP // 
#[test]
fn cswap_contiguous() {
    let n = 800usize;

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (-0.2 + 0.001 * k as f32, 0.3 - 0.003 * k as f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    cswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_cswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cswap_strided() {
    let n    = 513usize;
    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec_f32(n, incx, |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32));
    let y = make_strided_cvec_f32(n, incy, |k| (-0.01 * k as f32, 0.04 + 0.002 * k as f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    cswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_cswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f32; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            incy as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cswap_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f32(4, 1, |k| (k as f32, -(k as f32)));
    let y = make_strided_cvec_f32(4, 1, |k| (1.0 + k as f32, 2.0 - k as f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    cswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_cswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f32; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            1,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cswap_len1_strided() {
    let n    = 1usize;
    let incx = 7usize;
    let incy = 5usize;

    let x = make_strided_cvec_f32(n, incx, |_| (3.25f32, -4.5f32));
    let y = make_strided_cvec_f32(n, incy, |_| (-1.0f32, 2.0f32));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    cswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_cswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f32; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f32; 2],
            incy as i32,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// ZSWAP // 
#[test]
fn zswap_contiguous() {
    let n = 640usize;

    let x = make_strided_cvec_f64(n, 1, |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (-0.2 + 0.0005 * k as f64, 0.3 + 0.0003 * k as f64));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    zswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_zswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zswap_strided() {
    let n    = 511usize;
    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_cvec_f64(n, incx, |k| (0.001 * k as f64, -0.002 * k as f64));
    let y = make_strided_cvec_f64(n, incy, |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    zswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_zswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f64; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            incy as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zswap_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));
    let y = make_strided_cvec_f64(5, 1, |k| (-(k as f64) * 0.2, 0.5 + k as f64 * 0.03));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    zswap(n, &mut x_coral, 1, &mut y_coral, 1);

    unsafe {
        cblas_zswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f64; 2],
            1,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            1,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zswap_len1_strided() {
    let n    = 1usize;
    let incx = 9usize;
    let incy = 4usize;

    let x = make_strided_cvec_f64(n, incx, |_| (123.0f64, -456.0f64));
    let y = make_strided_cvec_f64(n, incy, |_| (-7.0f64, 8.0f64));

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    zswap(n, &mut x_coral, incx, &mut y_coral, incy);

    unsafe {
        cblas_zswap(
            n as i32,
            x_ref.as_mut_ptr() as *mut [f64; 2],
            incx as i32,
            y_ref.as_mut_ptr() as *mut [f64; 2],
            incy as i32,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

