use blas_src as _; 
use coral::level1::{
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
const ATOL_F32: f32 = 1e-6;

const RTOL_F64: f64 = 1e-12;
const ATOL_F64: f64 = 1e-12;

// SASUM // 
#[test]
fn sasum_unit_stride() {
    let n = 1024usize;

    let x = make_strided_vec_f32(n, 1, |k| ((-1i32).pow((k % 2) as u32) as f32) * (0.05 + 0.03 * (k as f32)));

    let s_coral = sasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_sasum(
            n as i32,
            x.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sasum_strided() {
    let n    = 777usize;
    let incx = 3usize;

    let x = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * (k as f32) * if k % 2 == 0 { 1.0 } else { -1.0 });

    let s_coral = sasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_sasum(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sasum_n_zero() {
    let n = 0usize;

    let x = vec![2.0f32; 5];

    let s_coral = sasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_sasum(
            n as i32,
            x.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sasum_len_one_strided() {
    let n    = 1usize;
    let incx = 5usize;

    let x = make_strided_vec_f32(n, incx, |_| -3.14159f32);

    let s_coral = sasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_sasum(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}


// DASUM // 
#[test]
fn dasum_unit_stride() {
    let n = 1536usize;

    let x = make_strided_vec_f64(n, 1, |k| ((-1i32).pow((k % 2) as u32) as f64) * (0.25 + 0.125 * (k as f64)));

    let s_coral = dasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_dasum(
            n as i32,
            x.as_ptr(),
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dasum_strided() {
    let n    = 1023usize;
    let incx = 4usize;

    let x = make_strided_vec_f64(n, incx, |k| (1.0 - 1e-3 * (k as f64)) * if k % 2 == 0 { 1.0 } else { -1.0 });

    let s_coral = dasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_dasum(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dasum_n_zero() {
    let n = 0usize;

    let x = vec![2.0f64; 6];

    let s_coral = dasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_dasum(
            n as i32,
            x.as_ptr(),
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dasum_len_one_strided() {
    let n    = 1usize;
    let incx = 6usize;

    let x = make_strided_vec_f64(n, incx, |_| 1234.56789f64);

    let s_coral = dasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_dasum(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}


// SCASUM // 
#[test]
fn scasum_unit_stride() {
    let n = 800usize;

    let x = make_strided_cvec_f32(n, 1, |k| {
        let re = 0.1 + 0.01 * k as f32;
        let im = -0.05 + 0.002 * k as f32;
        if k % 2 == 0 { ( re, im) } else { (-re, -im) }
    });

    let s_coral = scasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_scasum(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn scasum_strided() {
    let n    = 513usize;
    let incx = 2usize;

    let x = make_strided_cvec_f32(n, incx, |k| {
        let re = 0.02 * k as f32;
        let im = 0.03 - 0.001 * k as f32;
        if k % 2 == 0 { ( re, im) } else { (-re, -im) }
    });

    let s_coral = scasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_scasum(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn scasum_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f32(4, 1, |k| (k as f32, -(k as f32)));

    let s_coral = scasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_scasum(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn scasum_len_one_strided() {
    let n    = 1usize;
    let incx = 7usize;

    let x = make_strided_cvec_f32(n, incx, |_| (3.25f32, -4.5f32));

    let s_coral = scasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_scasum(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}


// DZASUM // 
#[test]
fn dzasum_unit_stride() {
    let n = 640usize;

    let x = make_strided_cvec_f64(n, 1, |k| {
        let re = 0.05 + 0.002 * k as f64;
        let im = 0.1  - 0.001 * k as f64;
        if k % 2 == 0 { ( re, im) } else { (-re, -im) }
    });

    let s_coral = dzasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_dzasum(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dzasum_strided() {
    let n    = 511usize;
    let incx = 3usize;

    let x = make_strided_cvec_f64(n, incx, |k| {
        let re = 0.001 * k as f64;
        let im = -0.002 * k as f64;
        if k % 2 == 0 { ( re, im) } else { (-re, -im) }
    });

    let s_coral = dzasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_dzasum(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dzasum_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));

    let s_coral = dzasum(n, &x, 1);
    let s_ref   = unsafe {
        cblas_dzasum(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn dzasum_len_one_strided() {
    let n    = 1usize;
    let incx = 9usize;

    let x = make_strided_cvec_f64(n, incx, |_| (123.0f64, -456.0f64));

    let s_coral = dzasum(n, &x, incx);
    let s_ref   = unsafe {
        cblas_dzasum(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

