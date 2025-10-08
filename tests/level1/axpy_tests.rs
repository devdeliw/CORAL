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

fn make_strided_vec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut v = vec![0.0; (len - 1) * inc + 1];
    let mut idx = 0;
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
    let mut v = vec![0.0; (len - 1) * inc + 1];
    let mut idx = 0;
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
    let mut v = vec![0.0; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0;
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
    let mut v = vec![0.0; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0;
    for k in 0..len {
        let (re, im) = f(k);
        let off = 2 * idx;
        v[off] = re;
        v[off + 1] = im;
        idx += inc;
    }
    v
}

fn assert_allclose_f32(
    a    : &[f32],
    b    : &[f32],
    rtol : f32,
    atol : f32,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
    }
}

fn assert_allclose_f64(
    a    : &[f64],
    b    : &[f64],
    rtol : f64,
    atol : f64,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
    }
}

const RTOL_F32 : f32 = 1e-6;
const ATOL_F32 : f32 = 1e-5;
const RTOL_F64 : f64 = 1e-12;
const ATOL_F64 : f64 = 1e-12;

#[test]
fn real() {
    {
        let n = 1024;
        let alpha = 3.1415926;
        let x = make_strided_vec_f32(
            n,
            1,
            |k| 0.05 + 0.03 * k as f32,
        );
        let y = make_strided_vec_f32(
            n,
            1,
            |k| 0.10 + 0.06 * k as f32,
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        saxpy(
            n,
            alpha as f32,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_saxpy(
                n as i32,
                alpha as f32,
                x.as_ptr(),
                1,
                yr.as_mut_ptr(),
                1,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }
    {
        let n = 777;
        let alpha = -0.75;
        let incx = 3;
        let incy = 2;
        let x = make_strided_vec_f32(
            n,
            incx,
            |k| 0.2 - 0.001 * k as f32,
        );
        let y = make_strided_vec_f32(
            n,
            incy,
            |k| -0.3 + 0.002 * k as f32,
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        saxpy(
            n,
            alpha as f32,
            &x,
            incx,
            &mut yc,
            incy,
        );
        unsafe {
            cblas_saxpy(
                n as i32,
                alpha as f32,
                x.as_ptr(),
                incx as i32,
                yr.as_mut_ptr(),
                incy as i32,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }
    {
        let n = 0;
        let alpha = 1.23;
        let x = vec![2.0; 4];
        let mut yc = vec![1.0; 4];
        let mut yr = vec![1.0; 4];

        saxpy(
            n,
            alpha as f32,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_saxpy(
                n as i32,
                alpha as f32,
                x.as_ptr(),
                1,
                yr.as_mut_ptr(),
                1,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }

    {
        let n = 1536;
        let alpha = -2.5;
        let x = make_strided_vec_f64(
            n,
            1,
            |k| 0.25 + 0.125 * k as f64,
        );
        let y = make_strided_vec_f64(
            n,
            1,
            |k| -0.75 + 0.05 * k as f64,
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        daxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_daxpy(
                n as i32,
                alpha,
                x.as_ptr(),
                1,
                yr.as_mut_ptr(),
                1,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
    {
        let n = 1023;
        let alpha = 0.333333333333;
        let incx = 4;
        let incy = 3;
        let x = make_strided_vec_f64(
            n,
            incx,
            |k| 1.0 - 1e-3 * k as f64,
        );
        let y = make_strided_vec_f64(
            n,
            incy,
            |k| -0.5 + 2e-3 * k as f64,
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        daxpy(
            n,
            alpha,
            &x,
            incx,
            &mut yc,
            incy,
        );
        unsafe {
            cblas_daxpy(
                n as i32,
                alpha,
                x.as_ptr(),
                incx as i32,
                yr.as_mut_ptr(),
                incy as i32,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
    {
        let n = 0;
        let alpha = 9.0;
        let x = vec![2.0; 6];
        let mut yc = vec![1.0; 6];
        let mut yr = vec![1.0; 6];

        daxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_daxpy(
                n as i32,
                alpha,
                x.as_ptr(),
                1,
                yr.as_mut_ptr(),
                1,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
}

#[test]
fn complex() {
    {
        let n = 800;
        let alpha = [0.75, -0.20];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32),
        );
        let y = make_strided_cvec_f32(
            n,
            1,
            |k| (-0.2 + 0.001 * k as f32, 0.3 - 0.003 * k as f32),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        caxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_caxpy(
                n as i32,
                alpha.as_ptr() as *const [f32; 2],
                x.as_ptr() as *const [f32; 2],
                1,
                yr.as_mut_ptr() as *mut [f32; 2],
                1,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }
    {
        let n = 513;
        let alpha = [-1.1, 0.4];
        let incx = 2;
        let incy = 3;
        let x = make_strided_cvec_f32(
            n,
            incx,
            |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32),
        );
        let y = make_strided_cvec_f32(
            n,
            incy,
            |k| (-0.01 * k as f32, 0.04 + 0.002 * k as f32),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        caxpy(
            n,
            alpha,
            &x,
            incx,
            &mut yc,
            incy,
        );
        unsafe {
            cblas_caxpy(
                n as i32,
                alpha.as_ptr() as *const [f32; 2],
                x.as_ptr() as *const [f32; 2],
                incx as i32,
                yr.as_mut_ptr() as *mut [f32; 2],
                incy as i32,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }
    {
        let n = 256;
        let alpha = [0.0, 0.0];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.001 * k as f32, -0.002 * k as f32),
        );
        let y = make_strided_cvec_f32(
            n,
            1,
            |k| (0.5 - 0.01 * k as f32, -0.25 + 0.02 * k as f32),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        caxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_caxpy(
                n as i32,
                alpha.as_ptr() as *const [f32; 2],
                x.as_ptr() as *const [f32; 2],
                1,
                yr.as_mut_ptr() as *mut [f32; 2],
                1,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }
    {
        let n = 300;
        let alpha = [0.0, 1.0];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.1 + 0.001 * k as f32, -0.2 + 0.002 * k as f32),
        );
        let y = make_strided_cvec_f32(
            n,
            1,
            |k| (-0.05 + 0.0015 * k as f32, 0.03 - 0.001 * k as f32),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        caxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_caxpy(
                n as i32,
                alpha.as_ptr() as *const [f32; 2],
                x.as_ptr() as *const [f32; 2],
                1,
                yr.as_mut_ptr() as *mut [f32; 2],
                1,
            );
        }
        assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    }

    {
        let n = 640;
        let alpha = [1.25, -0.75];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64),
        );
        let y = make_strided_cvec_f64(
            n,
            1,
            |k| (-0.2 + 0.0005 * k as f64, 0.3 + 0.0003 * k as f64),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        zaxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_zaxpy(
                n as i32,
                alpha.as_ptr() as *const [f64; 2],
                x.as_ptr() as *const [f64; 2],
                1,
                yr.as_mut_ptr() as *mut [f64; 2],
                1,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
    {
        let n = 511;
        let alpha = [-0.2, 0.9];
        let incx = 3;
        let incy = 2;
        let x = make_strided_cvec_f64(
            n,
            incx,
            |k| (0.001 * k as f64, -0.002 * k as f64),
        );
        let y = make_strided_cvec_f64(
            n,
            incy,
            |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        zaxpy(
            n,
            alpha,
            &x,
            incx,
            &mut yc,
            incy,
        );
        unsafe {
            cblas_zaxpy(
                n as i32,
                alpha.as_ptr() as *const [f64; 2],
                x.as_ptr() as *const [f64; 2],
                incx as i32,
                yr.as_mut_ptr() as *mut [f64; 2],
                incy as i32,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
    {
        let n = 2048;
        let alpha = [0.0, 0.0];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (0.01 * k as f64, -0.01 * k as f64),
        );
        let y = make_strided_cvec_f64(
            n,
            1,
            |k| (1.0 - 0.0001 * k as f64, -0.5 + 0.0002 * k as f64),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        zaxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_zaxpy(
                n as i32,
                alpha.as_ptr() as *const [f64; 2],
                x.as_ptr() as *const [f64; 2],
                1,
                yr.as_mut_ptr() as *mut [f64; 2],
                1,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
    {
        let n = 333;
        let alpha = [2.0, 0.0];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (0.3 - 0.001 * k as f64, 0.25 + 0.002 * k as f64),
        );
        let y = make_strided_cvec_f64(
            n,
            1,
            |k| (-0.15 + 0.0005 * k as f64, 0.07 - 0.0003 * k as f64),
        );
        let mut yc = y.clone();
        let mut yr = y.clone();

        zaxpy(
            n,
            alpha,
            &x,
            1,
            &mut yc,
            1,
        );
        unsafe {
            cblas_zaxpy(
                n as i32,
                alpha.as_ptr() as *const [f64; 2],
                x.as_ptr() as *const [f64; 2],
                1,
                yr.as_mut_ptr() as *mut [f64; 2],
                1,
            );
        }
        assert_allclose_f64(&yc, &yr, RTOL_F64, ATOL_F64);
    }
}

