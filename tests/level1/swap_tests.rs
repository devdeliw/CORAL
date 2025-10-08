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

fn make_strided_vec_f32(
    len: usize,
    inc: usize,
    f: impl Fn(usize) -> f32,
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
    len: usize,
    inc: usize,
    f: impl Fn(usize) -> f64,
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
    len: usize,
    inc: usize,
    f: impl Fn(usize) -> (f32, f32),
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
    len: usize,
    inc: usize,
    f: impl Fn(usize) -> (f64, f64),
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
    a: &[f32],
    b: &[f32],
    rtol: f32,
    atol: f32,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
    }
}

fn assert_allclose_f64(
    a: &[f64],
    b: &[f64],
    rtol: f64,
    atol: f64,
) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (delta={diff}, tol={tol})");
    }
}

const RTOL_F32: f32 = 0.0;
const ATOL_F32: f32 = 0.0;
const RTOL_F64: f64 = 0.0;
const ATOL_F64: f64 = 0.0;

#[test]
fn real() {
    {
        let n = 1024;
        let x = make_strided_vec_f32(
            n,
            1,
            |k| 0.05 + 0.03 * k as f32,
        );
        let y = make_strided_vec_f32(
            n,
            1,
            |k| -0.7 + 0.001 * k as f32,
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        sswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 777;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        sswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
    {
        let n = 0;

        let x = vec![2.0; 5];
        let y = vec![1.0; 7];

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        sswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 1;
        let incx = 5;
        let incy = 7;
        let x = make_strided_vec_f32(
            n,
            incx,
            |_| 3.14159,
        );
        let y = make_strided_vec_f32(
            n,
            incy,
            |_| -2.71828,
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        sswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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

    {
        let n = 1536;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        dswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 1023;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        dswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
    {
        let n = 0;

        let x = vec![2.0; 6];
        let y = vec![1.0; 6];

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        dswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 1;
        let incx = 6;
        let incy = 5;
        let x = make_strided_vec_f64(
            n,
            incx,
            |_| -1234.56789,
        );
        let y = make_strided_vec_f64(
            n,
            incy,
            |_| 9876.54321,
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        dswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
}

#[test]
fn complex() {
    {
        let n = 800;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        cswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 513;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        cswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
    {
        let n = 0;

        let x = make_strided_cvec_f32(
            4,
            1,
            |k| (k as f32, -(k as f32)),
        );
        let y = make_strided_cvec_f32(
            4,
            1,
            |k| (1.0 + k as f32, 2.0 - k as f32),
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        cswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 1;
        let incx = 7;
        let incy = 5;
        let x = make_strided_cvec_f32(
            n,
            incx,
            |_| (3.25, -4.5),
        );
        let y = make_strided_cvec_f32(
            n,
            incy,
            |_| (-1.0, 2.0),
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        cswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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

    {
        let n = 640;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        zswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 511;
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

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        zswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
    {
        let n = 0;

        let x = make_strided_cvec_f64(
            5,
            1,
            |k| (0.1 * k as f64, 1.0 - 0.05 * k as f64),
        );
        let y = make_strided_cvec_f64(
            5,
            1,
            |k| (-0.2 * k as f64, 0.5 + 0.03 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        zswap(
            n,
            &mut x_coral,
            1,
            &mut y_coral,
            1,
        );

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
    {
        let n = 1;
        let incx = 9;
        let incy = 4;
        let x = make_strided_cvec_f64(
            n,
            incx,
            |_| (123.0, -456.0),
        );
        let y = make_strided_cvec_f64(
            n,
            incy,
            |_| (-7.0, 8.0),
        );

        let mut x_coral = x.clone();
        let mut y_coral = y.clone();
        let mut x_ref = x.clone();
        let mut y_ref = y.clone();

        zswap(
            n,
            &mut x_coral,
            incx,
            &mut y_coral,
            incy,
        );

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
}

