use blas_src as _;
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

const RTOL_F32: f32 = 1e-6;
const ATOL_F32: f32 = 1e-5;
const RTOL_F64: f64 = 1e-12;
const ATOL_F64: f64 = 1e-12;

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

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        sscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 777;
        let alpha = -0.75;
        let incx = 3;
        let x = make_strided_vec_f32(
            n,
            incx,
            |k| 0.2 - 0.001 * k as f32,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        sscal(
            n,
            alpha,
            &mut x_coral,
            incx,
        );

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
    {
        let n = 4096;
        let alpha = 0.0;
        let x = make_strided_vec_f32(
            n,
            1,
            |k| -0.2 + 0.001 * k as f32,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        sscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 257;
        let alpha = 1.0;
        let x = make_strided_vec_f32(
            n,
            2,
            |k| 0.01 * k as f32 - 0.3,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        sscal(
            n,
            alpha,
            &mut x_coral,
            2,
        );

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
    {
        let n = 0;
        let alpha = 1.23;

        let mut x_coral = vec![1.0; 4];
        let mut x_ref = vec![1.0; 4];

        sscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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

    {
        let n = 1536;
        let alpha = -2.5;
        let x = make_strided_vec_f64(
            n,
            1,
            |k| 0.25 + 0.125 * k as f64,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        dscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 1023;
        let alpha = 1.0 / 3.0;
        let incx = 4;
        let x = make_strided_vec_f64(
            n,
            incx,
            |k| 1.0 - 1e-3 * k as f64,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        dscal(
            n,
            alpha,
            &mut x_coral,
            incx,
        );

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
    {
        let n = 512;
        let alpha = 0.0;
        let x = make_strided_vec_f64(
            n,
            2,
            |k| 0.2 - 0.02 * k as f64,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        dscal(
            n,
            alpha,
            &mut x_coral,
            2,
        );

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
    {
        let n = 129;
        let alpha = 1.0;
        let x = make_strided_vec_f64(
            n,
            3,
            |k| -0.5 + 2e-3 * k as f64,
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        dscal(
            n,
            alpha,
            &mut x_coral,
            3,
        );

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
    {
        let n = 0;
        let alpha = 9.0;

        let mut x_coral = vec![1.0; 6];
        let mut x_ref = vec![1.0; 6];

        dscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 513;
        let alpha = [-1.1, 0.4];
        let incx = 3;
        let x = make_strided_cvec_f32(
            n,
            incx,
            |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            incx,
        );

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
    {
        let n = 256;
        let alpha = [0.0, 0.0];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.001 * k as f32, -0.002 * k as f32),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 300;
        let alpha = [2.0, 0.0];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.3 - 0.001 * k as f32, 0.25 + 0.002 * k as f32),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 300;
        let alpha = [0.0, 1.0];
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| (0.1 + 0.001 * k as f32, -0.2 + 0.002 * k as f32),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 0;
        let alpha = [1.0, -0.5];

        let mut x_coral = make_strided_cvec_f32(
            5,
            2,
            |k| (0.1 * k as f32, 1.0 - 0.05 * k as f32),
        );
        let mut x_ref = x_coral.clone();

        cscal(
            n,
            alpha,
            &mut x_coral,
            2,
        );

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

    {
        let n = 640;
        let alpha = [1.25, -0.75];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 511;
        let alpha = [-0.2, 0.9];
        let incx = 2;
        let x = make_strided_cvec_f64(
            n,
            incx,
            |k| (0.001 * k as f64, -0.002 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            incx,
        );

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
    {
        let n = 2048;
        let alpha = [0.0, 0.0];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (1.0 - 0.0001 * k as f64, -0.5 + 0.0002 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 333;
        let alpha = [2.0, 0.0];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (-0.15 + 0.0005 * k as f64, 0.07 - 0.0003 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 320;
        let alpha = [0.0, 1.0];
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64),
        );

        let mut x_coral = x.clone();
        let mut x_ref = x.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
    {
        let n = 0;
        let alpha = [0.5, -0.25];

        let mut x_coral = make_strided_cvec_f64(
            5,
            1,
            |k| (0.1 * k as f64, 1.0 - 0.05 * k as f64),
        );
        let mut x_ref = x_coral.clone();

        zscal(
            n,
            alpha,
            &mut x_coral,
            1,
        );

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
}

