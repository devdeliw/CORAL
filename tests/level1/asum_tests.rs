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

fn make_strided_vec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut v   = vec![0.0; (len - 1) * inc + 1];
    let mut idx = 0;
    for k in 0..len {
        v[idx] = f(k);
        idx   += inc;
    }
    v
}

fn make_strided_vec_f64(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut v   = vec![0.0; (len - 1) * inc + 1];
    let mut idx = 0;
    for k in 0..len {
        v[idx] = f(k);
        idx   += inc;
    }
    v
}

fn make_strided_cvec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> (f32, f32),
) -> Vec<f32> {
    let mut v   = vec![0.0; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0;
    for k in 0..len {
        let (re, im) = f(k);
        let off      = 2 * idx;
        v[off]       = re;
        v[off + 1]   = im;
        idx         += inc;
    }
    v
}

fn make_strided_cvec_f64(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> (f64, f64),
) -> Vec<f64> {
    let mut v   = vec![0.0; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0;
    for k in 0..len {
        let (re, im) = f(k);
        let off      = 2 * idx;
        v[off]       = re;
        v[off + 1]   = im;
        idx         += inc;
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
const ATOL_F32 : f32 = 1e-6;
const RTOL_F64 : f64 = 1e-12;
const ATOL_F64 : f64 = 1e-12;

#[test]
fn real() {
    {
        let n = 1024;
        let x = make_strided_vec_f32(
            n,
            1,
            |k| (-1f32).powi((k % 2) as i32) * (0.05 + 0.03 * k as f32),
        );
        let coral = sasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_sasum(
                n as i32,
                x.as_ptr(),
                1,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n    = 777;
        let incx = 3;
        let x = make_strided_vec_f32(
            n,
            incx,
            |k| 0.2 - 0.001 * k as f32 * if k % 2 == 0 { 1.0 } else { -1.0 },
        );
        let coral = sasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_sasum(
                n as i32,
                x.as_ptr(),
                incx as i32,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n = 0;
        let x = vec![2.0; 5];
        let coral = sasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_sasum(
                n as i32,
                x.as_ptr(),
                1,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n    = 1;
        let incx = 5;
        let x = make_strided_vec_f32(
            n,
            incx,
            |_| -3.14159,
        );
        let coral = sasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_sasum(
                n as i32,
                x.as_ptr(),
                incx as i32,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }

    {
        let n = 1536;
        let x = make_strided_vec_f64(
            n,
            1,
            |k| (-1f64).powi((k % 2) as i32) * (0.25 + 0.125 * k as f64),
        );
        let coral = dasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_dasum(
                n as i32,
                x.as_ptr(),
                1,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n = 1023;
        let incx = 4;
        let x = make_strided_vec_f64(
            n,
            incx,
            |k| (1.0 - 1e-3 * k as f64) * if k % 2 == 0 { 1.0 } else { -1.0 },
        );
        let coral = dasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_dasum(
                n as i32,
                x.as_ptr(),
                incx as i32,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n = 0;
        let x = vec![2.0; 6];
        let coral = dasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_dasum(
                n as i32,
                x.as_ptr(),
                1,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n = 1;
        let incx = 6;
        let x = make_strided_vec_f64(
            n,
            incx,
            |_| 1234.56789,
        );
        let coral = dasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_dasum(
                n as i32,
                x.as_ptr(),
                incx as i32,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
}

#[test]
fn complex() {
    {
        let n = 800;
        let x = make_strided_cvec_f32(
            n,
            1,
            |k| {
                let re = 0.1 + 0.01 * k as f32;
                let im = -0.05 + 0.002 * k as f32;
                if k % 2 == 0 { ( re, im) } else { (-re, -im) }
            },
        );
        let coral = scasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_scasum(
                n as i32,
                x.as_ptr() as *const [f32; 2],
                1,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n = 513;
        let incx = 2;
        let x = make_strided_cvec_f32(
            n,
            incx,
            |k| {
                let re = 0.02 * k as f32;
                let im = 0.03 - 0.001 * k as f32;
                if k % 2 == 0 { ( re, im) } else { (-re, -im) }
            },
        );
        let coral = scasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_scasum(
                n as i32,
                x.as_ptr() as *const [f32; 2],
                incx as i32,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n = 0;
        let x = make_strided_cvec_f32(
            4,
            1,
            |k| (k as f32, -(k as f32)),
        );
        let coral = scasum(
            n,
            &x,
            1,
        );
        let blas = unsafe {
            cblas_scasum(
                n as i32,
                x.as_ptr() as *const [f32; 2],
                1,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }
    {
        let n    = 1;
        let incx = 7;
        let x = make_strided_cvec_f32(
            n,
            incx,
            |_| (3.25, -4.5),
        );
        let coral = scasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_scasum(
                n as i32,
                x.as_ptr() as *const [f32; 2],
                incx as i32,
            )
        };
        assert_allclose_f32(
            &[coral],
            &[blas],
            RTOL_F32,
            ATOL_F32,
        );
    }

    {
        let n = 640;
        let x = make_strided_cvec_f64(
            n,
            1,
            |k| {
                let re = 0.05 + 0.002 * k as f64;
                let im = 0.1  - 0.001 * k as f64;
                if k % 2 == 0 { ( re, im) } else { (-re, -im) }
            },
        );
        let coral = dzasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_dzasum(
                n as i32,
                x.as_ptr() as *const [f64; 2],
                1,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n    = 511;
        let incx = 3;
        let x = make_strided_cvec_f64(
            n,
            incx,
            |k| {
                let re = 0.001 * k as f64;
                let im = -0.002 * k as f64;
                if k % 2 == 0 { ( re, im) } else { (-re, -im) }
            },
        );
        let coral = dzasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_dzasum(
                n as i32,
                x.as_ptr() as *const [f64; 2],
                incx as i32,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n = 0;
        let x = make_strided_cvec_f64(
            5,
            1,
            |k| (0.1 * k as f64, 1.0 - 0.05 * k as f64),
        );
        let coral = dzasum(
            n,
            &x,
            1,
        );
        let blas  = unsafe {
            cblas_dzasum(
                n as i32,
                x.as_ptr() as *const [f64; 2],
                1,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
    {
        let n    = 1;
        let incx = 9;
        let x = make_strided_cvec_f64(
            n,
            incx,
            |_| (123.0, -456.0),
        );
        let coral = dzasum(
            n,
            &x,
            incx,
        );
        let blas  = unsafe {
            cblas_dzasum(
                n as i32,
                x.as_ptr() as *const [f64; 2],
                incx as i32,
            )
        };
        assert_allclose_f64(
            &[coral],
            &[blas],
            RTOL_F64,
            ATOL_F64,
        );
    }
}

