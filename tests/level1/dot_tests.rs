use blas_src as _; 
use coral::level1::{
    sdot::sdot,
    ddot::ddot,
    cdotc::cdotc,
    cdotu::cdotu,
    zdotc::zdotc,
    zdotu::zdotu,
};

use cblas_sys::{
    cblas_sdot,
    cblas_ddot,
    cblas_cdotc_sub,
    cblas_cdotu_sub,
    cblas_zdotc_sub,
    cblas_zdotu_sub,
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

fn assert_allclose_c32(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);

    for i in (0..a.len()).step_by(2) {
        let (ar, ai) = (a[i], a[i + 1]);
        let (br, bi) = (b[i], b[i + 1]);

        let dr   = (ar - br).abs();
        let di   = (ai - bi).abs();
        let tolr = atol + rtol * ar.abs().max(br.abs());
        let toli = atol + rtol * ai.abs().max(bi.abs());

        assert!(dr <= tolr, "re mismatch at pair {}: {} vs {} (|Δ|={}, tol={})", i/2, ar, br, dr, tolr);
        assert!(di <= toli, "im mismatch at pair {}: {} vs {} (|Δ|={}, tol={})", i/2, ai, bi, di, toli);
    }
}

fn assert_allclose_c64(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);

    for i in (0..a.len()).step_by(2) {
        let (ar, ai) = (a[i], a[i + 1]);
        let (br, bi) = (b[i], b[i + 1]);

        let dr   = (ar - br).abs();
        let di   = (ai - bi).abs();
        let tolr = atol + rtol * ar.abs().max(br.abs());
        let toli = atol + rtol * ai.abs().max(bi.abs());

        assert!(dr <= tolr, "re mismatch at pair {}: {} vs {} (|Δ|={}, tol={})", i/2, ar, br, dr, tolr);
        assert!(di <= toli, "im mismatch at pair {}: {} vs {} (|Δ|={}, tol={})", i/2, ai, bi, di, toli);
    }
}

// just barely leaner for single precision
const RTOL_F32: f32 = 4e-6;
const ATOL_F32: f32 = 4e-6;

const RTOL_F64: f64 = 1e-12;
const ATOL_F64: f64 = 1e-12;

// SDOT //
#[test]
fn sdot_contiguous() {
    let n = 1024usize;

    let x = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * (k as f32));
    let y = make_strided_vec_f32(n, 1, |k| -0.3 + 0.002 * (k as f32));

    let s_coral = sdot(n, &x, 1, &y, 1);
    let s_ref   = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sdot_strided() {
    let n    = 777usize;
    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_vec_f32(n, incx, |k| (0.2 - 0.001 * (k as f32)) * if k % 2 == 0 { 1.0 } else { -1.0 });
    let y = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * (k as f32));

    let s_coral = sdot(n, &x, incx, &y, incy);
    let s_ref   = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            incx as i32,
            y.as_ptr(),
            incy as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sdot_n_zero() {
    let n = 0usize;

    let x = vec![2.0f32; 5];
    let y = vec![1.0f32; 7];

    let s_coral = sdot(n, &x, 1, &y, 1);
    let s_ref   = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn sdot_len1_strided() {
    let n    = 1usize;
    let incx = 5usize;
    let incy = 7usize;

    let x = make_strided_vec_f32(n, incx, |_| 3.14159f32);
    let y = make_strided_vec_f32(n, incy, |_| -2.71828f32);

    let s_coral = sdot(n, &x, incx, &y, incy);
    let s_ref   = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            incx as i32,
            y.as_ptr(),
            incy as i32,
        )
    };

    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

// DDOT //
#[test]
fn ddot_contiguous() {
    let n = 1536usize;

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * (k as f64));
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * (k as f64));

    let s_coral = ddot(n, &x, 1, &y, 1);
    let s_ref   = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn ddot_strided() {
    let n    = 1023usize;
    let incx = 4usize;
    let incy = 3usize;

    let x = make_strided_vec_f64(n, incx, |k| (1.0 - 1e-3 * (k as f64)) * if k % 2 == 0 { 1.0 } else { -1.0 });
    let y = make_strided_vec_f64(n, incy, |k| -0.5 + 2e-3 * (k as f64));

    let s_coral = ddot(n, &x, incx, &y, incy);
    let s_ref   = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            incx as i32,
            y.as_ptr(),
            incy as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn ddot_n_zero() {
    let n = 0usize;

    let x = vec![2.0f64; 6];
    let y = vec![1.0f64; 6];

    let s_coral = ddot(n, &x, 1, &y, 1);
    let s_ref   = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn ddot_len1_strided() {
    let n    = 1usize;
    let incx = 6usize;
    let incy = 5usize;

    let x = make_strided_vec_f64(n, incx, |_| -1234.56789f64);
    let y = make_strided_vec_f64(n, incy, |_|  9876.54321f64);

    let s_coral = ddot(n, &x, incx, &y, incy);
    let s_ref   = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            incx as i32,
            y.as_ptr(),
            incy as i32,
        )
    };

    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

// CDOTC //
#[test]
fn cdotc_contiguous() {
    let n = 800usize;

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (-0.2 + 0.001 * k as f32, 0.3 - 0.003 * k as f32));

    let res_coral = cdotc(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotc_strided() {
    let n    = 513usize;
    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec_f32(n, incx, |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32));
    let y = make_strided_cvec_f32(n, incy, |k| (-0.01 * k as f32, 0.04 + 0.002 * k as f32));

    let res_coral = cdotc(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
            y.as_ptr() as *const [f32; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotc_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f32(4, 1, |k| (k as f32, -(k as f32)));
    let y = make_strided_cvec_f32(4, 1, |k| (1.0 + k as f32, 2.0 - k as f32));

    let res_coral = cdotc(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotc_len1_strided() {
    let n    = 1usize;
    let incx = 7usize;
    let incy = 5usize;

    let x = make_strided_cvec_f32(n, incx, |_| (3.25f32, -4.5f32));
    let y = make_strided_cvec_f32(n, incy, |_| (-1.0f32, 2.0f32));

    let res_coral = cdotc(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
            y.as_ptr() as *const [f32; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

// CDOTU //
#[test]
fn cdotu_contiguous() {
    let n = 800usize;

    let x = make_strided_cvec_f32(n, 1, |k| (0.1 + 0.01 * k as f32, -0.05 + 0.002 * k as f32));
    let y = make_strided_cvec_f32(n, 1, |k| (-0.2 + 0.001 * k as f32, 0.3 - 0.003 * k as f32));

    let res_coral = cdotu(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotu_strided() {
    let n    = 513usize;
    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_cvec_f32(n, incx, |k| (0.02 * k as f32, 0.03 - 0.001 * k as f32));
    let y = make_strided_cvec_f32(n, incy, |k| (-0.01 * k as f32, 0.04 + 0.002 * k as f32));

    let res_coral = cdotu(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
            y.as_ptr() as *const [f32; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotu_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f32(4, 1, |k| (k as f32, -(k as f32)));
    let y = make_strided_cvec_f32(4, 1, |k| (1.0 + k as f32, 2.0 - k as f32));

    let res_coral = cdotu(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn cdotu_len1_strided() {
    let n    = 1usize;
    let incx = 7usize;
    let incy = 5usize;

    let x = make_strided_cvec_f32(n, incx, |_| (3.25f32, -4.5f32));
    let y = make_strided_cvec_f32(n, incy, |_| (-1.0f32, 2.0f32));

    let res_coral = cdotu(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f32; 2];

    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
            y.as_ptr() as *const [f32; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f32; 2],
        );
    }

    assert_allclose_c32(&res_coral, &res_ref, RTOL_F32, ATOL_F32);
}

// ZDOTC //
#[test]
fn zdotc_contiguous() {
    let n = 640usize;

    let x = make_strided_cvec_f64(n, 1, |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (-0.2 + 0.0005 * k as f64, 0.3 + 0.0003 * k as f64));

    let res_coral = zdotc(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotc_strided() {
    let n    = 511usize;
    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_cvec_f64(n, incx, |k| (0.001 * k as f64, -0.002 * k as f64));
    let y = make_strided_cvec_f64(n, incy, |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64));

    let res_coral = zdotc(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
            y.as_ptr() as *const [f64; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotc_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));
    let y = make_strided_cvec_f64(5, 1, |k| (-(k as f64) * 0.2, 0.5 + k as f64 * 0.03));

    let res_coral = zdotc(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotc_len1_strided() {
    let n    = 1usize;
    let incx = 9usize;
    let incy = 4usize;

    let x = make_strided_cvec_f64(n, incx, |_| (123.0f64, -456.0f64));
    let y = make_strided_cvec_f64(n, incy, |_| (-7.0f64, 8.0f64));

    let res_coral = zdotc(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
            y.as_ptr() as *const [f64; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

// ZDOTU //
#[test]
fn zdotu_contiguous() {
    let n = 640usize;

    let x = make_strided_cvec_f64(n, 1, |k| (0.05 + 0.002 * k as f64, 0.1 - 0.001 * k as f64));
    let y = make_strided_cvec_f64(n, 1, |k| (-0.2 + 0.0005 * k as f64, 0.3 + 0.0003 * k as f64));

    let res_coral = zdotu(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotu_strided() {
    let n    = 511usize;
    let incx = 3usize;
    let incy = 2usize;

    let x = make_strided_cvec_f64(n, incx, |k| (0.001 * k as f64, -0.002 * k as f64));
    let y = make_strided_cvec_f64(n, incy, |k| (0.4 - 0.003 * k as f64, -0.1 + 0.004 * k as f64));

    let res_coral = zdotu(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
            y.as_ptr() as *const [f64; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotu_n_zero() {
    let n = 0usize;

    let x = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));
    let y = make_strided_cvec_f64(5, 1, |k| (-(k as f64) * 0.2, 0.5 + k as f64 * 0.03));

    let res_coral = zdotu(n, &x, 1, &y, 1);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn zdotu_len1_strided() {
    let n    = 1usize;
    let incx = 9usize;
    let incy = 4usize;

    let x = make_strided_cvec_f64(n, incx, |_| (123.0f64, -456.0f64));
    let y = make_strided_cvec_f64(n, incy, |_| (-7.0f64, 8.0f64));

    let res_coral = zdotu(n, &x, incx, &y, incy);
    let mut res_ref = [0.0f64; 2];

    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
            y.as_ptr() as *const [f64; 2],
            incy as i32,
            res_ref.as_mut_ptr() as *mut [f64; 2],
        );
    }

    assert_allclose_c64(&res_coral, &res_ref, RTOL_F64, ATOL_F64);
}

