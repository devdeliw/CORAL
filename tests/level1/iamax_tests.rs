use blas_src as _;
use coral::level1::{
    isamax::isamax,
    idamax::idamax,
    icamax::icamax,
    izamax::izamax,
};

use cblas_sys::{
    cblas_isamax,
    cblas_idamax,
    cblas_icamax,
    cblas_izamax,
};

// builders
fn make_strided_vec_f32(
    len : usize,
    inc : usize,
    f   : impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut v   = vec![0.0f32; (len - 1) * inc + 1];
    let mut idx = 0usize;
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
    let mut v   = vec![0.0f64; (len - 1) * inc + 1];
    let mut idx = 0usize;
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
    let mut v   = vec![0.0f32; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0usize;
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
    let mut v   = vec![0.0f64; 2 * ((len - 1) * inc + 1)];
    let mut idx = 0usize;
    for k in 0..len {
        let (re, im) = f(k);
        let off      = 2 * idx;
        v[off]       = re;
        v[off + 1]   = im;
        idx         += inc;
    }
    v
}

// ISAMAX //
#[test]
fn isamax_contiguous() {
    let n = 1024usize;
    let x = make_strided_vec_f32(n, 1, |k| {
        if k == 777 { -1234.5 }
        else { ((k as f32).sin() * 3.0).abs() * if k % 2 == 0 { 1.0 } else { -1.0 } }
    });

    let i_coral = isamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_isamax(
            n as i32,
            x.as_ptr(),
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn isamax_strided() {
    let n    = 777usize;
    let incx = 3usize;
    let x    = make_strided_vec_f32(n, incx, |k| {
        if k == 321 { 9_999.0 }
        else { (0.2 - 0.001 * k as f32) * if k % 2 == 0 { 1.0 } else { -1.0 } }
    });

    let i_coral = isamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_isamax(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn isamax_n_zero() {
    let n = 0usize;
    let x = vec![2.0f32; 5];

    let i_coral = isamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_isamax(
            n as i32,
            x.as_ptr(),
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn isamax_len1_strided() {
    let n    = 1usize;
    let incx = 5usize;
    let x    = make_strided_vec_f32(n, incx, |_| -3.14159f32);

    let i_coral = isamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_isamax(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

// IDAMAX //
#[test]
fn idamax_contiguous() {
    let n = 1536usize;
    let x = make_strided_vec_f64(n, 1, |k| {
        if k == 1024 { -8_888.75 }
        else { ((k as f64).cos() * 1.7).abs() * if k % 2 == 0 { 1.0 } else { -1.0 } }
    });

    let i_coral = idamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_idamax(
            n as i32,
            x.as_ptr(),
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn idamax_strided() {
    let n    = 1023usize;
    let incx = 4usize;
    let x    = make_strided_vec_f64(n, incx, |k| {
        if k == 511 { 12_345.0 }
        else { (1.0 - 1e-3 * k as f64) * if k % 2 == 0 { 1.0 } else { -1.0 } }
    });

    let i_coral = idamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_idamax(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn idamax_n_zero() {
    let n = 0usize;
    let x = vec![2.0f64; 6];

    let i_coral = idamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_idamax(
            n as i32,
            x.as_ptr(),
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn idamax_len1_strided() {
    let n    = 1usize;
    let incx = 6usize;
    let x    = make_strided_vec_f64(n, incx, |_| 1234.56789f64);

    let i_coral = idamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_idamax(
            n as i32,
            x.as_ptr(),
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

// ICAMAX //
#[test]
fn icamax_contiguous() {
    let n = 800usize;
    let x = make_strided_cvec_f32(n, 1, |k| {
        if k == 400 { (999.0, -999.0) }
        else {
            let re = 0.1 + 0.01 * k as f32;
            let im = -0.05 + 0.002 * k as f32;
            if k % 2 == 0 { ( re, im) } else { (-re, -im) }
        }
    });

    let i_coral = icamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_icamax(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn icamax_strided() {
    let n    = 513usize;
    let incx = 2usize;
    let x    = make_strided_cvec_f32(n, incx, |k| {
        if k == 256 { (500.0, -700.0) }
        else {
            let re = 0.02 * k as f32;
            let im = 0.03 - 0.001 * k as f32;
            if k % 2 == 0 { ( re, im) } else { (-re, -im) }
        }
    });

    let i_coral = icamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_icamax(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn icamax_n_zero() {
    let n = 0usize;
    let x = make_strided_cvec_f32(4, 1, |k| (k as f32, -(k as f32)));

    let i_coral = icamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_icamax(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn icamax_len1_strided() {
    let n    = 1usize;
    let incx = 7usize;
    let x    = make_strided_cvec_f32(n, incx, |_| (3.25f32, -4.5f32));

    let i_coral = icamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_icamax(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

// IZAMAX //
#[test]
fn izamax_contiguous() {
    let n = 640usize;
    let x = make_strided_cvec_f64(n, 1, |k| {
        if k == 333 { (7777.0, -8888.0) }
        else {
            let re = 0.05 + 0.002 * k as f64;
            let im = 0.1  - 0.001 * k as f64;
            if k % 2 == 0 { ( re, im) } else { (-re, -im) }
        }
    });

    let i_coral = izamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_izamax(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn izamax_strided() {
    let n    = 511usize;
    let incx = 3usize;
    let x    = make_strided_cvec_f64(n, incx, |k| {
        if k == 123 { (1.0e5, -1.0e5) }
        else {
            let re = 0.001 * k as f64;
            let im = -0.002 * k as f64;
            if k % 2 == 0 { ( re, im) } else { (-re, -im) }
        }
    });

    let i_coral = izamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_izamax(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn izamax_n_zero() {
    let n = 0usize;
    let x = make_strided_cvec_f64(5, 1, |k| (k as f64 * 0.1, 1.0 - k as f64 * 0.05));

    let i_coral = izamax(
        n,
        &x,
        1,
    );
    let i_ref   = unsafe {
        cblas_izamax(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

#[test]
fn izamax_len1_strided() {
    let n    = 1usize;
    let incx = 9usize;
    let x    = make_strided_cvec_f64(n, incx, |_| (123.0f64, -456.0f64));

    let i_coral = izamax(
        n,
        &x,
        incx,
    );
    let i_ref   = unsafe {
        cblas_izamax(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            incx as i32,
        )
    } as usize;

    assert_eq!(i_coral, i_ref);
}

