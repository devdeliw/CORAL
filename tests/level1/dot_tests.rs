use cblas_sys::{
    cblas_ddot,
    cblas_sdot,
    cblas_cdotu_sub,
    cblas_cdotc_sub,
    cblas_zdotu_sub,
    cblas_zdotc_sub,
};
use rusty_blas::level1::{
    sdot::sdot,
    ddot::ddot,
    cdotu::cdotu,
    zdotu::zdotu,
    cdotc::cdotc,
    zdotc::zdotc,
};


#[inline]
fn approx_eq_f32(a: f32, b: f32, rel: f32, abs: f32) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "f32 not approx equal: a={a:?}, b={b:?}, diff={diff}, rel={rel}, abs={abs}"
    );
}

#[inline]
fn approx_eq_f64(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "f64 not approx equal: a={a:?}, b={b:?}, diff={diff}, rel={rel}, abs={abs}"
    );
}

#[inline]
fn approx_eq_c32(a: [f32; 2], b: [f32; 2], rel: f32, abs: f32) {
    approx_eq_f32(a[0], b[0], rel, abs);
    approx_eq_f32(a[1], b[1], rel, abs);
}

#[inline]
fn approx_eq_c64(a: [f64; 2], b: [f64; 2], rel: f64, abs: f64) {
    approx_eq_f64(a[0], b[0], rel, abs);
    approx_eq_f64(a[1], b[1], rel, abs);
}

#[inline]
fn make_vec_stride_f32(
    n: usize, 
    inc: usize, 
    f: impl Fn(usize) -> f32
) -> Vec<f32> {

    let len   = if n == 0 { 0 } else { 1 + (n - 1) * inc };
    let mut v = vec![0.0f32; len];
    let mut idx = 0usize;
    for i in 0..n {
        v[idx] = f(i);
        idx += inc;
    }
    v
}

#[inline]
fn make_vec_stride_f64(
    n: usize, 
    inc: usize, 
    f: impl Fn(usize) -> f64
) -> Vec<f64> {

    let len   = if n == 0 { 0 } else { 1 + (n - 1) * inc };
    let mut v = vec![0.0f64; len];
    let mut idx = 0usize;
    for i in 0..n {
        v[idx] = f(i);
        idx += inc;
    }
    v
}

#[inline]
fn make_vec_stride_c32(
    n: usize, 
    inc: usize, 
    f: impl Fn(usize) -> (f32, f32)
) -> Vec<f32> {

    let len   = if n == 0 { 0 } else { 2 * (1 + (n - 1) * inc) };
    let mut v = vec![0.0f32; len];
    let mut idx = 0usize;
    for i in 0..n {
        let (re, im) = f(i);
        v[idx]     = re;
        v[idx + 1] = im;
        idx += 2 * inc;
    }
    v
}

#[inline]
fn make_vec_stride_c64(
    n: usize, 
    inc: usize, 
    f: impl Fn(usize) -> (f64, f64)
) -> Vec<f64> {

    let len   = if n == 0 { 0 } else { 2 * (1 + (n - 1) * inc) };
    let mut v = vec![0.0f64; len];
    let mut idx = 0usize;
    for i in 0..n {
        let (re, im) = f(i);
        v[idx]     = re;
        v[idx + 1] = im;
        idx += 2 * inc;
    }
    v
}


#[test]
fn sdot_n0_is_zero() {
    let x: Vec<f32> = vec![];
    let y: Vec<f32> = vec![];

    let rusty1 = sdot(0, &x, 1, &y, 1);
    let cblas1 = unsafe {
        cblas_sdot(
            0,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    assert_eq!(rusty1.to_bits(), 0.0f32.to_bits());
    assert_eq!(cblas1.to_bits(), 0.0f32.to_bits());
}

#[test]
fn sdot_matches_cblas() {
    let n  = 100_003usize;
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y: Vec<f32> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let rusty1 = sdot(n, &x, 1, &y, 1);
    let cblas1 = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    approx_eq_f32(rusty1, cblas1, 1e-6, 1e-6);
}

#[test]
fn sdot_matches_cblas_stride() {
    let n   = 50_000usize;
    let inc = 2usize;

    let x = make_vec_stride_f32(
        n,
        inc,
        |i| (i as f32) * 0.002 - 0.75 + ((i & 1) as f32) * 0.5,
    );
    let y = make_vec_stride_f32(
        n,
        inc,
        |i| 0.1 * (i as f32) + 1.0 - ((i % 5) as f32) * 0.02,
    );

    let rusty1 = sdot(n, &x, inc as isize, &y, inc as isize);
    let cblas1 = unsafe {
        cblas_sdot(
            n as i32,
            x.as_ptr(),
            inc as i32,
            y.as_ptr(),
            inc as i32,
        )
    };

    approx_eq_f32(rusty1, cblas1, 1e-6, 1e-6);
}

#[test]
fn ddot_matches_cblas() {
    let n  = 120_007usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let rusty1 = ddot(n, &x, 1, &y, 1);
    let cblas1 = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
        )
    };

    approx_eq_f64(rusty1, cblas1, 1e-12, 1e-12);
}

#[test]
fn ddot_matches_cblas_stride() {
    let n   = 60_000usize;
    let inc = 2usize;

    let x = make_vec_stride_f64(
        n,
        inc,
        |i| (i as f64) * 0.002 - 0.75 + ((i & 1) as f64) * 0.5,
    );
    let y = make_vec_stride_f64(
        n,
        inc,
        |i| 0.1 * (i as f64) + 1.0 - ((i % 5) as f64) * 0.02,
    );

    let rusty1 = ddot(n, &x, inc as isize, &y, inc as isize);
    let cblas1 = unsafe {
        cblas_ddot(
            n as i32,
            x.as_ptr(),
            inc as i32,
            y.as_ptr(),
            inc as i32,
        )
    };

    approx_eq_f64(rusty1, cblas1, 1e-12, 1e-12);
}

#[test]
fn cdotu_n0_is_zero() {
    let x: Vec<f32> = vec![];
    let y: Vec<f32> = vec![];

    let rusty1 = cdotu(0, &x, 1, &y, 1);

    let mut cblas1: [f32; 2] = [0.0, 0.0];
    unsafe {
        cblas_cdotu_sub(
            0,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            &mut cblas1 as *mut [f32; 2],
        );
    }

    assert_eq!(rusty1, [0.0, 0.0]);
    assert_eq!(cblas1, [0.0, 0.0]);
}

#[test]
fn cdotu_matches_cblas() {
    let n = 80_003usize;

    let x = make_vec_stride_c32(
        n,
        1,
        |i| ((i as f32) * 0.01 - 0.3, (i % 5) as f32 * 0.02 - 0.05),
    );
    let y = make_vec_stride_c32(
        n,
        1,
        |i| (1.0 + (i % 7) as f32 * 0.03, if i % 3 == 0 { -0.2 } else { 0.1 }),
    );

    let rusty1 = cdotu(n, &x, 1, &y, 1);

    let mut cblas1: [f32; 2] = [0.0, 0.0];
    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            &mut cblas1 as *mut [f32; 2],
        );
    }

    approx_eq_c32(rusty1, cblas1, 1e-5, 1e-5);
}

#[test]
fn cdotu_matches_cblas_stride() {
    let n   = 50_000usize;
    let inc = 2usize;

    let x = make_vec_stride_c32(
        n,
        inc,
        |i| (0.001 * i as f32 - 0.4, 0.002 * i as f32 + 0.1),
    );
    let y = make_vec_stride_c32(
        n,
        inc,
        |i| (1.0 + 0.003 * i as f32, -0.5 + 0.001 * (i % 11) as f32),
    );

    let rusty1 = cdotu(n, &x, inc as isize, &y, inc as isize);

    let mut cblas1: [f32; 2] = [0.0, 0.0];
    unsafe {
        cblas_cdotu_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            inc as i32,
            y.as_ptr() as *const [f32; 2],
            inc as i32,
            &mut cblas1 as *mut [f32; 2],
        );
    }

    approx_eq_c32(rusty1, cblas1, 1e-5, 1e-5);
}

#[test]
fn cdotc_matches_cblas() {
    let n = 77_777usize;

    let x = make_vec_stride_c32(
        n,
        1,
        |i| ((i as f32) * 0.015 - 0.1, 0.2 - (i % 9) as f32 * 0.01),
    );
    let y = make_vec_stride_c32(
        n,
        1,
        |i| (1.0 - (i % 4) as f32 * 0.02, 0.05 * ((i & 3) as f32)),
    );

    let rusty1 = cdotc(n, &x, 1, &y, 1);

    let mut cblas1: [f32; 2] = [0.0, 0.0];
    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            1,
            y.as_ptr() as *const [f32; 2],
            1,
            &mut cblas1 as *mut [f32; 2],
        );
    }

    approx_eq_c32(rusty1, cblas1, 3e-5, 3e-5);
}

#[test]
fn cdotc_matches_cblas_stride() {
    let n   = 40_000usize;
    let inc = 2usize;

    let x = make_vec_stride_c32(
        n,
        inc,
        |i| (0.02 * i as f32 - 0.5, -0.03 * (i as f32)),
    );
    let y = make_vec_stride_c32(
        n,
        inc,
        |i| (0.7 + 0.01 * (i % 13) as f32, 0.25 - 0.02 * (i % 5) as f32),
    );

    let rusty1 = cdotc(n, &x, inc as isize, &y, inc as isize);

    let mut cblas1: [f32; 2] = [0.0, 0.0];
    unsafe {
        cblas_cdotc_sub(
            n as i32,
            x.as_ptr() as *const [f32; 2],
            inc as i32,
            y.as_ptr() as *const [f32; 2],
            inc as i32,
            &mut cblas1 as *mut [f32; 2],
        );
    }

    approx_eq_c32(rusty1, cblas1, 1e-5, 1e-5);
}

#[test]
fn zdotu_n0_is_zero() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];

    let rusty1 = zdotu(0, &x, 1, &y, 1);

    let mut cblas1: [f64; 2] = [0.0, 0.0];
    unsafe {
        cblas_zdotu_sub(
            0,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            &mut cblas1 as *mut [f64; 2],
        );
    }

    assert_eq!(rusty1, [0.0, 0.0]);
    assert_eq!(cblas1, [0.0, 0.0]);
}

#[test]
fn zdotu_matches_cblas() {
    let n = 64_123usize;

    let x = make_vec_stride_c64(
        n,
        1,
        |i| ((i as f64) * 0.01 - 0.3, (i % 5) as f64 * 0.02 - 0.05),
    );
    let y = make_vec_stride_c64(
        n,
        1,
        |i| (1.0 + (i % 7) as f64 * 0.03, if i % 3 == 0 { -0.2 } else { 0.1 }),
    );

    let rusty1 = zdotu(n, &x, 1, &y, 1);

    let mut cblas1: [f64; 2] = [0.0, 0.0];
    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            &mut cblas1 as *mut [f64; 2],
        );
    }

    approx_eq_c64(rusty1, cblas1, 1e-12, 1e-12);
}

#[test]
fn zdotu_matches_cblas_stride() {
    let n   = 35_000usize;
    let inc = 2usize;

    let x = make_vec_stride_c64(
        n,
        inc,
        |i| (0.001 * i as f64 - 0.4, 0.002 * i as f64 + 0.1),
    );
    let y = make_vec_stride_c64(
        n,
        inc,
        |i| (1.0 + 0.003 * i as f64, -0.5 + 0.001 * (i % 11) as f64),
    );

    let rusty1 = zdotu(n, &x, inc as isize, &y, inc as isize);

    let mut cblas1: [f64; 2] = [0.0, 0.0];
    unsafe {
        cblas_zdotu_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            inc as i32,
            y.as_ptr() as *const [f64; 2],
            inc as i32,
            &mut cblas1 as *mut [f64; 2],
        );
    }

    approx_eq_c64(rusty1, cblas1, 1e-12, 1e-12);
}

#[test]
fn zdotc_matches_cblas() {
    let n = 59_999usize;

    let x = make_vec_stride_c64(
        n,
        1,
        |i| ((i as f64) * 0.015 - 0.1, 0.2 - (i % 9) as f64 * 0.01),
    );
    let y = make_vec_stride_c64(
        n,
        1,
        |i| (1.0 - (i % 4) as f64 * 0.02, 0.05 * ((i & 3) as f64)),
    );

    let rusty1 = zdotc(n, &x, 1, &y, 1);

    let mut cblas1: [f64; 2] = [0.0, 0.0];
    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            1,
            y.as_ptr() as *const [f64; 2],
            1,
            &mut cblas1 as *mut [f64; 2],
        );
    }

    approx_eq_c64(rusty1, cblas1, 1e-12, 1e-12);
}

#[test]
fn zdotc_matches_cblas_stride() {
    let n   = 33_333usize;
    let inc = 2usize;

    let x = make_vec_stride_c64(
        n,
        inc,
        |i| (0.02 * i as f64 - 0.5, -0.03 * (i as f64)),
    );
    let y = make_vec_stride_c64(
        n,
        inc,
        |i| (0.7 + 0.01 * (i % 13) as f64, 0.25 - 0.02 * (i % 5) as f64),
    );

    let rusty1 = zdotc(n, &x, inc as isize, &y, inc as isize);

    let mut cblas1: [f64; 2] = [0.0, 0.0];
    unsafe {
        cblas_zdotc_sub(
            n as i32,
            x.as_ptr() as *const [f64; 2],
            inc as i32,
            y.as_ptr() as *const [f64; 2],
            inc as i32,
            &mut cblas1 as *mut [f64; 2],
        );
    }

    approx_eq_c64(rusty1, cblas1, 1e-12, 1e-12);
}

