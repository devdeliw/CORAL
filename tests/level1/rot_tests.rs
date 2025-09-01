use cblas_sys::{
    cblas_srot, cblas_drot,
    cblas_srotg, cblas_drotg,
    cblas_srotm, cblas_drotm,
    cblas_srotmg, cblas_drotmg,
};
use rusty_blas::level1::{
    srot::srot,
    drot::drot,
    srotg::srotg,
    drotg::drotg,
    srotm::srotm,
    drotm::drotm,
    srotmg::srotmg,
    drotmg::drotmg,
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
fn assert_vec_approx_eq_f32(a: &[f32], b: &[f32], rel: f32, abs: f32) {
    assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        approx_eq_f32(a[i], b[i], rel, abs);
    }
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
fn approx_eq_f64(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= abs + rel * a.abs().max(b.abs()),
        "f64 not approx equal: a={a:?}, b={b:?}, diff={diff}, rel={rel}, abs={abs}"
    );
}

#[inline]
fn assert_vec_approx_eq_f64(a: &[f64], b: &[f64], rel: f64, abs: f64) {
    assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        approx_eq_f64(a[i], b[i], rel, abs);
    }
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
fn classify_flag_f32(p: f32) -> i32 {
    if (p + 2.0).abs() < 0.5      { -2 }
    else if (p + 1.0).abs() < 0.5 { -1 }
    else if p.abs() < 0.5         { 0  }
    else if (p - 1.0).abs() < 0.5 { 1  }
    else { panic!("unexpected flag value: {p}") }
}

#[inline]
fn classify_flag_f64(p: f64) -> i32 {
    if (p + 2.0).abs() < 0.5      { -2 }
    else if (p + 1.0).abs() < 0.5 { -1 }
    else if p.abs() < 0.5         { 0  }
    else if (p - 1.0).abs() < 0.5 { 1  }
    else { panic!("unexpected flag value: {p}") }
}

#[test]
fn srot_matches_cblas() {
    let n = 100_013usize;
    let c = 0.8f32;
    let s = 0.6f32;

    let x_init: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f32> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srot(n, &mut x_rusty, 1, &mut y_rusty, 1, c, s);
    unsafe {
        cblas_srot(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn srot_matches_cblas_stride() {
    let n   = 60_000usize;
    let inc = 2usize;
    let c   = 0.8f32;
    let s   = 0.6f32;

    let x_init = make_vec_stride_f32(
        n,
        inc,
        |i| (i as f32) * 0.002 - 0.75 + if i & 1 == 0 { 0.5 } else { -0.5 },
    );
    let y_init = make_vec_stride_f32(
        n,
        inc,
        |i| 0.1 * (i as f32) + 1.0 - ((i % 5) as f32) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srot(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, c, s);
    unsafe {
        cblas_srot(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            c,
            s,
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn drot_n0_noop() {
    let c = 0.8f64;
    let s = 0.6f64;

    let mut x_rusty: Vec<f64> = vec![];
    let mut y_rusty: Vec<f64> = vec![];

    let mut x_cblas: Vec<f64> = vec![];
    let mut y_cblas: Vec<f64> = vec![];

    drot(0, &mut x_rusty, 1, &mut y_rusty, 1, c, s);
    unsafe {
        cblas_drot(
            0,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert!(x_rusty.is_empty() && y_rusty.is_empty());
    assert!(x_cblas.is_empty() && y_cblas.is_empty());
}

#[test]
fn drot_matches_cblas() {
    let n = 100_007usize;
    let c = 0.8f64;
    let s = 0.6f64;

    let x_init: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drot(n, &mut x_rusty, 1, &mut y_rusty, 1, c, s);
    unsafe {
        cblas_drot(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn drot_matches_cblas_stride() {
    let n   = 60_000usize;
    let inc = 2usize;
    let c   = 0.8f64;
    let s   = 0.6f64;

    let x_init = make_vec_stride_f64(
        n,
        inc,
        |i| (i as f64) * 0.002 - 0.75 + ((i & 1) as f64) * 0.5,
    );
    let y_init = make_vec_stride_f64(
        n,
        inc,
        |i| 0.1 * (i as f64) + 1.0 - ((i % 5) as f64) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drot(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, c, s);
    unsafe {
        cblas_drot(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            c,
            s,
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn srotg_matches_cblas() {
    let mut a_rusty: f32 = 3.25;
    let mut b_rusty: f32 = -4.75;
    let mut c_rusty: f32 = 0.0;
    let mut s_rusty: f32 = 0.0;

    let mut a_cblas = a_rusty;
    let mut b_cblas = b_rusty;
    let mut c_cblas: f32 = 0.0;
    let mut s_cblas: f32 = 0.0;

    srotg(&mut a_rusty, &mut b_rusty, &mut c_rusty, &mut s_rusty);
    unsafe {
        cblas_srotg(
            &mut a_cblas,
            &mut b_cblas,
            &mut c_cblas,
            &mut s_cblas,
        );
    }

    approx_eq_f32(a_rusty, a_cblas, 1e-6, 1e-6);
    approx_eq_f32(b_rusty, b_cblas, 1e-6, 1e-6);
    approx_eq_f32(c_rusty, c_cblas, 1e-6, 1e-6);
    approx_eq_f32(s_rusty, s_cblas, 1e-6, 1e-6);
}

#[test]
fn drotg_matches_cblas() {
    let mut a_rusty: f64 = 6.0;
    let mut b_rusty: f64 = -2.0;
    let mut c_rusty: f64 = 0.0;
    let mut s_rusty: f64 = 0.0;

    let mut a_cblas = a_rusty;
    let mut b_cblas = b_rusty;
    let mut c_cblas: f64 = 0.0;
    let mut s_cblas: f64 = 0.0;

    drotg(&mut a_rusty, &mut b_rusty, &mut c_rusty, &mut s_rusty);
    unsafe {
        cblas_drotg(
            &mut a_cblas,
            &mut b_cblas,
            &mut c_cblas,
            &mut s_cblas,
        );
    }

    approx_eq_f64(a_rusty, a_cblas, 1e-12, 1e-12);
    approx_eq_f64(b_rusty, b_cblas, 1e-12, 1e-12);
    approx_eq_f64(c_rusty, c_cblas, 1e-12, 1e-12);
    approx_eq_f64(s_rusty, s_cblas, 1e-12, 1e-12);
}

#[test]
fn srotm_matches_cblas_flagm1() {
    let n = 100_123usize;
    let param: [f32; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2];

    let x_init: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f32> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srotm(n, &mut x_rusty, 1, &mut y_rusty, 1, &param);
    unsafe {
        cblas_srotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn srotm_matches_cblas_stride_flagm2() {
    let n = 60_000usize;
    let inc = 2usize;
    let param: [f32; 5] = [-2.0, 0.0, 0.0, 0.0, 0.0];

    let x_init = make_vec_stride_f32(
        n,
        inc,
        |i| (i as f32) * 0.002 - 0.75 + if i & 1 == 0 { 0.5 } else { -0.5 },
    );
    let y_init = make_vec_stride_f32(
        n,
        inc,
        |i| 0.1 * (i as f32) + 1.0 - ((i % 5) as f32) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srotm(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, &param);
    unsafe {
        cblas_srotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn srotm_matches_cblas_flag0() {
    let n = 100_019usize;
    let param: [f32; 5] = [0.0, 0.0, 0.5, -0.25, 0.0];

    let x_init: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f32> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srotm(n, &mut x_rusty, 1, &mut y_rusty, 1, &param);
    unsafe {
        cblas_srotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn srotm_matches_cblas_stride_flagp1() {
    let n = 60_000usize;
    let inc = 2usize;
    let param: [f32; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];

    let x_init = make_vec_stride_f32(
        n,
        inc,
        |i| (i as f32) * 0.002 - 0.75 + if i & 1 == 0 { 0.5 } else { -0.5 },
    );
    let y_init = make_vec_stride_f32(
        n,
        inc,
        |i| 0.1 * (i as f32) + 1.0 - ((i % 5) as f32) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    srotm(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, &param);
    unsafe {
        cblas_srotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f32(&x_rusty, &x_cblas, 1e-6, 1e-6);
    assert_vec_approx_eq_f32(&y_rusty, &y_cblas, 1e-6, 1e-6);
}

#[test]
fn drotm_full_matches_cblas_flagm1() {
    let n = 100_101usize;
    let param: [f64; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2];

    let x_init: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drotm(n, &mut x_rusty, 1, &mut y_rusty, 1, &param);
    unsafe {
        cblas_drotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn drotm_matches_cblas_stride_flagm2() {
    let n = 60_000usize;
    let inc = 2usize;
    let param: [f64; 5] = [-2.0, 0.0, 0.0, 0.0, 0.0];

    let x_init = make_vec_stride_f64(
        n,
        inc,
        |i| (i as f64) * 0.002 - 0.75 + ((i & 1) as f64) * 0.5,
    );
    let y_init = make_vec_stride_f64(
        n,
        inc,
        |i| 0.1 * (i as f64) + 1.0 - ((i % 5) as f64) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drotm(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, &param);
    unsafe {
        cblas_drotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn drotm_matches_cblas_flag0() {
    let n = 100_057usize;
    let param: [f64; 5] = [0.0, 0.0, 0.5, -0.25, 0.0];

    let x_init: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_init: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drotm(n, &mut x_rusty, 1, &mut y_rusty, 1, &param);
    unsafe {
        cblas_drotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            1,
            y_cblas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn drotm_matches_cblas_stride_flagp1() {
    let n = 60_000usize;
    let inc = 2usize;
    let param: [f64; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];

    let x_init = make_vec_stride_f64(
        n,
        inc,
        |i| (i as f64) * 0.002 - 0.75 + ((i & 1) as f64) * 0.5,
    );
    let y_init = make_vec_stride_f64(
        n,
        inc,
        |i| 0.1 * (i as f64) + 1.0 - ((i % 5) as f64) * 0.02,
    );

    let mut x_rusty  = x_init.clone();
    let mut y_rusty  = y_init.clone();
    let mut x_cblas  = x_init.clone();
    let mut y_cblas  = y_init.clone();

    drotm(n, &mut x_rusty, inc as isize, &mut y_rusty, inc as isize, &param);
    unsafe {
        cblas_drotm(
            n as i32,
            x_cblas.as_mut_ptr(),
            inc as i32,
            y_cblas.as_mut_ptr(),
            inc as i32,
            param.as_ptr(),
        );
    }

    assert_vec_approx_eq_f64(&x_rusty, &x_cblas, 1e-12, 1e-12);
    assert_vec_approx_eq_f64(&y_rusty, &y_cblas, 1e-12, 1e-12);
}

#[test]
fn srotmg_sp2_zero_yields_flagm2() { 
    let sy1: f32 = 0.0;
    let (sd1_init, sd2_init, sx1_init) = (1.5f32, 2.3f32, 0.7f32);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f32; 5];
    let mut param_cblas = [0.0f32; 5];

    srotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_srotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f32(param_rusty[0]), -2);
    assert_eq!(classify_flag_f32(param_cblas[0]), -2);

    approx_eq_f32(sd1_rusty, sd1_cblas, 1e-6, 1e-6);
    approx_eq_f32(sd2_rusty, sd2_cblas, 1e-6, 1e-6);
    approx_eq_f32(sx1_rusty, sx1_cblas, 1e-6, 1e-6);
}

#[test]
fn srotmg_sd1_negative_yields_flagm1() { 
    let sy1: f32 = 1.25;
    let (sd1_init, sd2_init, sx1_init) = (-1.0f32, 2.0f32, 3.0f32);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f32; 5];
    let mut param_cblas = [0.0f32; 5];

    srotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_srotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f32(param_rusty[0]), -1);
    assert_eq!(classify_flag_f32(param_cblas[0]), -1);

    approx_eq_f32(sd1_rusty, sd1_cblas, 1e-6, 1e-6);
    approx_eq_f32(sd2_rusty, sd2_cblas, 1e-6, 1e-6);
    approx_eq_f32(sx1_rusty, sx1_cblas, 1e-6, 1e-6);

    approx_eq_f32(param_rusty[1], param_cblas[1], 1e-6, 1e-6);
    approx_eq_f32(param_rusty[2], param_cblas[2], 1e-6, 1e-6);
    approx_eq_f32(param_rusty[3], param_cblas[3], 1e-6, 1e-6);
    approx_eq_f32(param_rusty[4], param_cblas[4], 1e-6, 1e-6);
}

#[test]
fn srotmg_offdiag_yields_flag0() { 
    let (sd1_init, sd2_init, sx1_init, sy1) = (2.0f32, 0.5f32, 3.0f32, 0.25f32);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f32; 5];
    let mut param_cblas = [0.0f32; 5];

    srotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_srotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f32(param_rusty[0]), 0);
    assert_eq!(classify_flag_f32(param_cblas[0]), 0);

    approx_eq_f32(sd1_rusty, sd1_cblas, 1e-5, 1e-5);
    approx_eq_f32(sd2_rusty, sd2_cblas, 1e-5, 1e-5);
    approx_eq_f32(sx1_rusty, sx1_cblas, 1e-5, 1e-5);

    approx_eq_f32(param_rusty[2], param_cblas[2], 1e-5, 1e-5);
    approx_eq_f32(param_rusty[3], param_cblas[3], 1e-5, 1e-5);
}

#[test]
fn srotmg_diag_matches_cblas_flagp1() { 
    let (sd1_init, sd2_init, sx1_init, sy1) = (0.5f32, 2.0f32, 0.5f32, 3.0f32);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f32; 5];
    let mut param_cblas = [0.0f32; 5];

    srotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_srotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f32(param_rusty[0]), 1);
    assert_eq!(classify_flag_f32(param_cblas[0]), 1);

    approx_eq_f32(sd1_rusty, sd1_cblas, 1e-5, 1e-5);
    approx_eq_f32(sd2_rusty, sd2_cblas, 1e-5, 1e-5);
    approx_eq_f32(sx1_rusty, sx1_cblas, 1e-5, 1e-5);

    approx_eq_f32(param_rusty[1], param_cblas[1], 1e-5, 1e-5);
    approx_eq_f32(param_rusty[4], param_cblas[4], 1e-5, 1e-5);
}

#[test]
fn drotmg_sp2_zero_yields_flagm2() { 
    let sy1: f64 = 0.0;
    let (sd1_init, sd2_init, sx1_init) = (1.5f64, 2.3f64, 0.7f64);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f64; 5];
    let mut param_cblas = [0.0f64; 5];

    drotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_drotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f64(param_rusty[0]), -2);
    assert_eq!(classify_flag_f64(param_cblas[0]), -2);

    approx_eq_f64(sd1_rusty, sd1_cblas, 1e-12, 1e-12);
    approx_eq_f64(sd2_rusty, sd2_cblas, 1e-12, 1e-12);
    approx_eq_f64(sx1_rusty, sx1_cblas, 1e-12, 1e-12);
}

#[test]
fn drotmg_sd1_negative_yields_flagm1() { 
    let sy1: f64 = 1.25;
    let (sd1_init, sd2_init, sx1_init) = (-1.0f64, 2.0f64, 3.0f64);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f64; 5];
    let mut param_cblas = [0.0f64; 5];

    drotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_drotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f64(param_rusty[0]), -1);
    assert_eq!(classify_flag_f64(param_cblas[0]), -1);

    approx_eq_f64(sd1_rusty, sd1_cblas, 1e-12, 1e-12);
    approx_eq_f64(sd2_rusty, sd2_cblas, 1e-12, 1e-12);
    approx_eq_f64(sx1_rusty, sx1_cblas, 1e-12, 1e-12);

    approx_eq_f64(param_rusty[1], param_cblas[1], 1e-12, 1e-12);
    approx_eq_f64(param_rusty[2], param_cblas[2], 1e-12, 1e-12);
    approx_eq_f64(param_rusty[3], param_cblas[3], 1e-12, 1e-12);
    approx_eq_f64(param_rusty[4], param_cblas[4], 1e-12, 1e-12);
}

#[test]
fn drotmg_offdiag_yields_flag0() {
    let (sd1_init, sd2_init, sx1_init, sy1) = (2.0f64, 0.5f64, 3.0f64, 0.25f64);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f64; 5];
    let mut param_cblas = [0.0f64; 5];

    drotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_drotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f64(param_rusty[0]), 0);
    assert_eq!(classify_flag_f64(param_cblas[0]), 0);

    approx_eq_f64(sd1_rusty, sd1_cblas, 1e-12, 1e-12);
    approx_eq_f64(sd2_rusty, sd2_cblas, 1e-12, 1e-12);
    approx_eq_f64(sx1_rusty, sx1_cblas, 1e-12, 1e-12);

    approx_eq_f64(param_rusty[2], param_cblas[2], 1e-12, 1e-12);
    approx_eq_f64(param_rusty[3], param_cblas[3], 1e-12, 1e-12);
}

#[test]
fn drotmg_diag_yields_flagp1() { 
    let (sd1_init, sd2_init, sx1_init, sy1) = (0.5f64, 2.0f64, 0.5f64, 3.0f64);

    let (mut sd1_rusty, mut sd2_rusty, mut sx1_rusty) = (sd1_init, sd2_init, sx1_init);
    let (mut sd1_cblas, mut sd2_cblas, mut sx1_cblas) = (sd1_init, sd2_init, sx1_init);
    let mut param_rusty = [0.0f64; 5];
    let mut param_cblas = [0.0f64; 5];

    drotmg(&mut sd1_rusty, &mut sd2_rusty, &mut sx1_rusty, sy1, &mut param_rusty);
    unsafe {
        cblas_drotmg(
            &mut sd1_cblas,
            &mut sd2_cblas,
            &mut sx1_cblas,
            sy1,
            param_cblas.as_mut_ptr(),
        );
    }

    assert_eq!(classify_flag_f64(param_rusty[0]), 1);
    assert_eq!(classify_flag_f64(param_cblas[0]), 1);

    approx_eq_f64(sd1_rusty, sd1_cblas, 1e-12, 1e-12);
    approx_eq_f64(sd2_rusty, sd2_cblas, 1e-12, 1e-12);
    approx_eq_f64(sx1_rusty, sx1_cblas, 1e-12, 1e-12);

    approx_eq_f64(param_rusty[1], param_cblas[1], 1e-12, 1e-12);
    approx_eq_f64(param_rusty[4], param_cblas[4], 1e-12, 1e-12);
}

