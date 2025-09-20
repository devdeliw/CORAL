use coral::level1::{
    srot::srot,
    srotg::srotg,
    srotm::srotm,
    srotmg::srotmg,
    drot::drot,
    drotg::drotg,
    drotm::drotm,
    drotmg::drotmg,
};

use cblas_sys::{
    cblas_srot,
    cblas_srotg,
    cblas_srotm,
    cblas_srotmg,
    cblas_drot,
    cblas_drotg,
    cblas_drotm,
    cblas_drotmg,
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


// SROT //
#[test]
fn srot_unit_stride() {
    let n = 1024usize;
    let theta = 0.375f32;
    let c = theta.cos();
    let s = theta.sin();

    let x = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * k as f32);
    let y = make_strided_vec_f32(n, 1, |k| -0.3 + 0.002 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srot(n, &mut x_coral, 1, &mut y_coral, 1, c, s);

    unsafe {
        cblas_srot(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srot_strided() {
    let n    = 777usize;
    let incx = 3usize;
    let incy = 2usize;
    let theta = -0.91f32;
    let c = theta.cos();
    let s = theta.sin();

    let x = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);
    let y = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srot(n, &mut x_coral, incx, &mut y_coral, incy, c, s);

    unsafe {
        cblas_srot(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            c,
            s,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srot_n_zero() {
    let n = 0usize;
    let c = 0.8f32;
    let s = 0.6f32;

    let x = vec![2.0f32; 5];
    let y = vec![1.0f32; 7];

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srot(n, &mut x_coral, 1, &mut y_coral, 1, c, s);

    unsafe {
        cblas_srot(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srot_len_one_strided() {
    let n    = 1usize;
    let incx = 5usize;
    let incy = 7usize;
    let c = 0.70710677f32;
    let s = 0.70710677f32;

    let x = make_strided_vec_f32(n, incx, |_| 3.14159f32);
    let y = make_strided_vec_f32(n, incy, |_| -2.71828f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srot(n, &mut x_coral, incx, &mut y_coral, incy, c, s);

    unsafe {
        cblas_srot(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            c,
            s,
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// SROTG //
#[test]
fn srotg_general() {
    let mut a_coral = 3.5f32;
    let mut b_coral = -2.25f32;
    let mut c_coral = 0.0f32;
    let mut s_coral = 0.0f32;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f32;
    let mut s_ref = 0.0f32;

    srotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_srotg(
            &mut a_ref as *mut f32,
            &mut b_ref as *mut f32,
            &mut c_ref as *mut f32,
            &mut s_ref as *mut f32,
        );
    }

    assert_allclose_f32(&[a_coral], &[a_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[b_coral], &[b_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[c_coral], &[c_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn srotg_a_zero() {
    let mut a_coral = 0.0f32;
    let mut b_coral = 5.0f32;
    let mut c_coral = 0.0f32;
    let mut s_coral = 0.0f32;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f32;
    let mut s_ref = 0.0f32;

    srotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_srotg(
            &mut a_ref as *mut f32,
            &mut b_ref as *mut f32,
            &mut c_ref as *mut f32,
            &mut s_ref as *mut f32,
        );
    }

    assert_allclose_f32(&[a_coral], &[a_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[b_coral], &[b_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[c_coral], &[c_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn srotg_b_zero() {
    let mut a_coral = -4.0f32;
    let mut b_coral = 0.0f32;
    let mut c_coral = 0.0f32;
    let mut s_coral = 0.0f32;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f32;
    let mut s_ref = 0.0f32;

    srotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_srotg(
            &mut a_ref as *mut f32,
            &mut b_ref as *mut f32,
            &mut c_ref as *mut f32,
            &mut s_ref as *mut f32,
        );
    }

    assert_allclose_f32(&[a_coral], &[a_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[b_coral], &[b_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[c_coral], &[c_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}

#[test]
fn srotg_mixed_signs() {
    let mut a_coral = -1.25f32;
    let mut b_coral = -7.5f32;
    let mut c_coral = 0.0f32;
    let mut s_coral = 0.0f32;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f32;
    let mut s_ref = 0.0f32;

    srotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_srotg(
            &mut a_ref as *mut f32,
            &mut b_ref as *mut f32,
            &mut c_ref as *mut f32,
            &mut s_ref as *mut f32,
        );
    }

    assert_allclose_f32(&[a_coral], &[a_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[b_coral], &[b_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[c_coral], &[c_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[s_coral], &[s_ref], RTOL_F32, ATOL_F32);
}


// SROTM //
#[test]
fn srotm_flag_neg2_identity_unit_stride() {
    let n = 256usize;
    let param = [-2.0f32, 0.0, 0.0, 0.0, 0.0];

    let x = make_strided_vec_f32(n, 1, |k| 0.1 + 0.01 * k as f32);
    let y = make_strided_vec_f32(n, 1, |k| -0.2 + 0.003 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_srotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srotm_flag_neg1_general_unit_stride() {
    let n = 320usize;
    let param = [-1.0f32, 1.1, -0.4, 0.7, 0.9];

    let x = make_strided_vec_f32(n, 1, |k| 0.03 * k as f32 - 0.5);
    let y = make_strided_vec_f32(n, 1, |k| 0.002 * k as f32 + 0.1);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_srotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srotm_flag_zero_unit_stride() {
    let n = 300usize;
    let param = [0.0f32, 0.0, -0.6, 0.8, 0.0];

    let x = make_strided_vec_f32(n, 1, |k| 0.2 - 0.001 * k as f32);
    let y = make_strided_vec_f32(n, 1, |k| -0.3 + 0.002 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_srotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srotm_flag_pos1_unit_stride() {
    let n = 300usize;
    let param = [1.0f32, 1.2, 0.0, 0.0, -0.75];

    let x = make_strided_vec_f32(n, 1, |k| 0.01 * k as f32);
    let y = make_strided_vec_f32(n, 1, |k| 1.0 - 0.005 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_srotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srotm_neg1_strided() {
    let n    = 257usize;
    let incx = 2usize;
    let incy = 3usize;
    let param = [-1.0f32, -0.3, 0.5, -0.9, 0.4];

    let x = make_strided_vec_f32(n, incx, |k| 0.05 + 0.01 * k as f32);
    let y = make_strided_vec_f32(n, incy, |k| -0.2 + 0.003 * k as f32);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    srotm(n, &mut x_coral, incx, &mut y_coral, incy, &param);

    unsafe {
        cblas_srotm(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            param.as_ptr(),
        );
    }

    assert_allclose_f32(&x_coral, &x_ref, RTOL_F32, ATOL_F32);
    assert_allclose_f32(&y_coral, &y_ref, RTOL_F32, ATOL_F32);
}


// SROTMG //
#[test]
fn srotmg_general() {
    let mut sd1_coral = 1.5f32;
    let mut sd2_coral = 2.0f32;
    let mut sx1_coral = 0.75f32;
    let     sy1_coral = -1.25f32;
    let mut param_coral = [0.0f32; 5];

    let mut sd1_ref = sd1_coral;
    let mut sd2_ref = sd2_coral;
    let mut sx1_ref = sx1_coral;
    let     sy1_ref = sy1_coral;
    let mut param_ref = [0.0f32; 5];

    srotmg(&mut sd1_coral, &mut sd2_coral, &mut sx1_coral, sy1_coral, &mut param_coral);

    unsafe {
        cblas_srotmg(
            &mut sd1_ref as *mut f32,
            &mut sd2_ref as *mut f32,
            &mut sx1_ref as *mut f32,
            sy1_ref,
            param_ref.as_mut_ptr(),
        );
    }

    assert_allclose_f32(&[sd1_coral], &[sd1_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[sd2_coral], &[sd2_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[sx1_coral], &[sx1_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&param_coral, &param_ref, RTOL_F32, ATOL_F32);
}

#[test]
fn srotmg_scaled_inputs() {
    let mut sd1_coral = 1e-3f32;
    let mut sd2_coral = 3e+2f32;
    let mut sx1_coral = -4.5f32;
    let     sy1_coral = 7.25f32;
    let mut param_coral = [0.0f32; 5];

    let mut sd1_ref = sd1_coral;
    let mut sd2_ref = sd2_coral;
    let mut sx1_ref = sx1_coral;
    let     sy1_ref = sy1_coral;
    let mut param_ref = [0.0f32; 5];

    srotmg(&mut sd1_coral, &mut sd2_coral, &mut sx1_coral, sy1_coral, &mut param_coral);

    unsafe {
        cblas_srotmg(
            &mut sd1_ref as *mut f32,
            &mut sd2_ref as *mut f32,
            &mut sx1_ref as *mut f32,
            sy1_ref,
            param_ref.as_mut_ptr(),
        );
    }

    assert_allclose_f32(&[sd1_coral], &[sd1_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[sd2_coral], &[sd2_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&[sx1_coral], &[sx1_ref], RTOL_F32, ATOL_F32);
    assert_allclose_f32(&param_coral, &param_ref, RTOL_F32, ATOL_F32);
}


// DROT //
#[test]
fn drot_unit_stride() {
    let n = 1536usize;
    let theta = 0.22f64;
    let c = theta.cos();
    let s = theta.sin();

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * k as f64);
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drot(n, &mut x_coral, 1, &mut y_coral, 1, c, s);

    unsafe {
        cblas_drot(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drot_strided() {
    let n    = 1023usize;
    let incx = 4usize;
    let incy = 3usize;
    let theta = -0.61f64;
    let c = theta.cos();
    let s = theta.sin();

    let x = make_strided_vec_f64(n, incx, |k| 1.0 - 1e-3 * k as f64);
    let y = make_strided_vec_f64(n, incy, |k| -0.5 + 2e-3 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drot(n, &mut x_coral, incx, &mut y_coral, incy, c, s);

    unsafe {
        cblas_drot(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            c,
            s,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drot_n_zero() {
    let n = 0usize;
    let c = 0.9f64;
    let s = 0.1f64;

    let x = vec![2.0f64; 6];
    let y = vec![1.0f64; 6];

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drot(n, &mut x_coral, 1, &mut y_coral, 1, c, s);

    unsafe {
        cblas_drot(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drot_len_one_strided() {
    let n    = 1usize;
    let incx = 6usize;
    let incy = 5usize;
    let c = 0.6f64;
    let s = -0.8f64;

    let x = make_strided_vec_f64(n, incx, |_| -1234.56789f64);
    let y = make_strided_vec_f64(n, incy, |_|  9876.54321f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drot(n, &mut x_coral, incx, &mut y_coral, incy, c, s);

    unsafe {
        cblas_drot(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            c,
            s,
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}


// DROTG //
#[test]
fn drotg_general() {
    let mut a_coral = 12.0f64;
    let mut b_coral = -0.75f64;
    let mut c_coral = 0.0f64;
    let mut s_coral = 0.0f64;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f64;
    let mut s_ref = 0.0f64;

    drotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_drotg(
            &mut a_ref as *mut f64,
            &mut b_ref as *mut f64,
            &mut c_ref as *mut f64,
            &mut s_ref as *mut f64,
        );
    }

    assert_allclose_f64(&[a_coral], &[a_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[b_coral], &[b_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[c_coral], &[c_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn drotg_a_zero() {
    let mut a_coral = 0.0f64;
    let mut b_coral = 3.0f64;
    let mut c_coral = 0.0f64;
    let mut s_coral = 0.0f64;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f64;
    let mut s_ref = 0.0f64;

    drotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_drotg(
            &mut a_ref as *mut f64,
            &mut b_ref as *mut f64,
            &mut c_ref as *mut f64,
            &mut s_ref as *mut f64,
        );
    }

    assert_allclose_f64(&[a_coral], &[a_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[b_coral], &[b_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[c_coral], &[c_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn drotg_b_zero() {
    let mut a_coral = -5.0f64;
    let mut b_coral = 0.0f64;
    let mut c_coral = 0.0f64;
    let mut s_coral = 0.0f64;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f64;
    let mut s_ref = 0.0f64;

    drotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_drotg(
            &mut a_ref as *mut f64,
            &mut b_ref as *mut f64,
            &mut c_ref as *mut f64,
            &mut s_ref as *mut f64,
        );
    }

    assert_allclose_f64(&[a_coral], &[a_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[b_coral], &[b_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[c_coral], &[c_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}

#[test]
fn drotg_mixed_signs() {
    let mut a_coral = -2.0f64;
    let mut b_coral = 9.0f64;
    let mut c_coral = 0.0f64;
    let mut s_coral = 0.0f64;

    let mut a_ref = a_coral;
    let mut b_ref = b_coral;
    let mut c_ref = 0.0f64;
    let mut s_ref = 0.0f64;

    drotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral);

    unsafe {
        cblas_drotg(
            &mut a_ref as *mut f64,
            &mut b_ref as *mut f64,
            &mut c_ref as *mut f64,
            &mut s_ref as *mut f64,
        );
    }

    assert_allclose_f64(&[a_coral], &[a_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[b_coral], &[b_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[c_coral], &[c_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[s_coral], &[s_ref], RTOL_F64, ATOL_F64);
}


// DROTM //
#[test]
fn drotm_flag_neg2_identity_unit_stride() {
    let n = 256usize;
    let param = [-2.0f64, 0.0, 0.0, 0.0, 0.0];

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * k as f64);
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_drotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drotm_flag_neg1_general_unit_stride() {
    let n = 320usize;
    let param = [-1.0f64, 1.05, -0.6, 0.8, 0.95];

    let x = make_strided_vec_f64(n, 1, |k| 0.1 * k as f64 - 1.0);
    let y = make_strided_vec_f64(n, 1, |k| 0.004 * k as f64 + 0.2);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_drotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drotm_flag_zero_unit_stride() {
    let n = 300usize;
    let param = [0.0f64, 0.0, -0.25, 0.33, 0.0];

    let x = make_strided_vec_f64(n, 1, |k| 1.0 - 1e-3 * k as f64);
    let y = make_strided_vec_f64(n, 1, |k| -0.5 + 2e-3 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_drotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drotm_flag_pos1_unit_stride() {
    let n = 300usize;
    let param = [1.0f64, 0.75, 0.0, 0.0, -1.2];

    let x = make_strided_vec_f64(n, 1, |k| 0.25 + 0.125 * k as f64);
    let y = make_strided_vec_f64(n, 1, |k| -0.75 + 0.05 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drotm(n, &mut x_coral, 1, &mut y_coral, 1, &param);

    unsafe {
        cblas_drotm(
            n as i32,
            x_ref.as_mut_ptr(),
            1,
            y_ref.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drotm_neg1_strided() {
    let n    = 257usize;
    let incx = 2usize;
    let incy = 3usize;
    let param = [-1.0f64, -0.8, 0.2, -0.1, 1.3];

    let x = make_strided_vec_f64(n, incx, |k| 0.25 + 0.125 * k as f64);
    let y = make_strided_vec_f64(n, incy, |k| -0.75 + 0.05 * k as f64);

    let mut x_coral = x.clone();
    let mut y_coral = y.clone();
    let mut x_ref   = x.clone();
    let mut y_ref   = y.clone();

    drotm(n, &mut x_coral, incx, &mut y_coral, incy, &param);

    unsafe {
        cblas_drotm(
            n as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
            y_ref.as_mut_ptr(),
            incy as i32,
            param.as_ptr(),
        );
    }

    assert_allclose_f64(&x_coral, &x_ref, RTOL_F64, ATOL_F64);
    assert_allclose_f64(&y_coral, &y_ref, RTOL_F64, ATOL_F64);
}


// DROTMG //
#[test]
fn drotmg_general() {
    let mut sd1_coral = 5.0f64;
    let mut sd2_coral = 0.75f64;
    let mut sx1_coral = -2.0f64;
    let     sy1_coral = 3.0f64;
    let mut param_coral = [0.0f64; 5];

    let mut sd1_ref = sd1_coral;
    let mut sd2_ref = sd2_coral;
    let mut sx1_ref = sx1_coral;
    let     sy1_ref = sy1_coral;
    let mut param_ref = [0.0f64; 5];

    drotmg(&mut sd1_coral, &mut sd2_coral, &mut sx1_coral, sy1_coral, &mut param_coral);

    unsafe {
        cblas_drotmg(
            &mut sd1_ref as *mut f64,
            &mut sd2_ref as *mut f64,
            &mut sx1_ref as *mut f64,
            sy1_ref,
            param_ref.as_mut_ptr(),
        );
    }

    assert_allclose_f64(&[sd1_coral], &[sd1_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[sd2_coral], &[sd2_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[sx1_coral], &[sx1_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&param_coral, &param_ref, RTOL_F64, ATOL_F64);
}

#[test]
fn drotmg_scaled_inputs() {
    let mut sd1_coral = 1e+4f64;
    let mut sd2_coral = 1e-5f64;
    let mut sx1_coral = 8.0f64;
    let     sy1_coral = -6.0f64;
    let mut param_coral = [0.0f64; 5];

    let mut sd1_ref = sd1_coral;
    let mut sd2_ref = sd2_coral;
    let mut sx1_ref = sx1_coral;
    let     sy1_ref = sy1_coral;
    let mut param_ref = [0.0f64; 5];

    drotmg(&mut sd1_coral, &mut sd2_coral, &mut sx1_coral, sy1_coral, &mut param_coral);

    unsafe {
        cblas_drotmg(
            &mut sd1_ref as *mut f64,
            &mut sd2_ref as *mut f64,
            &mut sx1_ref as *mut f64,
            sy1_ref,
            param_ref.as_mut_ptr(),
        );
    }

    assert_allclose_f64(&[sd1_coral], &[sd1_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[sd2_coral], &[sd2_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&[sx1_coral], &[sx1_ref], RTOL_F64, ATOL_F64);
    assert_allclose_f64(&param_coral, &param_ref, RTOL_F64, ATOL_F64);
}

