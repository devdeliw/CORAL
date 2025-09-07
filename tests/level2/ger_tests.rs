use cblas_sys::{cblas_sger, cblas_dger, CBLAS_LAYOUT};
use rusty_blas::level2::sger::sger;
use rusty_blas::level2::dger::dger;


fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

fn f64_from_u32(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}


fn fill_general_colmajor_f32(a: &mut [f32], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            a[j * lda + i] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
}

fn fill_general_colmajor_f64(a: &mut [f64], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            a[j * lda + i] = f64_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
}

fn assert_close_colmajor_f32(a: &[f32], b: &[f32], m: usize, n: usize, lda: usize, tol: f32) {
    for j in 0..n {
        for i in 0..m {
            let ai = a[j * lda + i];
            let bi = b[j * lda + i];
            let diff = (ai - bi).abs();
            let scale = 1.0_f32 + ai.abs().max(bi.abs());
            assert!(
                diff <= tol * scale,
                "mismatch at (i={}, j={})  got={}  ref={}  diff={}",
                i, j, ai, bi, diff
            );
        }
    }
}

fn assert_close_colmajor_f64(a: &[f64], b: &[f64], m: usize, n: usize, lda: usize, tol: f64) {
    for j in 0..n {
        for i in 0..m {
            let ai = a[j * lda + i];
            let bi = b[j * lda + i];
            let diff = (ai - bi).abs();
            let scale = 1.0_f64 + ai.abs().max(bi.abs());
            assert!(
                diff <= tol * scale,
                "mismatch (f64) at (i={}, j={})  got={}  ref={}  diff={}",
                i, j, ai, bi, diff
            );
        }
    }
}


fn make_pos_strided_vec_f32(n: usize, inc: isize) -> (Vec<f32>, usize) {
    assert!(inc > 0, "inc must be positive");
    let step = inc as usize;
    let len = if n == 0 { 0 } else { 1 + (n - 1) * step };
    let mut buf = vec![0.0f32; len];
    for i in 0..len {
        buf[i] = f32_from_u32(i as u32 * 37 + 11);
    }
    (buf, step)
}

fn make_pos_strided_vec_f64(n: usize, inc: isize) -> (Vec<f64>, usize) {
    assert!(inc > 0, "inc must be positive");
    let step = inc as usize;
    let len = if n == 0 { 0 } else { 1 + (n - 1) * step };
    let mut buf = vec![0.0f64; len];
    for i in 0..len {
        buf[i] = f64_from_u32(i as u32 * 37 + 11);
    }
    (buf, step)
}



#[test]
fn sger_matches_cblas() {
    let (m, n) = (128, 96);
    let lda = m + 8; 
    let alpha: f32 = 1.000123;

    let mut x = vec![0.0f32; m];
    let mut y = vec![0.0f32; n];
    for i in 0..m { x[i] = f32_from_u32((i as u32) * 17 + 3); }
    for j in 0..n { y[j] = f32_from_u32((j as u32) * 29 + 5); }

    let mut a_rust = vec![0.0f32; lda * n];
    let mut a_ref  = vec![0.0f32; lda * n];
    fill_general_colmajor_f32(&mut a_rust, m, n, lda);
    a_ref.copy_from_slice(&a_rust);

    sger(
        m, n, alpha,
        &x, 1,
        &y, 1,
        &mut a_rust, 1, lda as isize,
    );

    unsafe {
        cblas_sger(
            CBLAS_LAYOUT::CblasColMajor,
            m as i32, n as i32,
            alpha,
            x.as_ptr(), 1,
            y.as_ptr(), 1,
            a_ref.as_mut_ptr(), lda as i32,
        );
    }

    assert_close_colmajor_f32(&a_rust, &a_ref, m, n, lda, 1e-5);
}

#[test]
fn sger_matches_cblas_stride() {
    let (m, n) = (79, 65);
    let lda = m + 13;
    let alpha: f32 = -0.321987;

    let incx: isize = 2;
    let incy: isize = 3;

    let (xbuf, _sx) = make_pos_strided_vec_f32(m, incx);
    let (ybuf, _sy) = make_pos_strided_vec_f32(n, incy);

    let mut a_rust = vec![0.0f32; lda * n];
    let mut a_ref  = vec![0.0f32; lda * n];
    fill_general_colmajor_f32(&mut a_rust, m, n, lda);
    a_ref.copy_from_slice(&a_rust);

    sger(
        m, n, alpha,
        &xbuf, incx,
        &ybuf, incy,
        &mut a_rust, 1, lda as isize,
    );

    unsafe {
        cblas_sger(
            CBLAS_LAYOUT::CblasColMajor,
            m as i32, n as i32,
            alpha,
            xbuf.as_ptr(), incx as i32,
            ybuf.as_ptr(), incy as i32,
            a_ref.as_mut_ptr(), lda as i32,
        );
    }

    assert_close_colmajor_f32(&a_rust, &a_ref, m, n, lda, 2e-5);
}


#[test]
fn dger_matches_cblas() {
    let (m, n) = (128, 96);
    let lda = m + 8; 
    let alpha: f64 = 1.000123_f64;

    let mut x = vec![0.0f64; m];
    let mut y = vec![0.0f64; n];
    for i in 0..m { x[i] = f64_from_u32((i as u32) * 17 + 3); }
    for j in 0..n { y[j] = f64_from_u32((j as u32) * 29 + 5); }

    let mut a_rust = vec![0.0f64; lda * n];
    let mut a_ref  = vec![0.0f64; lda * n];
    fill_general_colmajor_f64(&mut a_rust, m, n, lda);
    a_ref.copy_from_slice(&a_rust);

    dger(
        m, n, alpha,
        &x, 1,
        &y, 1,
        &mut a_rust, 1, lda as isize,
    );

    unsafe {
        cblas_dger(
            CBLAS_LAYOUT::CblasColMajor,
            m as i32, n as i32,
            alpha,
            x.as_ptr(), 1,
            y.as_ptr(), 1,
            a_ref.as_mut_ptr(), lda as i32,
        );
    }

    assert_close_colmajor_f64(&a_rust, &a_ref, m, n, lda, 1e-12);
}

#[test]
fn dger_matches_cblas_stride() {
    let (m, n) = (79, 65);
    let lda = m + 13;
    let alpha: f64 = -0.321987_f64;

    let incx: isize = 2;
    let incy: isize = 3;

    let (xbuf, _sx) = make_pos_strided_vec_f64(m, incx);
    let (ybuf, _sy) = make_pos_strided_vec_f64(n, incy);

    let mut a_rust = vec![0.0f64; lda * n];
    let mut a_ref  = vec![0.0f64; lda * n];
    fill_general_colmajor_f64(&mut a_rust, m, n, lda);
    a_ref.copy_from_slice(&a_rust);

    dger(
        m, n, alpha,
        &xbuf, incx,
        &ybuf, incy,
        &mut a_rust, 1, lda as isize,
    );

    unsafe {
        cblas_dger(
            CBLAS_LAYOUT::CblasColMajor,
            m as i32, n as i32,
            alpha,
            xbuf.as_ptr(), incx as i32,
            ybuf.as_ptr(), incy as i32,
            a_ref.as_mut_ptr(), lda as i32,
        );
    }

    assert_close_colmajor_f64(&a_rust, &a_ref, m, n, lda, 2e-12);
}

