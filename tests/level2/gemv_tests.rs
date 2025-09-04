use std::cmp::max;

use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE,
    cblas_sgemv, 
    cblas_dgemv,
};

use rusty_blas::level2::{ 
    trans::Trans, 
    sgemv::sgemv,   
    dgemv::dgemv, 
};

// pseudo random float [-0.5, 0.5] 
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}
#[inline]
fn f64_from_u32(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}


// generate col major mat
fn fill_a_colmajor_f32(a: &mut [f32], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            let idx = j * lda + i;
            a[idx] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
}

fn fill_a_colmajor_f64(a: &mut [f64], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            let idx = j * lda + i;
            a[idx] = f64_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
}

// generate strided vec 
fn fill_vec_strided_f32(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f32_from_u32(0x9E3779B9u32.wrapping_mul(k as u32).wrapping_add(12345));
        pos += inc;
    }
}
fn fill_vec_strided_f64(buf: &mut [f64], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f64_from_u32(0x9E3779B9u32.wrapping_mul(k as u32).wrapping_add(12345));
        pos += inc;
    }
}


fn assert_close_strided_f32(y_test: &[f32], y_ref: &[f32], m: usize, incy: usize, tol: f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut p = 0usize;
    for _ in 0..m {
        let a = y_test[p];
        let b = y_ref[p];
        let abs = (a - b).abs();
        let denom = 1.0f32.max(b.abs());
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tol,
            "mismatch at pos {}: test={} ref={} |abs|={} rel={}",
            p, a, b, abs, rel
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {}, max_rel_diff = {}", max_abs, max_rel);
}

fn assert_close_strided_f64(y_test: &[f64], y_ref: &[f64], m: usize, incy: usize, tol: f64) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut p = 0usize;
    for _ in 0..m {
        let a = y_test[p];
        let b = y_ref[p];
        let abs = (a - b).abs();
        let denom = 1.0f64.max(b.abs());
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tol,
            "mismatch at pos {}: test={} ref={} |abs|={} rel={}",
            p, a, b, abs, rel
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {}, max_rel_diff = {}", max_abs, max_rel);
}


unsafe fn cblas_sgemv_colmajor_no_trans(
    m: i32, n: i32,
    alpha: f32,
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: f32,
    y: *mut f32, incy: i32,
) { unsafe { 
    cblas_sgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy,
    );
}} 

unsafe fn cblas_sgemv_colmajor_trans(
    m: i32, n: i32,
    alpha: f32,
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: f32,
    y: *mut f32, incy: i32,
) { unsafe {
    cblas_sgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy,
    );
}}

unsafe fn cblas_dgemv_colmajor_no_trans(
    m: i32, n: i32,
    alpha: f64,
    a: *const f64, lda: i32,
    x: *const f64, incx: i32,
    beta: f64,
    y: *mut f64, incy: i32,
) { unsafe { 
    cblas_dgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy,
    );
}}

unsafe fn cblas_dgemv_colmajor_trans(
    m: i32, n: i32,
    alpha: f64,
    a: *const f64, lda: i32,
    x: *const f64, incx: i32,
    beta: f64,
    y: *mut f64, incy: i32,
) { unsafe {
    cblas_dgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy,
    );
}}



#[test]
fn sgemv_matches_cblas() {
    let m = 67usize;
    let n = 53usize;

    let lda = max(m, m + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_a_colmajor_f32(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    fill_vec_strided_f32(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f32;
    let beta = -0.75f32;

    sgemv(
        Trans::NoTrans,
        m, n,
        alpha,
        &a,
        1,                 
        lda as isize,   
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_sgemv_colmajor_no_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, m, incy, 1e-5);
}

#[test]
fn sgemv_matches_cblas_stride() {
    let m = 64usize;
    let n = 48usize;

    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_a_colmajor_f32(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 2usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    fill_vec_strided_f32(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f32;
    let beta = 0.9f32;

    sgemv(
        Trans::NoTrans,
        m, n,
        alpha,
        &a,
        1,                 
        lda as isize,      
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_sgemv_colmajor_no_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, m, incy, 1e-5);
}

#[test]
fn sgemv_trans_matches_cblas() {
    let m = 67usize;
    let n = 53usize;

    let lda = max(m, m + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_a_colmajor_f32(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 1 + (m - 1) * incx];
    let mut y = vec![0.0f32; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, m, incx);
    fill_vec_strided_f32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f32;
    let beta = -0.75f32;

    sgemv(
        Trans::Trans,
        m, n,
        alpha,
        &a,
        1,                 
        lda as isize,      
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_sgemv_colmajor_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, n, incy, 1e-5);
}

#[test]
fn sgemv_trans_matches_cblas_stride() {
    let m = 64usize;
    let n = 48usize;

    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_a_colmajor_f32(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize; 
    let mut x = vec![0.0f32; 1 + (m - 1) * incx];
    let mut y = vec![0.0f32; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, m, incx);
    fill_vec_strided_f32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f32;
    let beta = 0.9f32;

    sgemv(
        Trans::Trans,
        m, n,
        alpha,
        &a,
        1,                 
        lda as isize,   
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_sgemv_colmajor_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, n, incy, 1e-5);
}



#[test]
fn dgemv_matches_cblas() {
    let m = 71usize;
    let n = 45usize;

    let lda = max(m, m + 3);

    let mut a = vec![0.0f64; lda * n];
    fill_a_colmajor_f64(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f64; 1 + (n - 1) * incx];
    let mut y = vec![0.0f64; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    fill_vec_strided_f64(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f64;
    let beta = -0.75f64;

    dgemv(
        Trans::NoTrans,
        m, n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_dgemv_colmajor_no_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f64(&y, &y_ref, m, incy, 1e-12);
}

#[test]
fn dgemv_matches_cblas_stride() {
    let m = 80usize;
    let n = 56usize;

    let lda = max(m, m + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_a_colmajor_f64(&mut a, m, n, lda);

    let incx = 3usize;
    let incy = 2usize;
    let mut x = vec![0.0f64; 1 + (n - 1) * incx];
    let mut y = vec![0.0f64; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    fill_vec_strided_f64(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f64;
    let beta = 0.9f64;

    dgemv(
        Trans::NoTrans,
        m, n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_dgemv_colmajor_no_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f64(&y, &y_ref, m, incy, 1e-12);
}

#[test]
fn dgemv_trans_matches_cblas() {
    let m = 71usize;
    let n = 45usize;

    let lda = max(m, m + 3);

    let mut a = vec![0.0f64; lda * n];
    fill_a_colmajor_f64(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f64; 1 + (m - 1) * incx];
    let mut y = vec![0.0f64; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_f64(&mut x, m, incx);
    fill_vec_strided_f64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f64;
    let beta = -0.75f64;

    dgemv(
        Trans::Trans,
        m, n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_dgemv_colmajor_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f64(&y, &y_ref, n, incy, 1e-12);
}

#[test]
fn dgemv_trans_matches_cblas_stride() {
    let m = 80usize;
    let n = 56usize;

    let lda = max(m, m + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_a_colmajor_f64(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x = vec![0.0f64; 1 + (m - 1) * incx];
    let mut y = vec![0.0f64; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_f64(&mut x, m, incx);
    fill_vec_strided_f64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f64;
    let beta = 0.9f64;

    dgemv(
        Trans::Trans,
        m, n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_dgemv_colmajor_trans(
            m as i32, n as i32,
            alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_f64(&y, &y_ref, n, incy, 1e-12);
}

