use std::cmp::max;

use cblas_sys::{CBLAS_LAYOUT, CBLAS_UPLO, cblas_ssymv};
use rusty_blas::level2::{enums::UpLo, ssymv::ssymv};

// pseudo random float [-0.5, 0.5] 
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

fn fill_symmetric_colmajor_f32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            let idx = j * lda + i;
            a[idx] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }

    // make symmetric 
    for j in 0..n {
        for i in 0..j {
            let v = 0.5f32 * (a[j * lda + i] + a[i * lda + j]);
            a[j * lda + i] = v;
            a[i * lda + j] = v;
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

unsafe fn cblas_ssymv_colmajor_upper(
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) { unsafe { 
    cblas_ssymv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        n,
        alpha,
        a,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    );
}} 

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

#[test]
fn ssymv_upper_lower_equivalent() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_symmetric_colmajor_f32(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y_upp = vec![0.0f32; 1 + (n - 1) * incy];
    let mut y_low = vec![0.0f32; 1 + (n - 1) * incy]; 

    fill_vec_strided_f32(&mut x, n, incx);
    fill_vec_strided_f32(&mut y_upp, n, incy);
    fill_vec_strided_f32(&mut y_low, n, incy);

    let alpha = 1.25f32;
    let beta = -0.75f32;

    // ensuring using upper or lower doesn't matter
    ssymv(
        UpLo::UpperTriangular,
        n,
        alpha,
        &a,
        1,                 // row stride
        lda as isize,      // col stride
        &x,
        incx as isize,
        beta,
        &mut y_upp,
        incy as isize,
    ); 

    ssymv(
        UpLo::LowerTriangular,
        n,
        alpha,
        &a,
        1,                 // row stride
        lda as isize,      // col stride
        &x,
        incx as isize,
        beta,
        &mut y_low,
        incy as isize,
    );

    assert_close_strided_f32(&y_upp, &y_low, n, incy, 1e-5);
}

#[test]
fn ssymv_matches_cblas() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_symmetric_colmajor_f32(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    fill_vec_strided_f32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f32;
    let beta = -0.75f32;

    ssymv(
        UpLo::UpperTriangular,
        n,
        alpha,
        &a,
        1,                 // row stride
        lda as isize,      // col stride
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_ssymv_colmajor_upper(
            n as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            x.as_ptr(),
            incx as i32,
            beta,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, n, incy, 1e-5);
}

#[test]
fn ssymv_matches_cblas_stride() {
    let n = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_symmetric_colmajor_f32(&mut a, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (n - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    fill_vec_strided_f32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f32;
    let beta = 0.9f32;

    ssymv(
        UpLo::UpperTriangular,
        n,
        alpha,
        &a,
        1,                 // row stride
        lda as isize,      // col stride
        &x,
        incx as isize,
        beta,
        &mut y,
        incy as isize,
    );

    unsafe {
        cblas_ssymv_colmajor_upper(
            n as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            x.as_ptr(),
            incx as i32,
            beta,
            y_ref.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_close_strided_f32(&y, &y_ref, n, incy, 1e-5);
}

