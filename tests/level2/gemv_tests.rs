use std::cmp::max;

use cblas_sys::{
    cblas_sgemv, CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans,
};

use rusty_blas::level2::sgemv::sgemv;   


fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

fn fill_a_colmajor(a: &mut [f32], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            let idx = j * lda + i;
            a[idx] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
}

fn fill_vec_strided(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f32_from_u32(0x9E3779B9u32.wrapping_mul(k as u32).wrapping_add(12345));
        pos += inc;
    }
}

fn assert_close_strided(y_test: &[f32], y_ref: &[f32], m: usize, incy: usize, tol: f32) {
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


unsafe fn cblas_sgemv_colmajor_no_trans(
    m: i32, n: i32,
    alpha: f32,
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: f32,
    y: *mut f32, incy: i32,
) { unsafe { 
    cblas_sgemv(
        CblasColMajor,
        CblasNoTrans,
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
    fill_a_colmajor(&mut a, m, n, lda);

    // x, y with unit stride
    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided(&mut x, n, incx);
    fill_vec_strided(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = 1.25f32;
    let beta = -0.75f32;

    sgemv(
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

    assert_close_strided(&y, &y_ref, m, incy, 1e-5);
}


#[test]
fn sgemv_matches_cblas_stride() {
    // Problem size
    let m = 64usize;
    let n = 48usize;

    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_a_colmajor(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 2usize;
    let mut x = vec![0.0f32; 1 + (n - 1) * incx];
    let mut y = vec![0.0f32; 1 + (m - 1) * incy];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided(&mut x, n, incx);
    fill_vec_strided(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = -0.6f32;
    let beta = 0.9f32;

    sgemv(
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

    // Call CBLAS (reference)
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

    // Compare only the strided positions
    assert_close_strided(&y, &y_ref, m, incy, 1e-5);
}

