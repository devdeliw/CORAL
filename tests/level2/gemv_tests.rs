use std::cmp::max;

use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE,
    cblas_sgemv, 
    cblas_dgemv,
    cblas_cgemv, 
};

use rusty_blas::level2::{ 
    enums::Trans, 
    sgemv::sgemv,   
    dgemv::dgemv, 
    cgemv::cgemv, 
};

// pseudo random float [-0.5, 0.5] 
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}
fn f64_from_u32(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}
// complex single and double precision 
fn f32_pair(seed: u32) -> (f32, f32) {
    let r = f32_from_u32(seed.wrapping_mul(0x9E3779B9));
    let i = f32_from_u32(seed.rotate_left(13) ^ 0xA5A5_5A5A);
    (r, i)
}
fn f64_pair(seed: u32) -> (f64, f64) {
    let r = f64_from_u32(seed.wrapping_mul(0x9E37_79B9));
    let i = f64_from_u32(seed.rotate_left(13) ^ 0xA5A5_5A5A);
    (r, i)
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
fn fill_a_colmajor_c32(a: &mut [f32], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            let (re, im) = f32_pair(((i as u32) << 16) ^ (j as u32));
            let idx_s = 2 * (j * lda + i);
            a[idx_s]     = re;
            a[idx_s + 1] = im;
        }
    }
}
fn fill_a_colmajor_c64(a: &mut [f64], m: usize, n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..m {
            let (re, im) = f64_pair(((i as u32) << 16) ^ (j as u32));
            let idx = 2 * (j * lda + i);
            a[idx]     = re;
            a[idx + 1] = im;
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
fn fill_vec_strided_c32(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        let (re, im) = f32_pair(0x517C_C1B7u32.wrapping_mul(k as u32).wrapping_add(777));
        let s = 2 * pos;
        buf[s]     = re;
        buf[s + 1] = im;
        pos += inc;
    }
}
fn fill_vec_strided_c64(buf: &mut [f64], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        let (re, im) = f64_pair(0x517C_C1B7u32.wrapping_mul(k as u32).wrapping_add(1337));
        let s = 2 * pos;
        buf[s]     = re;
        buf[s + 1] = im;
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
fn assert_close_strided_c32(y_test: &[f32], y_ref: &[f32], n: usize, incy: usize, tol: f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut p = 0usize;
    for _ in 0..n {
        let s = 2 * p;
        let a_re = y_test[s];
        let a_im = y_test[s + 1];
        let b_re = y_ref[s];
        let b_im = y_ref[s + 1];

        let abs_re = (a_re - b_re).abs();
        let abs_im = (a_im - b_im).abs();
        let rel_re = abs_re / 1.0f32.max(b_re.abs());
        let rel_im = abs_im / 1.0f32.max(b_im.abs());

        max_abs = max_abs.max(abs_re.max(abs_im));
        max_rel = max_rel.max(rel_re.max(rel_im));

        assert!(
            rel_re <= tol && rel_im <= tol,
            "mismatch at complex idx {} (buf base {}): test=({},{}) ref=({},{}) |abs|=({}, {}) rel=({}, {})",
            p, s, a_re, a_im, b_re, b_im, abs_re, abs_im, rel_re, rel_im
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {}, max_rel_diff = {}", max_abs, max_rel);
}
fn assert_close_strided_c64(y_test: &[f64], y_ref: &[f64], n: usize, incy: usize, tol: f64) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut p = 0usize;
    for _ in 0..n {
        let s = 2 * p;
        let a_re = y_test[s];
        let a_im = y_test[s + 1];
        let b_re = y_ref[s];
        let b_im = y_ref[s + 1];

        let abs_re = (a_re - b_re).abs();
        let abs_im = (a_im - b_im).abs();
        let rel_re = abs_re / 1.0f64.max(b_re.abs());
        let rel_im = abs_im / 1.0f64.max(b_im.abs());

        max_abs = max_abs.max(abs_re.max(abs_im));
        max_rel = max_rel.max(rel_re.max(rel_im));

        assert!(
            rel_re <= tol && rel_im <= tol,
            "mismatch@{} base {}: test=({:.3e},{:.3e}) ref=({:.3e},{:.3e}) |abs|=({:.3e},{:.3e}) rel=({:.3e},{:.3e})",
            p, s, a_re, a_im, b_re, b_im, abs_re, abs_im, rel_re, rel_im
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {:.3e}, max_rel_diff = {:.3e}", max_abs, max_rel);
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

unsafe fn cblas_cgemv_colmajor_no_trans(
    m: i32, n: i32,
    alpha: &[f32; 2],
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: &[f32; 2],
    y: *mut f32, incy: i32,
) { unsafe {
    cblas_cgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f32; 2]>(), lda,
        x.cast::<[f32; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f32; 2]>(), incy,
    );
}}

unsafe fn cblas_cgemv_colmajor_trans(
    m: i32, n: i32,
    alpha: &[f32; 2],
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: &[f32; 2],
    y: *mut f32, incy: i32,
) { unsafe {
    cblas_cgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f32; 2]>(), lda,
        x.cast::<[f32; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f32; 2]>(), incy,
    );
}}

unsafe fn cblas_cgemv_colmajor_conjtrans(
    m: i32, n: i32,
    alpha: &[f32; 2],
    a: *const f32, lda: i32,
    x: *const f32, incx: i32,
    beta: &[f32; 2],
    y: *mut f32, incy: i32,
) { unsafe {
    cblas_cgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasConjTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f32; 2]>(), lda,
        x.cast::<[f32; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f32; 2]>(), incy,
    );
}}
unsafe fn cblas_zgemv_colmajor_no_trans(
    m: i32, n: i32,
    alpha: &[f64; 2],
    a: *const f64, lda: i32,
    x: *const f64, incx: i32,
    beta: &[f64; 2],
    y: *mut f64, incy: i32,
) { unsafe {
    cblas_sys::cblas_zgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f64; 2]>(), lda,
        x.cast::<[f64; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f64; 2]>(), incy,
    );
}}

unsafe fn cblas_zgemv_colmajor_trans(
    m: i32, n: i32,
    alpha: &[f64; 2],
    a: *const f64, lda: i32,
    x: *const f64, incx: i32,
    beta: &[f64; 2],
    y: *mut f64, incy: i32,
) { unsafe {
    cblas_sys::cblas_zgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f64; 2]>(), lda,
        x.cast::<[f64; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f64; 2]>(), incy,
    );
}}

unsafe fn cblas_zgemv_colmajor_conjtrans(
    m: i32, n: i32,
    alpha: &[f64; 2],
    a: *const f64, lda: i32,
    x: *const f64, incx: i32,
    beta: &[f64; 2],
    y: *mut f64, incy: i32,
) { unsafe {
    cblas_sys::cblas_zgemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_TRANSPOSE::CblasConjTrans,
        m, n,
        alpha.as_ptr() as *const _,
        a.cast::<[f64; 2]>(), lda,
        x.cast::<[f64; 2]>(), incx,
        beta.as_ptr() as *const _,
        y.cast::<[f64; 2]>(), incy,
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

#[test]
fn cgemv_matches_cblas_notrans() {
    let m = 63usize;
    let n = 49usize;
    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (m - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    fill_vec_strided_c32(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 1.10f32, -0.75f32 ];
    let beta  = [ -0.40f32,  0.25f32 ];

    cgemv(
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
        cblas_cgemv_colmajor_no_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, m, incy, 1e-4);
}

#[test]
fn cgemv_matches_cblas_notrans_stride() {
    let m = 70usize;
    let n = 52usize;
    let lda = max(m, m + 9);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x     = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (m - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    fill_vec_strided_c32(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.65f32, 0.35f32 ];
    let beta  = [  0.80f32, 0.10f32 ];

    cgemv(
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
        cblas_cgemv_colmajor_no_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, m, incy, 1e-4);
}

#[test]
fn cgemv_trans_matches_cblas() {
    let m = 63usize;
    let n = 49usize;
    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f32; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, m, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 1.10f32, -0.75f32 ];
    let beta  = [ -0.40f32,  0.25f32 ];

    cgemv(
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
        cblas_cgemv_colmajor_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-4);
}

#[test]
fn cgemv_trans_matches_cblas_stride() {
    let m = 70usize;
    let n = 52usize;
    let lda = max(m, m + 9);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 3usize;
    let incy = 2usize;
    let mut x     = vec![0.0f32; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, m, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.65f32, 0.35f32 ];
    let beta  = [  0.80f32, 0.10f32 ];

    cgemv(
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
        cblas_cgemv_colmajor_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-4);
}

#[test]
fn cgemv_conjtrans_matches_cblas() {
    let m = 63usize;
    let n = 49usize;
    let lda = max(m, m + 7);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f32; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, m, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 0.75f32,  0.30f32 ];
    let beta  = [ -0.20f32, 0.10f32 ];

    cgemv(
        Trans::ConjTrans,
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
        cblas_cgemv_colmajor_conjtrans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-4);
}

#[test]
fn cgemv_conjtrans_matches_cblas_stride() {
    let m = 70usize;
    let n = 52usize;
    let lda = max(m, m + 9);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_a_colmajor_c32(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x     = vec![0.0f32; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, m, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.65f32, 0.35f32 ];
    let beta  = [  0.80f32, 0.10f32 ];

    cgemv(
        Trans::ConjTrans,
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
        cblas_cgemv_colmajor_conjtrans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-4);
}

#[test]
fn zgemv_matches_cblas_notrans() {
    let m = 66usize;
    let n = 50usize;
    let lda = max(m, m + 6);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (m - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    fill_vec_strided_c64(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 1.10f64, -0.45f64 ];
    let beta  = [ -0.30f64,  0.20f64 ];

    rusty_blas::level2::zgemv::zgemv(
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
        cblas_zgemv_colmajor_no_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, m, incy, 1e-12);
}

#[test]
fn zgemv_matches_cblas_notrans_stride() {
    let m = 72usize;
    let n = 58usize;
    let lda = max(m, m + 4);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x     = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (m - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    fill_vec_strided_c64(&mut y, m, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.65f64, 0.35f64 ];
    let beta  = [  0.80f64, 0.10f64 ];

    rusty_blas::level2::zgemv::zgemv(
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
        cblas_zgemv_colmajor_no_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, m, incy, 1e-12);
}

#[test]
fn zgemv_trans_matches_cblas() {
    let m = 66usize;
    let n = 50usize;
    let lda = max(m, m + 6);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f64; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, m, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 0.90f64, -0.30f64 ];
    let beta  = [ -0.25f64,  0.15f64 ];

    rusty_blas::level2::zgemv::zgemv(
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
        cblas_zgemv_colmajor_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}

#[test]
fn zgemv_trans_matches_cblas_stride() {
    let m = 72usize;
    let n = 58usize;
    let lda = max(m, m + 4);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 3usize;
    let incy = 2usize;
    let mut x     = vec![0.0f64; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, m, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.75f64, 0.50f64 ];
    let beta  = [  0.10f64, 0.05f64 ];

    rusty_blas::level2::zgemv::zgemv(
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
        cblas_zgemv_colmajor_trans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}

#[test]
fn zgemv_conjtrans_matches_cblas() {
    let m = 66usize;
    let n = 50usize;
    let lda = max(m, m + 6);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x     = vec![0.0f64; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, m, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ 0.70f64,  0.25f64 ];
    let beta  = [ -0.20f64, 0.10f64 ];

    rusty_blas::level2::zgemv::zgemv(
        Trans::ConjTrans,
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
        cblas_zgemv_colmajor_conjtrans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}

#[test]
fn zgemv_conjtrans_matches_cblas_stride() {
    let m = 72usize;
    let n = 58usize;
    let lda = max(m, m + 4);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_a_colmajor_c64(&mut a, m, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x     = vec![0.0f64; 2 * (1 + (m - 1) * incx)];
    let mut y     = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, m, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [ -0.55f64, 0.20f64 ];
    let beta  = [  0.85f64, 0.05f64 ];

    rusty_blas::level2::zgemv::zgemv(
        Trans::ConjTrans,
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
        cblas_zgemv_colmajor_conjtrans(
            m as i32, n as i32,
            &alpha,
            a.as_ptr(), lda as i32,
            x.as_ptr(), incx as i32,
            &beta,
            y_ref.as_mut_ptr(), incy as i32,
        );
    }

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}
