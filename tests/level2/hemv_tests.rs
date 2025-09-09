use std::cmp::max;

use cblas_sys::{CBLAS_LAYOUT, CBLAS_UPLO, cblas_chemv, cblas_zhemv};
use rusty_blas::level2::{enums::UpLo, chemv::chemv, zhemv::zhemv};

fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

fn f64_from_u32(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}

fn fill_hermitian_colmajor_c32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (j * lda + i);
            let seed_r = ((i as u32) << 16) ^ (j as u32);
            let seed_i = ((j as u32) << 16) ^ (i as u32) ^ 0xA5A5;
            a[idx + 0] = f32_from_u32(seed_r);
            a[idx + 1] = f32_from_u32(seed_i);
        }
    }
    for j in 0..n {
        let d = 2 * (j * lda + j);
        a[d + 1] = 0.0;
        for i in 0..j {
            let ij = 2 * (j * lda + i);
            let ji = 2 * (i * lda + j);
            let ar = 0.5 * (a[ij + 0] + a[ji + 0]);
            let ai = 0.5 * (a[ij + 1] - a[ji + 1]);
            a[ij + 0] = ar;
            a[ij + 1] = ai;
            a[ji + 0] = ar;
            a[ji + 1] = -ai;
        }
    }
}

fn fill_hermitian_colmajor_c64(a: &mut [f64], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (j * lda + i);
            let seed_r = ((i as u32) << 16) ^ (j as u32);
            let seed_i = ((j as u32) << 16) ^ (i as u32) ^ 0x5A5A;
            a[idx + 0] = f64_from_u32(seed_r);
            a[idx + 1] = f64_from_u32(seed_i);
        }
    }
    for j in 0..n {
        let d = 2 * (j * lda + j);
        a[d + 1] = 0.0;
        for i in 0..j {
            let ij = 2 * (j * lda + i);
            let ji = 2 * (i * lda + j);
            let ar = 0.5 * (a[ij + 0] + a[ji + 0]);
            let ai = 0.5 * (a[ij + 1] - a[ji + 1]);
            a[ij + 0] = ar;
            a[ij + 1] = ai;
            a[ji + 0] = ar;
            a[ji + 1] = -ai;
        }
    }
}

fn fill_vec_strided_c32(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        let re = f32_from_u32(0x9E37_79B9u32.wrapping_mul(2 * k as u32).wrapping_add(12345));
        let im = f32_from_u32(0x85EB_CA6Bu32.wrapping_mul(2 * k as u32 + 1).wrapping_add(54321));
        let p = 2 * pos;
        buf[p + 0] = re;
        buf[p + 1] = im;
        pos += inc;
    }
}

fn fill_vec_strided_c64(buf: &mut [f64], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        let re = f64_from_u32(0x9E37_79B9u32.wrapping_mul(2 * k as u32).wrapping_add(11111));
        let im = f64_from_u32(0x85EB_CA6Bu32.wrapping_mul(2 * k as u32 + 1).wrapping_add(22222));
        let p = 2 * pos;
        buf[p + 0] = re;
        buf[p + 1] = im;
        pos += inc;
    }
}

unsafe fn cblas_chemv_colmajor_upper(
    n: i32,
    alpha: [f32; 2],
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: [f32; 2],
    y: *mut f32,
    incy: i32,
) { unsafe { 
    cblas_chemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        n,
        &alpha as *const [f32; 2],
        a.cast::<[f32; 2]>(),
        lda,
        x.cast::<[f32; 2]>(),
        incx,
        &beta as *const [f32; 2],
        y.cast::<[f32; 2]>(),
        incy,
    );
}}

unsafe fn cblas_zhemv_colmajor_upper(
    n: i32,
    alpha: [f64; 2],
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: [f64; 2],
    y: *mut f64,
    incy: i32,
) { unsafe { 
    cblas_zhemv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        n,
        &alpha as *const [f64; 2],
        a.cast::<[f64; 2]>(),
        lda,
        x.cast::<[f64; 2]>(),
        incx,
        &beta as *const [f64; 2],
        y.cast::<[f64; 2]>(),
        incy,
    );
}}

fn assert_close_strided_c32(
    y_test: &[f32], 
    y_ref: &[f32], 
    m: usize, 
    incy: usize, 
    tol: f32
) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut p = 0usize;
    for _ in 0..m {
        let i = 2 * p;
        let ar = y_test[i + 0];
        let ai = y_test[i + 1];
        let br = y_ref[i + 0];
        let bi = y_ref[i + 1];
        let dr = ar - br;
        let di = ai - bi;
        let abs = (dr * dr + di * di).sqrt();
        let denom = 1.0f32.max((br * br + bi * bi).sqrt());
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tol,
            "mismatch at complex pos {}: test=({},{}) ref=({},{}) |abs|={} rel={}",
            p, ar, ai, br, bi, abs, rel
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {}, max_rel_diff = {}", max_abs, max_rel);
}

fn assert_close_strided_c64(
    y_test: &[f64],
    y_ref: &[f64],
    m: usize, 
    incy: usize, 
    tol: f64
) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut p = 0usize;
    for _ in 0..m {
        let i = 2 * p;
        let ar = y_test[i + 0];
        let ai = y_test[i + 1];
        let br = y_ref[i + 0];
        let bi = y_ref[i + 1];
        let dr = ar - br;
        let di = ai - bi;
        let abs = (dr * dr + di * di).sqrt();
        let denom = 1.0f64.max((br * br + bi * bi).sqrt());
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tol,
            "mismatch at complex pos {}: test=({},{}) ref=({},{}) |abs|={} rel={}",
            p, ar, ai, br, bi, abs, rel
        );
        p += incy;
    }
    eprintln!("max_abs_diff = {}, max_rel_diff = {}", max_abs, max_rel);
}

#[test]
fn chemv_upper_lower_equivalent() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_hermitian_colmajor_c32(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut y_upp = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_low = vec![0.0f32; 2 * (1 + (n - 1) * incy)];

    fill_vec_strided_c32(&mut x, n, incx);
    fill_vec_strided_c32(&mut y_upp, n, incy);
    fill_vec_strided_c32(&mut y_low, n, incy);

    let alpha = [1.25f32, -0.35f32];
    let beta  = [-0.75f32, 0.40f32];

    chemv(
        UpLo::UpperTriangular,
        n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y_upp,
        incy as isize,
    );

    chemv(
        UpLo::LowerTriangular,
        n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y_low,
        incy as isize,
    );

    assert_close_strided_c32(&y_upp, &y_low, n, incy, 1e-5);
}

#[test]
fn chemv_matches_cblas() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_hermitian_colmajor_c32(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut y = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [1.25f32, -0.35f32];
    let beta  = [-0.75f32, 0.40f32];

    chemv(
        UpLo::UpperTriangular,
        n,
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
        cblas_chemv_colmajor_upper(
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

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-5);
}

#[test]
fn chemv_matches_cblas_stride() {
    let n = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_hermitian_colmajor_c32(&mut a, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut y = vec![0.0f32; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f32; y.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    fill_vec_strided_c32(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [-0.6f32, 0.2f32];
    let beta  = [0.9f32, -0.1f32];

    chemv(
        UpLo::UpperTriangular,
        n,
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
        cblas_chemv_colmajor_upper(
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

    assert_close_strided_c32(&y, &y_ref, n, incy, 1e-5);
}

#[test]
fn zhemv_upper_lower_equivalent() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_hermitian_colmajor_c64(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut y_upp = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_low = vec![0.0f64; 2 * (1 + (n - 1) * incy)];

    fill_vec_strided_c64(&mut x, n, incx);
    fill_vec_strided_c64(&mut y_upp, n, incy);
    fill_vec_strided_c64(&mut y_low, n, incy);

    let alpha = [1.25f64, -0.35f64];
    let beta  = [-0.75f64, 0.40f64];

    zhemv(
        UpLo::UpperTriangular,
        n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y_upp,
        incy as isize,
    );

    zhemv(
        UpLo::LowerTriangular,
        n,
        alpha,
        &a,
        1,
        lda as isize,
        &x,
        incx as isize,
        beta,
        &mut y_low,
        incy as isize,
    );

    assert_close_strided_c64(&y_upp, &y_low, n, incy, 1e-12);
}

#[test]
fn zhemv_matches_cblas() {
    let n = 67usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_hermitian_colmajor_c64(&mut a, n, lda);

    let incx = 1usize;
    let incy = 1usize;
    let mut x = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut y = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [1.25f64, -0.35f64];
    let beta  = [-0.75f64, 0.40f64];

    zhemv(
        UpLo::UpperTriangular,
        n,
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
        cblas_zhemv_colmajor_upper(
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

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}

#[test]
fn zhemv_matches_cblas_stride() {
    let n = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_hermitian_colmajor_c64(&mut a, n, lda);

    let incx = 2usize;
    let incy = 3usize;
    let mut x = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut y = vec![0.0f64; 2 * (1 + (n - 1) * incy)];
    let mut y_ref = vec![0.0f64; y.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    fill_vec_strided_c64(&mut y, n, incy);
    y_ref.copy_from_slice(&y);

    let alpha = [-0.6f64, 0.2f64];
    let beta  = [0.9f64, -0.1f64];

    zhemv(
        UpLo::UpperTriangular,
        n,
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
        cblas_zhemv_colmajor_upper(
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

    assert_close_strided_c64(&y, &y_ref, n, incy, 1e-12);
}

