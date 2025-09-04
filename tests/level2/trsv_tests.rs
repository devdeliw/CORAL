use std::cmp::max;
use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_UPLO, 
    CBLAS_TRANSPOSE, 
    CBLAS_DIAG,
    cblas_strsv,
};
use rusty_blas::level2::{
    strusv::strusv, 
    strlsv::strlsv,
};


// pseudorandom floats in [-0.5, 0.5] 
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

fn fill_upper_colmajor(a: &mut [f32], n: usize, lda: usize, zero_lower: bool) {
    for j in 0..n {
        for i in 0..n {
            let idx = j * lda + i;
            a[idx] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    if zero_lower {
        for j in 0..n {
            for i in (j+1)..n { 
                a[j * lda + i] = 0.0;
            }
        }
    }
}

fn fill_lower_colmajor(a: &mut [f32], n: usize, lda: usize, zero_upper: bool) {
    for j in 0..n {
        for i in 0..n {
            let idx = j * lda + i;
            a[idx] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }

    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    if zero_upper {
        for j in 0..n {
            for i in 0..j {
                a[j * lda + i] = 0.0;
            }
        }
    }
}

fn assert_close(y_test: &[f32], y_ref: &[f32], n: usize, inc: usize, tol: f32) {
    let mut p = 0usize;
    for k in 0..n {
        let a = y_test[p];
        let b = y_ref[p];
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(1.0);
        assert!(
            rel <= tol,
            "mismatch at logical idx {} (buf idx {}): test={} ref={} |abs|={} rel={}",
            k, p, a, b, abs, rel
        );
        p += inc;
    }
}

unsafe fn cblas_strsv_upper_notrans_nonunit(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe { 
    cblas_strsv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda,
        x, incx,
    );
}}

unsafe fn cblas_strsv_lower_notrans_nonunit(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strsv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda,
        x, incx,
    );
}}

// pseudorandom floats, but strided via inc 
fn fill_vec_strided(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f32_from_u32(
            0x9E3779B9u32.wrapping_mul(k as u32)
                .wrapping_add(12345));
        pos += inc;
    }
}


#[test]
fn strusv_matches_cblas() {
    let n = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strusv(
        n,
        false, 
        &a,
        1,               
        lda as isize,    
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strsv_upper_notrans_nonunit(
            n as i32,
            a.as_ptr(), 
            lda as i32,
            x_ref.as_mut_ptr(), 
            incx as i32,
        );
    }

    assert_close(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strusv_matches_cblas_stride() {
    let n = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strusv(
        n,
        false,
        &a,
        1,               
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strsv_upper_notrans_nonunit(
            n as i32,
            a.as_ptr(), 
            lda as i32,
            x_ref.as_mut_ptr(), 
            incx as i32,
        );
    }

    assert_close(&x, &x_ref, n, incx, 1e-5);
}
 
#[test]
fn strlsv_matches_cblas() {
    let n = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strlsv(
        n,
        false,
        &a,
        1,                
        lda as isize,     
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strsv_lower_notrans_nonunit(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strlsv_matches_cblas_stride() {
    let n = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strlsv(
        n,
        false,
        &a,
        1,                
        lda as isize,   
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strsv_lower_notrans_nonunit(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close(&x, &x_ref, n, incx, 1e-5);
}
