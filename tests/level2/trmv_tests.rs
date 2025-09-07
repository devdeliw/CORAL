use std::cmp::max;
use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_UPLO,
    CBLAS_TRANSPOSE,
    CBLAS_DIAG,
    cblas_strmv,
    cblas_dtrmv,
};
use rusty_blas::level2::{
    enums::{ Trans, Diag, UpLo },
    strmv::strmv,
    dtrmv::dtrmv,
};

// pseudorandom floats in [-0.5, 0.5]
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

// pseudorandom doubles in [-0.5, 0.5]
fn f64_from_u32(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}

fn fill_upper_colmajor_f32(a: &mut [f32], n: usize, lda: usize, zero_lower: bool) {
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
            for i in (j + 1)..n {
                a[j * lda + i] = 0.0;
            }
        }
    }
}

fn fill_lower_colmajor_f32(a: &mut [f32], n: usize, lda: usize, zero_upper: bool) {
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

fn fill_upper_colmajor_f64(a: &mut [f64], n: usize, lda: usize, zero_lower: bool) {
    for j in 0..n {
        for i in 0..n {
            let idx = j * lda + i;
            a[idx] = f64_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    if zero_lower {
        for j in 0..n {
            for i in (j + 1)..n {
                a[j * lda + i] = 0.0;
            }
        }
    }
}

fn fill_lower_colmajor_f64(a: &mut [f64], n: usize, lda: usize, zero_upper: bool) {
    for j in 0..n {
        for i in 0..n {
            let idx = j * lda + i;
            a[idx] = f64_from_u32(((i as u32) << 16) ^ (j as u32));
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

fn assert_close_f32(y_test: &[f32], y_ref: &[f32], n: usize, inc: usize, tol: f32) {
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

fn assert_close_f64(y_test: &[f64], y_ref: &[f64], n: usize, inc: usize, tol: f64) {
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

// pseudorandom floats, but strided via inc
fn fill_vec_strided_f32(buf: &mut [f32], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f32_from_u32(
            0x9E3779B9u32
                .wrapping_mul(k as u32)
                .wrapping_add(12345),
        );
        pos += inc;
    }
}

// pseudorandom doubles, but strided via inc
fn fill_vec_strided_f64(buf: &mut [f64], n: usize, inc: usize) {
    let mut pos = 0usize;
    for k in 0..n {
        buf[pos] = f64_from_u32(
            0x9E3779B9u32
                .wrapping_mul(k as u32)
                .wrapping_add(12345),
        );
        pos += inc;
    }
}


unsafe fn cblas_strmv_upper_notrans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_strmv_upper_trans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_strmv_lower_notrans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_strmv_lower_trans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}


unsafe fn cblas_dtrmv_upper_notrans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_dtrmv_upper_trans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_dtrmv_lower_notrans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}

unsafe fn cblas_dtrmv_lower_trans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda, x, incx,
    );
}}


#[test]
fn strumv_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor_f32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_upper_notrans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strumv_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor_f32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::UpperTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_upper_trans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strumv_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor_f32(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_upper_notrans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strumv_trans_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_upper_colmajor_f32(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::UpperTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_upper_trans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}


#[test]
fn strlmv_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor_f32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_lower_notrans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strlmv_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor_f32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::LowerTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_lower_trans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strlmv_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor_f32(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_lower_notrans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn strlmv_trans_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f32; lda * n];
    fill_lower_colmajor_f32(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f32; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_f32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    strmv(
        UpLo::LowerTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_strmv_lower_trans_nonunit_f32(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f32(&x, &x_ref, n, incx, 1e-5);
}


#[test]
fn dtrumv_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_upper_colmajor_f64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_upper_notrans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrumv_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_upper_colmajor_f64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::UpperTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_upper_trans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrumv_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f64; lda * n];
    fill_upper_colmajor_f64(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_upper_notrans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrumv_trans_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f64; lda * n];
    fill_upper_colmajor_f64(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::UpperTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_upper_trans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}


#[test]
fn dtrlmv_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_lower_colmajor_f64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_lower_notrans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrlmv_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; lda * n];
    fill_lower_colmajor_f64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::LowerTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_lower_trans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrlmv_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f64; lda * n];
    fill_lower_colmajor_f64(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_lower_notrans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn dtrlmv_trans_matches_cblas_stride() {
    let n   = 64usize;
    let lda = max(n, n + 7);

    let mut a = vec![0.0f64; lda * n];
    fill_lower_colmajor_f64(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f64; 1 + (n - 1) * incx];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_f64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    dtrmv(
        UpLo::LowerTriangular,
        Trans::Trans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_dtrmv_lower_trans_nonunit_f64(
            n as i32, a.as_ptr(), lda as i32, x_ref.as_mut_ptr(), incx as i32
        );
    }

    assert_close_f64(&x, &x_ref, n, incx, 1e-12);
}

