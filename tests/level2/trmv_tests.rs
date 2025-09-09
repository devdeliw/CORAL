use std::cmp::max;
use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_UPLO,
    CBLAS_TRANSPOSE,
    CBLAS_DIAG,
    cblas_strmv,
    cblas_dtrmv,
    cblas_ctrmv,
    cblas_ztrmv
};
use rusty_blas::level2::{
    enums::{ Trans, Diag, UpLo },
    strmv::strmv,
    dtrmv::dtrmv,
    ctrmv::ctrmv,
    ztrmv::ztrmv, 
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

// complex [-0.5, 0.5] 
fn c32_from_u32(
    u: u32,
) -> (f32, f32) {
    let r = f32_from_u32(u ^ 0xA5A5_5A5A);
    let i = f32_from_u32(u.rotate_left(13) ^ 0x3C6E_F35F);
    (r, i)
}
fn c64_from_u32(
    u: u32,
) -> (f64, f64) {
    let r = f64_from_u32(u ^ 0xA5A5_5A5A);
    let i = f64_from_u32(u.rotate_left(13) ^ 0x3C6E_F35F);
    (r, i)
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
fn fill_upper_colmajor_c32(
    a          : &mut [f32], // interleaved
    n          : usize,
    lda        : usize,      // complex lda
    zero_lower : bool,
) {
    for j in 0..n {
        for i in 0..n {
            let idx       = j * lda + i;
            let (r, im)   = c32_from_u32(((i as u32) << 16) ^ (j as u32));
            a[2 * idx    ] = r;
            a[2 * idx + 1] = im;
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[2 * idx] += 1.0_f32.abs(); // keep imag as-is
    }
    if zero_lower {
        for j in 0..n {
            for i in (j + 1)..n {
                let idx       = j * lda + i;
                a[2 * idx    ] = 0.0;
                a[2 * idx + 1] = 0.0;
            }
        }
    }
}
fn fill_upper_colmajor_c64(
    a          : &mut [f64], // interleaved
    n          : usize,
    lda        : usize,      // complex lda
    zero_lower : bool,
) {
    for j in 0..n {
        for i in 0..n {
            let idx       = j * lda + i;
            let (r, im)   = c64_from_u32(((i as u32) << 16) ^ (j as u32));
            a[2 * idx    ] = r;
            a[2 * idx + 1] = im;
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[2 * idx] += 1.0_f64.abs();
    }
    if zero_lower {
        for j in 0..n {
            for i in (j + 1)..n {
                let idx       = j * lda + i;
                a[2 * idx    ] = 0.0;
                a[2 * idx + 1] = 0.0;
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
fn fill_vec_strided_c32(
    buf : &mut [f32], // interleaved
    n   : usize,
    inc : usize,      // complex inc
) {
    let mut pos = 0usize;
    for k in 0..n {
        let (r, i) = c32_from_u32(
            0x9E37_79B9u32
                .wrapping_mul(k as u32)
                .wrapping_add(12345),
        );
        buf[2 * pos    ] = r;
        buf[2 * pos + 1] = i;
        pos += inc;
    }
}
fn fill_vec_strided_c64(
    buf : &mut [f64], // interleaved
    n   : usize,
    inc : usize,      // complex inc
) {
    let mut pos = 0usize;
    for k in 0..n {
        let (r, i) = c64_from_u32(
            0x9E37_79B9u32
                .wrapping_mul(k as u32)
                .wrapping_add(12345),
        );
        buf[2 * pos    ] = r;
        buf[2 * pos + 1] = i;
        pos += inc;
    }
}

fn assert_close_c32(
    y_test : &[f32], // interleaved
    y_ref  : &[f32], // interleaved
    n      : usize,
    inc    : usize,  // complex inc
    tol    : f32,
) {
    let mut p = 0usize;
    for k in 0..n {
        let ar = y_test[2 * p];
        let ai = y_test[2 * p + 1];
        let br = y_ref [2 * p];
        let bi = y_ref [2 * p + 1];

        let abs_r = (ar - br).abs();
        let abs_i = (ai - bi).abs();

        let rel_r = abs_r / br.abs().max(1.0);
        let rel_i = abs_i / bi.abs().max(1.0);

        assert!(
            rel_r <= tol && rel_i <= tol,
            "mismatch at logical {} (buf idx {}): test=({},{}) ref=({},{}) |abs|=({},{}) rel=({},{})",
            k, p, ar, ai, br, bi, abs_r, abs_i, rel_r, rel_i
        );
        p += inc;
    }
}

fn fill_diag_only_colmajor_c32(
    a   : &mut [f32], // interleaved
    n   : usize,
    lda : usize,      // complex lda
) {
    for j in 0..n {
        for i in 0..n {
            let idx       = j * lda + i;
            a[2 * idx    ] = 0.0;
            a[2 * idx + 1] = 0.0;
        }
    }
    for i in 0..n {
        let idx       = i * lda + i;
        let (r, im)   = c32_from_u32((i as u32).wrapping_mul(2654435761));
        a[2 * idx    ] = 1.0 + r.abs();
        a[2 * idx + 1] = im;
    }
}
fn fill_diag_only_colmajor_c64(
    a   : &mut [f64], // interleaved
    n   : usize,
    lda : usize,      // complex lda
) {
    for j in 0..n {
        for i in 0..n {
            let idx       = j * lda + i;
            a[2 * idx    ] = 0.0;
            a[2 * idx + 1] = 0.0;
        }
    }
    for i in 0..n {
        let idx       = i * lda + i;
        let (r, im)   = c64_from_u32((i as u32).wrapping_mul(2654435761));
        a[2 * idx    ] = 1.0 + r.abs();
        a[2 * idx + 1] = im;
    }
}

fn assert_close_c64(
    y_test : &[f64], // interleaved
    y_ref  : &[f64], // interleaved
    n      : usize,
    inc    : usize,  // complex inc
    tol    : f64,
) {
    let mut p = 0usize;
    for k in 0..n {
        let ar = y_test[2 * p];
        let ai = y_test[2 * p + 1];
        let br = y_ref [2 * p];
        let bi = y_ref [2 * p + 1];

        let abs_r = (ar - br).abs();
        let abs_i = (ai - bi).abs();

        let rel_r = abs_r / br.abs().max(1.0);
        let rel_i = abs_i / bi.abs().max(1.0);

        assert!(
            rel_r <= tol && rel_i <= tol,
            "mismatch at logical {} (buf idx {}): test=({},{}) ref=({},{}) |abs|=({},{}) rel=({},{})",
            k, p, ar, ai, br, bi, abs_r, abs_i, rel_r, rel_i
        );
        p += inc;
    }
}



unsafe fn cblas_ctrmv_upper_conjtrans_nonunit_c32(
    n   : i32,
    a   : *const f32,
    lda : i32,
    x   : *mut f32,
    incx: i32,
) { unsafe {
    cblas_ctrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f32; 2]>(),
        lda,
        x.cast::<[f32; 2]>(),
        incx,
    );
}}




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


unsafe fn cblas_ctrmv_upper_notrans_nonunit_c32(
    n   : i32,
    a   : *const f32, // interleaved
    lda : i32,        // complex
    x   : *mut f32,   // interleaved
    incx: i32,        // complex
) { unsafe {
    cblas_ctrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f32; 2]>(),
        lda,
        x.cast::<[f32; 2]>(),
        incx,
    );
}}

unsafe fn cblas_ctrmv_upper_trans_nonunit_c32(
    n   : i32,
    a   : *const f32,
    lda : i32,
    x   : *mut f32,
    incx: i32,
) { unsafe {
    cblas_ctrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f32; 2]>(),
        lda,
        x.cast::<[f32; 2]>(),
        incx,
    );
}}

unsafe fn cblas_ztrmv_upper_notrans_nonunit_c64(
    n   : i32,
    a   : *const f64, // interleaved
    lda : i32,        // complex
    x   : *mut f64,   // interleaved
    incx: i32,        // complex
) { unsafe {
    cblas_ztrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f64; 2]>(),
        lda,
        x.cast::<[f64; 2]>(),
        incx,
    );
}}

unsafe fn cblas_ztrmv_upper_trans_nonunit_c64(
    n   : i32,
    a   : *const f64,
    lda : i32,
    x   : *mut f64,
    incx: i32,
) { unsafe {
    cblas_ztrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f64; 2]>(),
        lda,
        x.cast::<[f64; 2]>(),
        incx,
    );
}}

unsafe fn cblas_ztrmv_upper_conjtrans_nonunit_c64(
    n   : i32,
    a   : *const f64,
    lda : i32,
    x   : *mut f64,
    incx: i32,
) { unsafe {
    cblas_ztrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasConjTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a.cast::<[f64; 2]>(),
        lda,
        x.cast::<[f64; 2]>(),
        incx,
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

#[test]
fn ctrmv_upper_lower_equivalent() {
    let n   = 64usize;
    let lda = max(n, n + 3);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_diag_only_colmajor_c32(&mut a, n, lda);

    let incx = 1usize;
    let mut x_upper = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut x_lower = vec![0.0f32; x_upper.len()];

    fill_vec_strided_c32(&mut x_upper, n, incx);
    x_lower.copy_from_slice(&x_upper);

    ctrmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x_upper,
        incx as isize,
    );

    ctrmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x_lower,
        incx as isize,
    );

    assert_close_c32(&x_upper, &x_lower, n, incx, 1e-5);
}

#[test]
fn ctrmv_upper_matches_cblas_notrans() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_upper_colmajor_c32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ctrmv(
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
        cblas_ctrmv_upper_notrans_nonunit_c32(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn ctrmv_upper_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_upper_colmajor_c32(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ctrmv(
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
        cblas_ctrmv_upper_trans_nonunit_c32(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn ctrmv_upper_conjtrans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f32; 2 * lda * n];
    fill_upper_colmajor_c32(&mut a, n, lda, true);

    let incx = 2usize; // stride 
    let mut x     = vec![0.0f32; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f32; x.len()];

    fill_vec_strided_c32(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ctrmv(
        UpLo::UpperTriangular,
        Trans::ConjTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_ctrmv_upper_conjtrans_nonunit_c32(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c32(&x, &x_ref, n, incx, 1e-5);
}

#[test]
fn ztrmv_upper_lower_equivalent() {
    let n   = 64usize;
    let lda = max(n, n + 3);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_diag_only_colmajor_c64(&mut a, n, lda);

    let incx = 1usize;
    let mut x_upper = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut x_lower = vec![0.0f64; x_upper.len()];

    fill_vec_strided_c64(&mut x_upper, n, incx);
    x_lower.copy_from_slice(&x_upper);

    ztrmv(
        UpLo::UpperTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x_upper,
        incx as isize,
    );

    ztrmv(
        UpLo::LowerTriangular,
        Trans::NoTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x_lower,
        incx as isize,
    );

    assert_close_c64(&x_upper, &x_lower, n, incx, 1e-12);
}

#[test]
fn ztrmv_upper_matches_cblas_notrans() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_upper_colmajor_c64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ztrmv(
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
        cblas_ztrmv_upper_notrans_nonunit_c64(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn ztrmv_upper_trans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_upper_colmajor_c64(&mut a, n, lda, true);

    let incx = 1usize;
    let mut x     = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ztrmv(
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
        cblas_ztrmv_upper_trans_nonunit_c64(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c64(&x, &x_ref, n, incx, 1e-12);
}

#[test]
fn ztrmv_upper_conjtrans_matches_cblas() {
    let n   = 73usize;
    let lda = max(n, n + 5);

    let mut a = vec![0.0f64; 2 * lda * n];
    fill_upper_colmajor_c64(&mut a, n, lda, true);

    let incx = 2usize;
    let mut x     = vec![0.0f64; 2 * (1 + (n - 1) * incx)];
    let mut x_ref = vec![0.0f64; x.len()];

    fill_vec_strided_c64(&mut x, n, incx);
    x_ref.copy_from_slice(&x);

    ztrmv(
        UpLo::UpperTriangular,
        Trans::ConjTrans,
        Diag::NonUnitDiag,
        n,
        &a,
        1,
        lda as isize,
        &mut x,
        incx as isize,
    );

    unsafe {
        cblas_ztrmv_upper_conjtrans_nonunit_c64(
            n as i32,
            a.as_ptr(),
            lda as i32,
            x_ref.as_mut_ptr(),
            incx as i32,
        );
    }

    assert_close_c64(&x, &x_ref, n, incx, 1e-12);
}
