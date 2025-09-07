use cblas_sys::{
    CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_UPLO,
    cblas_strsv, cblas_dtrsv,
};
use rusty_blas::level2::{
    enums::{Trans, Diag, UpLo},
    strsv::strsv, 
    dtrsv::dtrsv, 
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

// pseudo-random float in [-0.5, 0.5]
fn f32_from_u32_f32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}
fn f64_from_u32_f64(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}

fn fill_upper_colmajor_f32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f32_from_u32_f32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in (j + 1)..n {
            a[j * lda + i] = 0.0;
        }
    }
}
fn fill_upper_colmajor_f64(a: &mut [f64], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f64_from_u32_f64(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in (j + 1)..n {
            a[j * lda + i] = 0.0;
        }
    }
}

fn fill_lower_colmajor_f32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f32_from_u32_f32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in 0..j {
            a[j * lda + i] = 0.0;
        }
    }
}
fn fill_lower_colmajor_f64(a: &mut [f64], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f64_from_u32_f64(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in 0..j {
            a[j * lda + i] = 0.0;
        }
    }
}

unsafe fn cblas_strsv_call(
    uplo: CBLAS_UPLO, 
    trans: CBLAS_TRANSPOSE, 
    n: i32,
    a: *const f32, 
    lda: i32,
    x: *mut f32,
    incx: i32
) { unsafe { 
    cblas_strsv(
        CBLAS_LAYOUT::CblasColMajor,
        uplo, 
        trans, 
        CBLAS_DIAG::CblasNonUnit, 
        n, 
        a, 
        lda, 
        x, 
        incx
    );
}}
unsafe fn cblas_dtrsv_call(
    uplo: CBLAS_UPLO, 
    trans: CBLAS_TRANSPOSE,
    n: i32,
    a: *const f64, 
    lda: i32, 
    x: *mut f64, 
    incx: i32
) { unsafe { 
    cblas_dtrsv(
        CBLAS_LAYOUT::CblasColMajor, 
        uplo, 
        trans, 
        CBLAS_DIAG::CblasNonUnit, 
        n, 
        a, 
        lda,
        x, 
        incx
    );
}}

fn bench_trsv(c: &mut Criterion) {
    let n: usize = 2048;
    let lda: usize = n + 8;

    let mut a32_upper = vec![0.0f32; lda * n];
    fill_upper_colmajor_f32(&mut a32_upper, n, lda);
    let mut a32_lower = vec![0.0f32; lda * n];
    fill_lower_colmajor_f32(&mut a32_lower, n, lda);

    let mut a64_upper = vec![0.0f64; lda * n];
    fill_upper_colmajor_f64(&mut a64_upper, n, lda);
    let mut a64_lower = vec![0.0f64; lda * n];
    fill_lower_colmajor_f64(&mut a64_lower, n, lda);

    let x32_0 = vec![1.0f32; n];
    let x64_0 = vec![1.0f64; n];

    c.bench_function("rusty_strusv_nt", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                strsv(
                    UpLo::UpperTriangular,
                    Trans::NoTrans,
                    Diag::NonUnitDiag,
                    n,
                    &a32_upper,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_strusv_nt", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_strsv_call(
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    n as i32,
                    a32_upper.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_strlsv_nt", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                strsv(
                    UpLo::LowerTriangular,
                    Trans::NoTrans,
                    Diag::NonUnitDiag,
                    n,
                    &a32_lower,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_strlsv_nt", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_strsv_call(
                    CBLAS_UPLO::CblasLower,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    n as i32,
                    a32_lower.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_strusv_t", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                strsv(
                    UpLo::UpperTriangular,
                    Trans::Trans,
                    Diag::NonUnitDiag,
                    n,
                    &a32_upper,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_strusv_t", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_strsv_call(
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasTrans,
                    n as i32,
                    a32_upper.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_strlsv_t", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                strsv(
                    UpLo::LowerTriangular,
                    Trans::Trans,
                    Diag::NonUnitDiag,
                    n,
                    &a32_lower,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_strlsv_t", |b| {
        let x0 = x32_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_strsv_call(
                    CBLAS_UPLO::CblasLower,
                    CBLAS_TRANSPOSE::CblasTrans,
                    n as i32,
                    a32_lower.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_dtrusv_nt", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                dtrsv(
                    UpLo::UpperTriangular,
                    Trans::NoTrans,
                    Diag::NonUnitDiag,
                    n,
                    &a64_upper,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_dtrusv_nt", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_dtrsv_call(
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    n as i32,
                    a64_upper.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_dtrlsv_nt", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                dtrsv(
                    UpLo::LowerTriangular,
                    Trans::NoTrans,
                    Diag::NonUnitDiag,
                    n,
                    &a64_lower,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_dtrlsv_nt", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_dtrsv_call(
                    CBLAS_UPLO::CblasLower,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    n as i32,
                    a64_lower.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_dtrusv_t", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                dtrsv(
                    UpLo::UpperTriangular,
                    Trans::Trans,
                    Diag::NonUnitDiag,
                    n,
                    &a64_upper,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_dtrusv_t", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_dtrsv_call(
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasTrans,
                    n as i32,
                    a64_upper.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("rusty_dtrlsv_t", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| {
                dtrsv(
                    UpLo::LowerTriangular,
                    Trans::Trans,
                    Diag::NonUnitDiag,
                    n,
                    &a64_lower,
                    1,
                    lda as isize,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("cblas_dtrlsv_t", |b| {
        let x0 = x64_0.clone();
        b.iter_batched_ref(
            || x0.clone(),
            |x| unsafe {
                cblas_dtrsv_call(
                    CBLAS_UPLO::CblasLower,
                    CBLAS_TRANSPOSE::CblasTrans,
                    n as i32,
                    a64_lower.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_trsv);
criterion_main!(benches);

