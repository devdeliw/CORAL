use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rusty_blas::level2::{enums::UpLo, ssymv::ssymv, dsymv::dsymv};
use cblas_sys::{CBLAS_LAYOUT, CBLAS_UPLO, cblas_ssymv, cblas_dsymv};

fn bench_symv(c: &mut Criterion) {
    let n: usize  = 1024;
    let lda: usize = n + 8;

    let alpha32: f32 = 1.000123_f32;
    let beta32 : f32 = 0.000321_f32;
    let alpha64: f64 = 1.000123_f64;
    let beta64 : f64 = 0.000321_f64;

    let a32 = vec![1.0f32; lda * n];
    let x32 = vec![1.0f32; n];

    let a64 = vec![1.0f64; lda * n];
    let x64 = vec![1.0f64; n];

    c.bench_function("rusty_ssymv_upper", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                ssymv(
                    UpLo::UpperTriangular,
                    n,
                    alpha32,
                    &a32,
                    1isize,
                    lda as isize,
                    &x32,
                    1isize,
                    beta32,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_ssymv_upper", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_ssymv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasUpper,
                    n as i32,
                    alpha32,
                    a32.as_ptr(),
                    lda as i32,
                    x32.as_ptr(),
                    1i32,
                    beta32,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_ssymv_lower", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                ssymv(
                    UpLo::LowerTriangular,
                    n,
                    alpha32,
                    &a32,
                    1isize,
                    lda as isize,
                    &x32,
                    1isize,
                    beta32,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_ssymv_lower", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_ssymv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasLower,
                    n as i32,
                    alpha32,
                    a32.as_ptr(),
                    lda as i32,
                    x32.as_ptr(),
                    1i32,
                    beta32,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dsymv_upper", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dsymv(
                    UpLo::UpperTriangular,
                    n,
                    alpha64,
                    &a64,
                    1isize,
                    lda as isize,
                    &x64,
                    1isize,
                    beta64,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dsymv_upper", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dsymv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasUpper,
                    n as i32,
                    alpha64,
                    a64.as_ptr(),
                    lda as i32,
                    x64.as_ptr(),
                    1i32,
                    beta64,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dsymv_lower", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dsymv(
                    UpLo::LowerTriangular,
                    n,
                    alpha64,
                    &a64,
                    1isize,
                    lda as isize,
                    &x64,
                    1isize,
                    beta64,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dsymv_lower", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dsymv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasLower,
                    n as i32,
                    alpha64,
                    a64.as_ptr(),
                    lda as i32,
                    x64.as_ptr(),
                    1i32,
                    beta64,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_symv);
criterion_main!(benches);

