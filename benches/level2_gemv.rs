use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rusty_blas::level2::{enums::Trans, sgemv::sgemv, dgemv::dgemv};
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv, cblas_dgemv};

pub fn bench_gemv(c: &mut Criterion) {
    let m: usize = 1024;
    let n: usize = 1024;
    let lda: usize = m + 8;

    let alpha32: f32 = 1.000123;
    let beta32 : f32 = 0.000321;
    let alpha64: f64 = 1.000123_f64;
    let beta64 : f64 = 0.000321_f64;

    let a32 = vec![1.0f32; lda * n];
    let a64 = vec![1.0f64; lda * n];

    let x32_nt  = vec![1.0f32; n];
    let x32_t   = vec![1.0f32; m];
    let x64_nt  = vec![1.0f64; n];
    let x64_t   = vec![1.0f64; m];

    c.bench_function("rusty_sgemv", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    Trans::NoTrans, m, n, alpha32,
                    &a32, 1, lda as isize,
                    &x32_nt, 1,
                    beta32,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_sgemv", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32, n as i32,
                    alpha32,
                    a32.as_ptr(), lda as i32,
                    x32_nt.as_ptr(), 1,
                    beta32,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    Trans::Trans, m, n, alpha32,
                    &a32, 1, lda as isize,
                    &x32_t, 1,
                    beta32,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m as i32, n as i32,
                    alpha32,
                    a32.as_ptr(), lda as i32,
                    x32_t.as_ptr(), 1,
                    beta32,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dgemv", |b| {
        let y0 = vec![2.0f64; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dgemv(
                    Trans::NoTrans, m, n, alpha64,
                    &a64, 1, lda as isize,
                    &x64_nt, 1,
                    beta64,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dgemv", |b| {
        let y0 = vec![2.0f64; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32, n as i32,
                    alpha64,
                    a64.as_ptr(), lda as i32,
                    x64_nt.as_ptr(), 1,
                    beta64,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dgemv(
                    Trans::Trans, m, n, alpha64,
                    &a64, 1, lda as isize,
                    &x64_t, 1,
                    beta64,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m as i32, n as i32,
                    alpha64,
                    a64.as_ptr(), lda as i32,
                    x64_t.as_ptr(), 1,
                    beta64,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_gemv);
criterion_main!(benches);

