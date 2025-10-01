use blas_src as _;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion,
};
use coral::level2::{
    enums::CoralTriangular,
    ssymv::ssymv,
};
use cblas_sys::{
    cblas_ssymv, CBLAS_LAYOUT, CBLAS_UPLO,
};

fn bench_ssymv(c: &mut Criterion) {
    let n: usize  = 1024;
    let lda: usize = n;

    let alpha: f32 = 1.000_123;
    let beta:  f32 = 0.000_321;
    let matrix = vec![1.0f32; lda * n];

    let x  = vec![1.0f32; n];
    let y0 = vec![2.0f32; n];

    c.bench_function("coral_ssymv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                ssymv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ssymv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_ssymv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_ssymv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                ssymv(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ssymv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_ssymv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_ssymv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("ssymv_upper");
    for &n in &sizes {
        let lda = n;

        let alpha: f32 = 1.000_123;
        let beta:  f32 = 0.000_321;

        let matrix = vec![1.0f32; lda * n];
        let x      = vec![1.0f32; n];
        let y0     = vec![2.0f32; n];

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    ssymv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_ssymv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("ssymv_lower");
    for &n in &sizes {
        let lda = n;

        let alpha: f32 = 1.000_123;
        let beta:  f32 = 0.000_321;

        let matrix = vec![1.0f32; lda * n];
        let x      = vec![1.0f32; n];
        let y0     = vec![2.0f32; n];

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    ssymv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_ssymv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();
}

criterion_group!(benches, bench_ssymv, bench_ssymv_n);
criterion_main!(benches);

