use blas_src as _;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use coral::level2::{
    enums::{CoralDiagonal, CoralTriangular, CoralTranspose},
    strmv::strmv,
};

use cblas_sys::{cblas_strmv, CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_UPLO};

pub fn bench_strmv(c: &mut Criterion) {
    let n: usize = 1024;
    let lda: usize = n;

    let matrix = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in 0..=j {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    let x_init = vec![1.0f32; n];

    c.bench_function("coral_strmv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strmv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(CoralTranspose::NoTranspose),
                    black_box(CoralDiagonal::NonUnitDiagonal),
                    black_box(n),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_strmv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strmv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_DIAG::CblasNonUnit),
                    black_box(n as i32),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_strmv_upper_trans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strmv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(CoralTranspose::Transpose),
                    black_box(CoralDiagonal::NonUnitDiagonal),
                    black_box(n),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_strmv_upper_trans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strmv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(CBLAS_DIAG::CblasNonUnit),
                    black_box(n as i32),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_strmv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    /*let mut group_u = c.benchmark_group("strmv_upper_notrans");
    for &n in &sizes {
        let lda = n;

        let matrix = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strmv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strmv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();*/

    let mut group_ut = c.benchmark_group("strmv_upper_trans");
    for &n in &sizes {
        let lda = n;

        let matrix = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_ut.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strmv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_ut.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strmv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_ut.finish();

    let mut group_l = c.benchmark_group("strmv_lower_notrans");
    for &n in &sizes {
        let lda = n;

        let matrix = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strmv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strmv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();

    let mut group_lt = c.benchmark_group("strmv_lower_trans");
    for &n in &sizes {
        let lda = n;

        let matrix = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_lt.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strmv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_lt.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strmv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_lt.finish();
}

criterion_group!(benches, bench_strmv, bench_strmv_n);
criterion_main!(benches);

