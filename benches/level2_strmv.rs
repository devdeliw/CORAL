use blas_src as _; 
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level2::{
    enums::{CoralTriangular, CoralTranspose, CoralDiagonal},
    strmv::strmv,
};

use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
    CBLAS_UPLO,
    CBLAS_DIAG,
    cblas_strmv,
};

pub fn bench_strmv(c: &mut Criterion) {
    let n:   usize = 1024;
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

    c.bench_function("cblas_strmv_upper_notrans", |b| {
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

    c.bench_function("cblas_strmv_upper_trans", |b| {
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

criterion_group!(benches, bench_strmv);
criterion_main!(benches);

