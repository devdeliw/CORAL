use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level2::{
    enums::{CoralTriangular, CoralTranspose, CoralDiagonal},
    strsv::strsv,
};

use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
    CBLAS_UPLO,
    CBLAS_DIAG,
    cblas_strsv,
};

pub fn bench_strsv(c: &mut Criterion) {
    let n:   usize = 1024;
    let lda: usize = n + 8;

    // upper-triangular with diagonal
    let matrix_upper = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in 0..=j {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    // lower-triangular with diagonal 
    let matrix_lower = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    let x_init = vec![1.0f32; n];

    // upper; coral
    c.bench_function("coral_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(CoralTranspose::NoTranspose),
                    black_box(CoralDiagonal::UnitDiagonal),
                    black_box(n),
                    black_box(&matrix_upper),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    // upper; cblas
    c.bench_function("cblas_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strsv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_DIAG::CblasUnit),
                    black_box(n as i32),
                    black_box(matrix_upper.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    // lower; coral
    c.bench_function("coral_strsv_lower_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(CoralTranspose::NoTranspose),
                    black_box(CoralDiagonal::NonUnitDiagonal),
                    black_box(n),
                    black_box(&matrix_lower),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    // lower: cblas
    c.bench_function("cblas_strsv_lower_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strsv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_DIAG::CblasNonUnit),
                    black_box(n as i32),
                    black_box(matrix_lower.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_strsv);
criterion_main!(benches);

