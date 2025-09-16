use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

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
                    CoralTriangular::UpperTriangular,
                    CoralTranspose::NoTranspose,
                    CoralDiagonal::NonUnitDiagonal,
                    n,
                    &matrix,
                    lda,
                    x.as_mut_slice(),
                    1,
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
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_DIAG::CblasNonUnit,
                    n as i32,
                    matrix.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
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
                    CoralTriangular::UpperTriangular,
                    CoralTranspose::Transpose,
                    CoralDiagonal::NonUnitDiagonal,
                    n,
                    &matrix,
                    lda,
                    x.as_mut_slice(),
                    1,
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
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasTrans,
                    CBLAS_DIAG::CblasNonUnit,
                    n as i32,
                    matrix.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_strmv);
criterion_main!(benches);
