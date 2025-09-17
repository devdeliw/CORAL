use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

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
    let lda: usize = n;

    // upper-triangular (including diagonal)
    let matrix_upper = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in 0..=j {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    // lower-triangular (including diagonal)
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

    // upper: coral
    c.bench_function("coral_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    CoralTriangular::UpperTriangular,
                    CoralTranspose::NoTranspose,
                    CoralDiagonal::NonUnitDiagonal,
                    n,
                    &matrix_upper,
                    lda,
                    x.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    // upper: cblas
    c.bench_function("cblas_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strsv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasUpper,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_DIAG::CblasNonUnit,
                    n as i32,
                    matrix_upper.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    // lower: coral
    c.bench_function("coral_strsv_lower_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    CoralTriangular::LowerTriangular,
                    CoralTranspose::NoTranspose,
                    CoralDiagonal::NonUnitDiagonal,
                    n,
                    &matrix_lower,
                    lda,
                    x.as_mut_slice(),
                    1,
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
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_UPLO::CblasLower,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_DIAG::CblasNonUnit,
                    n as i32,
                    matrix_lower.as_ptr(),
                    lda as i32,
                    x.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_strsv);
criterion_main!(benches);

