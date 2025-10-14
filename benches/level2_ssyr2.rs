use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level2::ssyr2::ssyr2;
use coral::enums::CoralTriangular;

use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_UPLO,
    cblas_ssyr2,
};

pub fn bench_ssyr2(c: &mut Criterion) {
    let n    : usize = 1024;
    let lda  : usize = n;
    let alpha: f32   = 1.000123;

    let a0 = vec![0.0f32; lda * n];
    let x  = vec![1.0f32; n];
    let y  = vec![1.0f32; n];

    c.bench_function("coral_ssyr2_upper", |b| {
        b.iter_batched_ref(
            || a0.clone(),
            |a| {
                ssyr2(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&x),
                    black_box(1),
                    black_box(&y),
                    black_box(1),
                    black_box(a.as_mut_slice()),
                    black_box(lda),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ssyr2_upper", |b| {
        b.iter_batched_ref(
            || a0.clone(),
            |a| unsafe {
                cblas_ssyr2(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(y.as_ptr()),
                    black_box(1),
                    black_box(a.as_mut_ptr()),
                    black_box(lda as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_ssyr2_lower", |b| {
        b.iter_batched_ref(
            || a0.clone(),
            |a| {
                ssyr2(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&x),
                    black_box(1),
                    black_box(&y),
                    black_box(1),
                    black_box(a.as_mut_slice()),
                    black_box(lda),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ssyr2_lower", |b| {
        b.iter_batched_ref(
            || a0.clone(),
            |a| unsafe {
                cblas_ssyr2(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(y.as_ptr()),
                    black_box(1),
                    black_box(a.as_mut_ptr()),
                    black_box(lda as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_ssyr2);
criterion_main!(benches);

