use blas_src as _; 
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box, BenchmarkId};

use coral::enums::CoralTranspose; 
use coral::level2::sgemv::sgemv; 

use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE, 
    cblas_sgemv, 
}; 

pub fn bench_sgemv(c: &mut Criterion) { 
    let m:   usize = 1024; 
    let n:   usize = 1024; 
    let lda: usize = m; 

    let alpha: f32 = 1.000123; 
    let beta : f32 = 0.000321; 
    let matrix     = vec![1.0f32; lda * n]; 

    let x_notrans  = vec![1.0f32; n]; 
    let x_trans    = vec![1.0f32; m];  

    c.bench_function("coral_sgemv_notrans", |b| { 
        let y0 = vec![2.0f32; m]; 
        b.iter_batched_ref(
            || y0.clone(), 
            |y| { 
                sgemv(
                    black_box(CoralTranspose::NoTranspose), 
                    black_box(m), 
                    black_box(n), 
                    black_box(alpha), 
                    black_box(&matrix), 
                    black_box(lda), 
                    black_box(&x_notrans), 
                    black_box(1),
                    black_box(beta), 
                    black_box(y.as_mut_slice()), 
                    black_box(1),
                );
            },
            BatchSize::SmallInput, 
        );
    });

    c.bench_function("blas_sgemv_notrans", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(m as i32), 
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x_notrans.as_ptr()), 
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    }); 

    c.bench_function("coral_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    black_box(CoralTranspose::Transpose),
                    black_box(m),
                    black_box(n), 
                    black_box(alpha),
                    black_box(&matrix), 
                    black_box(lda),
                    black_box(&x_trans),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(m as i32), 
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x_trans.as_ptr()),
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


pub fn bench_sgemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("sgemv_notrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f32 = 1.000123;
        let beta:  f32 = 0.000321;

        let matrix    = vec![1.0f32; lda * n];
        let x_notrans = vec![1.0f32; n];
        let y0        = vec![2.0f32; m];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    sgemv(
                        black_box(CoralTranspose::NoTranspose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_notrans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_sgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_notrans.as_ptr()),
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
    group.finish();

    let mut group_t = c.benchmark_group("sgemv_trans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f32 = 1.000123;
        let beta:  f32 = 0.000321;

        let matrix = vec![1.0f32; lda * n];
        let x_trans = vec![1.0f32; m];
        let y0      = vec![2.0f32; n];

        group_t.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    sgemv(
                        black_box(CoralTranspose::Transpose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_trans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_t.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_sgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_trans.as_ptr()),
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
    group_t.finish();
}

criterion_group!(benches, bench_sgemv, bench_sgemv_n);
criterion_main!(benches);

