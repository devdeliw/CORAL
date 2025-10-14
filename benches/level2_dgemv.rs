use blas_src as _; 
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::enums::CoralTranspose; 
use coral::level2::dgemv::dgemv; 

use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE, 
    cblas_dgemv, 
}; 

pub fn bench_dgemv(c: &mut Criterion) { 
    let m:   usize = 1024; 
    let n:   usize = 1024; 
    let lda: usize = m; 

    let alpha: f64 = 1.000123; 
    let beta : f64 = 0.000321; 
    let matrix     = vec![1.0f64; lda * n]; 

    let x_notrans  = vec![1.0f64; n]; 
    let x_trans    = vec![1.0f64; m];  

    c.bench_function("coral_dgemv_notrans", |b| { 
        let y0 = vec![2.0f64; m]; 
        b.iter_batched_ref(
            || y0.clone(), 
            |y| { 
                dgemv(
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

    c.bench_function("blas_dgemv_notrans", |b| {
        let y0 = vec![2.0f64; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
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

    c.bench_function("coral_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dgemv(
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

    c.bench_function("blas_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
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

criterion_group!(benches, bench_dgemv);
criterion_main!(benches);

