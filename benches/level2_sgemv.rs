use criterion::{criterion_group, criterion_main, BatchSize, Criterion}; 
use coral::level2::{
    enums::CoralTranspose, 
    sgemv::sgemv, 
}; 
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
                    CoralTranspose::NoTranspose, 
                    m, 
                    n, 
                    alpha, 
                    &matrix, 
                    lda, 
                    &x_notrans, 
                    1,
                    beta, 
                    y.as_mut_slice(), 
                    1,
                );
            },
            BatchSize::SmallInput, 
        );
    });

    c.bench_function("cblas_sgemv_notrans", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32, 
                    n as i32,
                    alpha,
                    matrix.as_ptr(),
                    lda as i32,
                    x_notrans.as_ptr(), 
                    1,
                    beta,
                    y.as_mut_ptr(),
                    1,
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
                    CoralTranspose::Transpose,
                    m,
                    n, 
                    alpha,
                    &matrix, 
                    lda,
                    &x_trans,
                    1,
                    beta,
                    y.as_mut_slice(),
                    1,
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
                    m as i32, 
                    n as i32,
                    alpha,
                    matrix.as_ptr(),
                    lda as i32,
                    x_trans.as_ptr(),
                    1,
                    beta,
                    y.as_mut_ptr(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_sgemv);
criterion_main!(benches);

