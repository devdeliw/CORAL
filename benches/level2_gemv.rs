use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rusty_blas::level2::{
    enums::Trans, 
    sgemv::sgemv, 
    dgemv::dgemv, 
    cgemv::cgemv, 
    zgemv::zgemv
};
use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE, 
    cblas_sgemv, 
    cblas_dgemv, 
    cblas_cgemv, 
    cblas_zgemv
};

pub fn bench_gemv(c: &mut Criterion) {
    let m: usize   = 1024;
    let n: usize   = 1024;
    let lda: usize = m + 8;

    let alpha32 : f32 = 1.000123;
    let beta32  : f32 = 0.000321;
    let alpha64 : f64 = 1.000123_f64;
    let beta64  : f64 = 0.000321_f64;

    let alpha32_c : [f32; 2] = [1.000123, -0.250321];
    let beta32_c  : [f32; 2] = [0.000321,  0.125777];

    let a32  = vec![1.0f32; lda * n];
    let a64  = vec![1.0f64; lda * n];
    let a32c = vec![1.0f32; 2 * lda * n];

    let x32_nt  = vec![1.0f32; n];
    let x32_t   = vec![1.0f32; m];
    let x64_nt  = vec![1.0f64; n];
    let x64_t   = vec![1.0f64; m];

    let x32c_nt = vec![1.0f32; 2 * n];
    let x32c_t  = vec![1.0f32; 2 * m];

    let alpha64_c : [f64; 2] = [1.000123_f64, -0.250321_f64];
    let beta64_c  : [f64; 2] = [0.000321_f64,  0.125777_f64];

    let a64c     = vec![1.0f64; 2 * lda * n];
    let x64c_nt  = vec![1.0f64; 2 * n];
    let x64c_t   = vec![1.0f64; 2 * m];

    c.bench_function("rusty_sgemv", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    Trans::NoTrans, m, n, alpha32,
                    &a32, 1, lda as isize,
                    &x32_nt, 1,
                    beta32,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_sgemv", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32, n as i32,
                    alpha32,
                    a32.as_ptr(), lda as i32,
                    x32_nt.as_ptr(), 1,
                    beta32,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    Trans::Trans, m, n, alpha32,
                    &a32, 1, lda as isize,
                    &x32_t, 1,
                    beta32,
                    y.as_mut_slice(), 1,
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
                    m as i32, n as i32,
                    alpha32,
                    a32.as_ptr(), lda as i32,
                    x32_t.as_ptr(), 1,
                    beta32,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dgemv", |b| {
        let y0 = vec![2.0f64; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dgemv(
                    Trans::NoTrans, m, n, alpha64,
                    &a64, 1, lda as isize,
                    &x64_nt, 1,
                    beta64,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dgemv", |b| {
        let y0 = vec![2.0f64; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32, n as i32,
                    alpha64,
                    a64.as_ptr(), lda as i32,
                    x64_nt.as_ptr(), 1,
                    beta64,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                dgemv(
                    Trans::Trans, m, n, alpha64,
                    &a64, 1, lda as isize,
                    &x64_t, 1,
                    beta64,
                    y.as_mut_slice(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dgemv_trans", |b| {
        let y0 = vec![2.0f64; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_dgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m as i32, n as i32,
                    alpha64,
                    a64.as_ptr(), lda as i32,
                    x64_t.as_ptr(), 1,
                    beta64,
                    y.as_mut_ptr(), 1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_cgemv", |b| {
        let y0 = vec![2.0f32; 2 * m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                cgemv(
                    Trans::NoTrans,
                    m, n,
                    alpha32_c,
                    &a32c,
                    1,              
                    lda as isize,    
                    &x32c_nt,
                    1,               
                    beta32_c,
                    y.as_mut_slice(),
                    1,               
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_cgemv", |b| {
        let y0 = vec![2.0f32; 2 * m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_cgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32,
                    n as i32,
                    alpha32_c.as_ptr() as *const _,
                    a32c.as_ptr().cast::<[f32; 2]>(),
                    lda as i32,
                    x32c_nt.as_ptr().cast::<[f32; 2]>(),
                    1,
                    beta32_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f32; 2]>(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_cgemv_trans", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                cgemv(
                    Trans::Trans,
                    m, n,
                    alpha32_c,
                    &a32c,
                    1,
                    lda as isize,
                    &x32c_t,
                    1,
                    beta32_c,
                    y.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_cgemv_trans", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_cgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m as i32,
                    n as i32,
                    alpha32_c.as_ptr() as *const _,
                    a32c.as_ptr().cast::<[f32; 2]>(), 
                    lda as i32,
                    x32c_t.as_ptr().cast::<[f32; 2]>(),
                    1,
                    beta32_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f32; 2]>(), 
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_cgemv_conjtrans", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                cgemv(
                    Trans::ConjTrans,
                    m, n,
                    alpha32_c,
                    &a32c,
                    1,
                    lda as isize,
                    &x32c_t,
                    1,
                    beta32_c,
                    y.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_cgemv_conjtrans", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_cgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasConjTrans,
                    m as i32, 
                    n as i32,
                    alpha32_c.as_ptr() as *const _,
                    a32c.as_ptr().cast::<[f32; 2]>(), 
                    lda as i32,
                    x32c_t.as_ptr().cast::<[f32; 2]>(), 
                    1,
                    beta32_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f32; 2]>(), 
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

c.bench_function("rusty_zgemv", |b| {
        let y0 = vec![2.0f64; 2 * m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zgemv(
                    Trans::NoTrans,
                    m, n,
                    alpha64_c,
                    &a64c,
                    1,               
                    lda as isize,    
                    &x64c_nt,
                    1,               
                    beta64_c,
                    y.as_mut_slice(),
                    1,               
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_zgemv", |b| {
        let y0 = vec![2.0f64; 2 * m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32,
                    n as i32,
                    alpha64_c.as_ptr() as *const _,
                    a64c.as_ptr().cast::<[f64; 2]>(),
                    lda as i32,
                    x64c_nt.as_ptr().cast::<[f64; 2]>(),
                    1,
                    beta64_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f64; 2]>(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_zgemv_trans", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zgemv(
                    Trans::Trans,
                    m, n,
                    alpha64_c,
                    &a64c,
                    1,
                    lda as isize,
                    &x64c_t,
                    1,
                    beta64_c,
                    y.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_zgemv_trans", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m as i32,
                    n as i32,
                    alpha64_c.as_ptr() as *const _,
                    a64c.as_ptr().cast::<[f64; 2]>(), 
                    lda as i32,
                    x64c_t.as_ptr().cast::<[f64; 2]>(),
                    1,
                    beta64_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f64; 2]>(), 
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_zgemv_conjtrans", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zgemv(
                    Trans::ConjTrans,
                    m, n,
                    alpha64_c,
                    &a64c,
                    1,
                    lda as isize,
                    &x64c_t,
                    1,
                    beta64_c,
                    y.as_mut_slice(),
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_zgemv_conjtrans", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zgemv(
                    CBLAS_LAYOUT::CblasColMajor,
                    CBLAS_TRANSPOSE::CblasConjTrans,
                    m as i32, 
                    n as i32,
                    alpha64_c.as_ptr() as *const _,
                    a64c.as_ptr().cast::<[f64; 2]>(), 
                    lda as i32,
                    x64c_t.as_ptr().cast::<[f64; 2]>(), 
                    1,
                    beta64_c.as_ptr() as *const _,
                    y.as_mut_ptr().cast::<[f64; 2]>(), 
                    1,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_gemv);
criterion_main!(benches);

