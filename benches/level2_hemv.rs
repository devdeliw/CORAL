use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rusty_blas::level2::{
    enums::UpLo,
    chemv::chemv,
    zhemv::zhemv,
};
use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_UPLO,
    cblas_chemv,
    cblas_zhemv,
};

unsafe fn cblas_chemv_colmajor(
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: [f32; 2],
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: [f32; 2],
    y: *mut f32,
    incy: i32,
) { unsafe { 
    cblas_chemv(
        CBLAS_LAYOUT::CblasColMajor,
        uplo,
        n,
        &alpha as *const [f32; 2],
        a.cast::<[f32; 2]>(),
        lda,
        x.cast::<[f32; 2]>(),
        incx,
        &beta as *const [f32; 2],
        y.cast::<[f32; 2]>(),
        incy,
    );
}} 

unsafe fn cblas_zhemv_colmajor(
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: [f64; 2],
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: [f64; 2],
    y: *mut f64,
    incy: i32,
) { unsafe { 
    cblas_zhemv(
        CBLAS_LAYOUT::CblasColMajor,
        uplo,
        n,
        &alpha as *const [f64; 2],
        a.cast::<[f64; 2]>(),
        lda,
        x.cast::<[f64; 2]>(),
        incx,
        &beta as *const [f64; 2],
        y.cast::<[f64; 2]>(),
        incy,
    );
}} 

fn bench_hemv(c: &mut Criterion) {
    let n: usize = 1024;
    let lda: usize = n + 8;

    let alpha32_c: [f32; 2] = [1.000123, -0.250321];
    let beta32_c:  [f32; 2] = [0.000321,  0.125555];
    let alpha64_c: [f64; 2] = [1.000123_f64, -0.250321_f64];
    let beta64_c:  [f64; 2] = [0.000321_f64,  0.125555_f64];

    let a32 = vec![1.0f32; 2 * lda * n];
    let x32 = vec![1.0f32; 2 * n];
    let a64 = vec![1.0f64; 2 * lda * n];
    let x64 = vec![1.0f64; 2 * n];

    c.bench_function("rusty_chemv_upper", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                chemv(
                    UpLo::UpperTriangular,
                    n,
                    alpha32_c,
                    &a32,
                    1isize,
                    lda as isize,
                    &x32,
                    1isize,
                    beta32_c,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_chemv_upper", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_chemv_colmajor(
                    CBLAS_UPLO::CblasUpper,
                    n as i32,
                    alpha32_c,
                    a32.as_ptr(),
                    lda as i32,
                    x32.as_ptr(),
                    1i32,
                    beta32_c,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_chemv_lower", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                chemv(
                    UpLo::LowerTriangular,
                    n,
                    alpha32_c,
                    &a32,
                    1isize,
                    lda as isize,
                    &x32,
                    1isize,
                    beta32_c,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_chemv_lower", |b| {
        let y0 = vec![2.0f32; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_chemv_colmajor(
                    CBLAS_UPLO::CblasLower,
                    n as i32,
                    alpha32_c,
                    a32.as_ptr(),
                    lda as i32,
                    x32.as_ptr(),
                    1i32,
                    beta32_c,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_zhemv_upper", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zhemv(
                    UpLo::UpperTriangular,
                    n,
                    alpha64_c,
                    &a64,
                    1isize,
                    lda as isize,
                    &x64,
                    1isize,
                    beta64_c,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_zhemv_upper", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zhemv_colmajor(
                    CBLAS_UPLO::CblasUpper,
                    n as i32,
                    alpha64_c,
                    a64.as_ptr(),
                    lda as i32,
                    x64.as_ptr(),
                    1i32,
                    beta64_c,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_zhemv_lower", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zhemv(
                    UpLo::LowerTriangular,
                    n,
                    alpha64_c,
                    &a64,
                    1isize,
                    lda as isize,
                    &x64,
                    1isize,
                    beta64_c,
                    y.as_mut_slice(),
                    1isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_zhemv_lower", |b| {
        let y0 = vec![2.0f64; 2 * n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zhemv_colmajor(
                    CBLAS_UPLO::CblasLower,
                    n as i32,
                    alpha64_c,
                    a64.as_ptr(),
                    lda as i32,
                    x64.as_ptr(),
                    1i32,
                    beta64_c,
                    y.as_mut_ptr(),
                    1i32,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_hemv);
criterion_main!(benches);

