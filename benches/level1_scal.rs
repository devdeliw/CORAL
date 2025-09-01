use rusty_blas::level1::{ 
    sscal::sscal, 
    dscal::dscal, 
    cscal::cscal, 
    zscal::zscal, 
    csscal::csscal, 
    zdscal::zdscal, 
};
use cblas_sys::{
    cblas_sscal, 
    cblas_dscal, 
    cblas_cscal, 
    cblas_zscal, 
    cblas_csscal, 
    cblas_zdscal
};
use criterion::{criterion_group, criterion_main, Criterion, black_box, BatchSize};

fn bench_scal(c: &mut Criterion) {
    let n: usize = 10_000_000;
    let alpha_f32: f32 = 1.000123;
    let alpha_f64: f64 = 0.99991;
    let alpha_cf32: [f32; 2] = [0.7, 0.3];
    let alpha_zf64: [f64; 2] = [0.6, -0.2];
    let alpha_css: f32 = 1.2345;
    let alpha_zds: f64 = 0.987654321;

    c.bench_function("rusty_sscal", |b| {
        b.iter_batched(
            || vec![1.0f32; n],
            |mut data| {
                sscal(
                    black_box(n), 
                    black_box(alpha_f32),
                    black_box(&mut data), 
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_sscal", |b| {
        b.iter_batched(
            || vec![1.0f32; n],
            |mut data| unsafe {
                cblas_sscal(
                    black_box(n as i32),
                    black_box(alpha_f32),
                    black_box(data.as_mut_ptr()),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_dscal", |b| {
        b.iter_batched(
            || vec![1.0f64; n],
            |mut data| {
                dscal(
                    black_box(n), 
                    black_box(alpha_f64),
                    black_box(&mut data), 
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_dscal", |b| {
        b.iter_batched(
            || vec![1.0f64; n],
            |mut data| unsafe {
                cblas_dscal(
                    black_box(n as i32),
                    black_box(alpha_f64),
                    black_box(data.as_mut_ptr()),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_cscal", |b| {
        b.iter_batched(
            || vec![1.0f32; 2 * n],
            |mut data| {
                cscal(
                    black_box(n),
                    black_box(alpha_cf32),
                    black_box(&mut data),
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_cscal", |b| {
        b.iter_batched(
            || vec![1.0f32; 2 * n],
            |mut data| unsafe {
                let alpha_ptr = &alpha_cf32 as *const [f32; 2];
                let x_ptr = data.as_mut_ptr().cast::<[f32; 2]>();
                cblas_cscal(
                    black_box(n as i32),
                    black_box(alpha_ptr),
                    black_box(x_ptr),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_zscal", |b| {
        b.iter_batched(
            || vec![1.0f64; 2 * n],
            |mut data| {
                zscal(
                    black_box(n),
                    black_box(alpha_zf64),
                    black_box(&mut data), 
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_zscal", |b| {
        b.iter_batched(
            || vec![1.0f64; 2 * n],
            |mut data| unsafe {
                let alpha_ptr = &alpha_zf64 as *const [f64; 2];
                let x_ptr = data.as_mut_ptr().cast::<[f64; 2]>();
                cblas_zscal(
                    black_box(n as i32),
                    black_box(alpha_ptr),
                    black_box(x_ptr),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_csscal", |b| {
        b.iter_batched(
            || vec![1.0f32; 2 * n],
            |mut data| {
                csscal(
                    black_box(n),
                    black_box(alpha_css), 
                    black_box(&mut data),
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_csscal", |b| {
        b.iter_batched(
            || vec![1.0f32; 2 * n],
            |mut data| unsafe {
                let x_ptr = data.as_mut_ptr().cast::<[f32; 2]>();
                cblas_csscal(
                    black_box(n as i32),
                    black_box(alpha_css),
                    black_box(x_ptr),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_zdscal", |b| {
        b.iter_batched(
            || vec![1.0f64; 2 * n],
            |mut data| {
                zdscal(
                    black_box(n),
                    black_box(alpha_zds),
                    black_box(&mut data), 
                    black_box(1isize),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_zdscal", |b| {
        b.iter_batched(
            || vec![1.0f64; 2 * n],
            |mut data| unsafe {
                let x_ptr = data.as_mut_ptr().cast::<[f64; 2]>();
                cblas_zdscal(
                    black_box(n as i32),
                    black_box(alpha_zds),
                    black_box(x_ptr),
                    black_box(1i32),
                );
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_scal);
criterion_main!(benches);

