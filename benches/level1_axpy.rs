use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use rusty_blas::level1::{
    saxpy::saxpy,
    daxpy::daxpy,
    caxpy::caxpy,
    zaxpy::zaxpy,
};
use cblas_sys::{
    cblas_saxpy, 
    cblas_daxpy, 
    cblas_caxpy, 
    cblas_zaxpy
};

fn bench_axpy(c: &mut Criterion) {
    let n: usize = 1_000_000;

    let alpha_s: f32 = 1.000123_f32;
    let alpha_d: f64 = 1.000123_f64;
    let alpha_c: [f32; 2] = [1.000123_f32, 0.000321_f32]; 
    let alpha_z: [f64; 2] = [1.000123_f64, 0.000321_f64];

    c.bench_function("rusty_saxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f32; n];
                let y = vec![2.0f32; n];
                (x, y)
            },
            |(x, mut y)| {
                saxpy(
                    black_box(n),
                    black_box(alpha_s),
                    black_box(&x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_saxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f32; n];
                let y = vec![2.0f32; n];
                (x, y)
            },
            |(x, mut y)| {
                unsafe {
                    cblas_saxpy(
                        black_box(n as i32),
                        black_box(alpha_s),
                        black_box(x.as_ptr()),
                        black_box(1i32),
                        black_box(y.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_daxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f64; n];
                let y = vec![2.0f64; n];
                (x, y)
            },
            |(x, mut y)| {
                daxpy(
                    black_box(n),
                    black_box(alpha_d),
                    black_box(&x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_daxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f64; n];
                let y = vec![2.0f64; n];
                (x, y)
            },
            |(x, mut y)| {
                unsafe {
                    cblas_daxpy(
                        black_box(n as i32),
                        black_box(alpha_d),
                        black_box(x.as_ptr()),
                        black_box(1i32),
                        black_box(y.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_caxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f32; 2 * n];
                let y = vec![2.0f32; 2 * n];
                (x, y)
            },
            |(x, mut y)| {
                caxpy(
                    black_box(n),
                    black_box(alpha_c),
                    black_box(&x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_caxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f32; 2 * n];
                let y = vec![2.0f32; 2 * n];
                (x, y)
            },
            |(x, mut y)| {
                unsafe {
                    let alpha_ptr: *const [f32; 2] = &alpha_c;
                    cblas_caxpy(
                        black_box(n as i32),
                        black_box(alpha_ptr),
                        black_box(x.as_ptr() as *const [f32; 2]),
                        black_box(1i32),
                        black_box(y.as_mut_ptr() as *mut [f32; 2]),
                        black_box(1i32),
                    );
                }
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_zaxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f64; 2 * n];
                let y = vec![2.0f64; 2 * n];
                (x, y)
            },
            |(x, mut y)| {
                zaxpy(
                    black_box(n),
                    black_box(alpha_z),
                    black_box(&x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_zaxpy", |b| {
        b.iter_batched(
            || {
                let x = vec![1.0f64; 2 * n];
                let y = vec![2.0f64; 2 * n];
                (x, y)
            },
            |(x, mut y)| {
                unsafe {
                    let alpha_ptr: *const [f64; 2] = &alpha_z;
                    cblas_zaxpy(
                        black_box(n as i32),
                        black_box(alpha_ptr),
                        black_box(x.as_ptr() as *const [f64; 2]),
                        black_box(1i32),
                        black_box(y.as_mut_ptr() as *mut [f64; 2]),
                        black_box(1i32),
                    );
                }
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_axpy);
criterion_main!(benches);

