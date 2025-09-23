use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    saxpy::saxpy,
    daxpy::daxpy,
    caxpy::caxpy,
    zaxpy::zaxpy,
};

use cblas_sys::{
    cblas_saxpy,
    cblas_daxpy,
    cblas_caxpy,
    cblas_zaxpy,
};

pub fn bench_axpy(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    let alpha_f32: f32 = 1.000123;
    let alpha_f64: f64 = 1.000123_f64;
    let alpha_c32: [f32; 2] = [1.000123, -0.999877];
    let alpha_c64: [f64; 2] = [1.000123_f64, -0.999877_f64];

    // real
    let x_f32  = vec![1.0f32; n];
    let x_f64  = vec![1.0f64; n];
    let y0_f32 = vec![0.0f32; n];
    let y0_f64 = vec![0.0f64; n];

    // complex 
    let mut x_c32  = vec![0.0f32; 2 * n];
    let     y0_c32 = vec![0.0f32; 2 * n];
    let mut x_c64  = vec![0.0f64; 2 * n];
    let     y0_c64 = vec![0.0f64; 2 * n];

    for i in 0..n {
        x_c32[2 * i]     =  1.0;
        x_c32[2 * i + 1] = -1.0;
        x_c64[2 * i]     =  1.0;
        x_c64[2 * i + 1] = -1.0;
    }

    c.bench_function("coral_saxpy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                saxpy(
                    black_box(n),
                    black_box(alpha_f32),
                    black_box(&x_f32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_saxpy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                unsafe {
                    cblas_saxpy(
                        black_box(n as i32),
                        black_box(alpha_f32),
                        black_box(x_f32.as_ptr()),
                        black_box(inc),
                        black_box(y.as_mut_ptr()),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_daxpy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                daxpy(
                    black_box(n),
                    black_box(alpha_f64),
                    black_box(&x_f64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_daxpy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                unsafe {
                    cblas_daxpy(
                        black_box(n as i32),
                        black_box(alpha_f64),
                        black_box(x_f64.as_ptr()),
                        black_box(inc),
                        black_box(y.as_mut_ptr()),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_caxpy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                caxpy(
                    black_box(n),
                    black_box(alpha_c32),
                    black_box(&x_c32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_caxpy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                unsafe {
                    cblas_caxpy(
                        black_box(n as i32),
                        black_box(alpha_c32.as_ptr() as *const _),
                        black_box(x_c32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zaxpy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                zaxpy(
                    black_box(n),
                    black_box(alpha_c64),
                    black_box(&x_c64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zaxpy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                unsafe {
                    cblas_zaxpy(
                        black_box(n as i32),
                        black_box(alpha_c64.as_ptr() as *const _),
                        black_box(x_c64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_axpy);
criterion_main!(benches);

