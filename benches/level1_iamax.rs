use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    isamax::isamax,
    idamax::idamax,
    icamax::icamax,
    izamax::izamax,
};

use cblas_sys::{
    cblas_isamax,
    cblas_idamax,
    cblas_icamax,
    cblas_izamax,
};

pub fn bench_iamax(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    // real
    let x_f32 = vec![1.0f32; n];
    let x_f64 = vec![1.0f64; n];

    // complex
    let mut x_c32 = vec![0.0f32; 2 * n];
    let mut x_c64 = vec![0.0f64; 2 * n];
    for i in 0..n {
        x_c32[2 * i]     =  1.0;
        x_c32[2 * i + 1] = -1.0;
        x_c64[2 * i]     =  1.0;
        x_c64[2 * i + 1] = -1.0;
    }

    c.bench_function("coral_isamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = isamax(
                    black_box(n),
                    black_box(&x_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_isamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_isamax(
                        black_box(n as i32),
                        black_box(x_f32.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_idamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = idamax(
                    black_box(n),
                    black_box(&x_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_idamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_idamax(
                        black_box(n as i32),
                        black_box(x_f64.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_icamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = icamax(
                    black_box(n),
                    black_box(&x_c32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_icamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_icamax(
                        black_box(n as i32),
                        black_box(x_c32.as_ptr() as *const _),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_izamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = izamax(
                    black_box(n),
                    black_box(&x_c64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_izamax", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_izamax(
                        black_box(n as i32),
                        black_box(x_c64.as_ptr() as *const _),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_iamax);
criterion_main!(benches);

