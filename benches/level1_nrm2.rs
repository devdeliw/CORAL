use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    snrm2::snrm2,
    dnrm2::dnrm2,
    scnrm2::scnrm2,
    dznrm2::dznrm2,
};

use cblas_sys::{
    cblas_snrm2,
    cblas_dnrm2,
    cblas_scnrm2,
    cblas_dznrm2,
};

pub fn bench_nrm2(c: &mut Criterion) {
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

    c.bench_function("coral_snrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = snrm2(
                    black_box(n),
                    black_box(&x_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_snrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_snrm2(
                        black_box(n as i32),
                        black_box(x_f32.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dnrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = dnrm2(
                    black_box(n),
                    black_box(&x_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dnrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_dnrm2(
                        black_box(n as i32),
                        black_box(x_f64.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_scnrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = scnrm2(
                    black_box(n),
                    black_box(&x_c32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_scnrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_scnrm2(
                        black_box(n as i32),
                        black_box(x_c32.as_ptr() as *const _),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dznrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = dznrm2(
                    black_box(n),
                    black_box(&x_c64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dznrm2", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_dznrm2(
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

criterion_group!(benches, bench_nrm2);
criterion_main!(benches);

