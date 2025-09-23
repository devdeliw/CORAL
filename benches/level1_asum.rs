use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    sasum::sasum,
    dasum::dasum,
    scasum::scasum,
    dzasum::dzasum,
};

use cblas_sys::{
    cblas_sasum,
    cblas_dasum,
    cblas_scasum,
    cblas_dzasum,
};


pub fn bench_asum(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    // real 
    let xr_f32 = vec![1.0f32; n];
    let xr_f64 = vec![1.0f64; n];

    // complex 
    let mut xc_f32 = vec![0.0f32; 2 * n];
    let mut xc_f64 = vec![0.0f64; 2 * n];
    for i in 0..n {
        xc_f32[2 * i]     =  1.0;
        xc_f32[2 * i + 1] = -1.0;
        xc_f64[2 * i]     =  1.0;
        xc_f64[2 * i + 1] = -1.0;
    }

    c.bench_function("coral_sasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = sasum(
                    black_box(n),
                    black_box(&xr_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_sasum(
                        black_box(n as i32),
                        black_box(xr_f32.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = dasum(
                    black_box(n),
                    black_box(&xr_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_dasum(
                        black_box(n as i32),
                        black_box(xr_f64.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_scasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = scasum(
                    black_box(n),
                    black_box(&xc_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_scasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_scasum(
                        black_box(n as i32),
                        black_box(xc_f32.as_ptr() as *const _),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dzasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = dzasum(
                    black_box(n),
                    black_box(&xc_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dzasum", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_dzasum(
                        black_box(n as i32),
                        black_box(xc_f64.as_ptr() as *const _),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_asum);
criterion_main!(benches);

