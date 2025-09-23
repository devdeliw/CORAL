use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    scopy::scopy,
    dcopy::dcopy,
    ccopy::ccopy,
    zcopy::zcopy,
};

use cblas_sys::{
    cblas_scopy,
    cblas_dcopy,
    cblas_ccopy,
    cblas_zcopy,
};

pub fn bench_copy(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

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

    c.bench_function("coral_scopy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                scopy(
                    black_box(n),
                    black_box(&x_f32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_scopy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                unsafe {
                    cblas_scopy(
                        black_box(n as i32),
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

    c.bench_function("coral_dcopy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                dcopy(
                    black_box(n),
                    black_box(&x_f64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dcopy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                unsafe {
                    cblas_dcopy(
                        black_box(n as i32),
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

    c.bench_function("coral_ccopy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                ccopy(
                    black_box(n),
                    black_box(&x_c32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ccopy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                unsafe {
                    cblas_ccopy(
                        black_box(n as i32),
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

    c.bench_function("coral_zcopy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                zcopy(
                    black_box(n),
                    black_box(&x_c64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zcopy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                unsafe {
                    cblas_zcopy(
                        black_box(n as i32),
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

criterion_group!(benches, bench_copy);
criterion_main!(benches);

