use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    sswap::sswap,
    dswap::dswap,
    cswap::cswap,
    zswap::zswap,
};

use cblas_sys::{
    cblas_sswap,
    cblas_dswap,
    cblas_cswap,
    cblas_zswap,
};

pub fn bench_swap(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    // real
    let x0_f32 = vec![1.0f32; n];
    let y0_f32 = vec![2.0f32; n];
    let x0_f64 = vec![1.0f64; n];
    let y0_f64 = vec![2.0f64; n];

    // complex 
    let mut x0_c32 = vec![0.0f32; 2 * n];
    let mut y0_c32 = vec![0.0f32; 2 * n];
    let mut x0_c64 = vec![0.0f64; 2 * n];
    let mut y0_c64 = vec![0.0f64; 2 * n];

    for i in 0..n {
        x0_c32[2 * i]     =  1.0;
        x0_c32[2 * i + 1] = -1.0;
        y0_c32[2 * i]     =  2.0;
        y0_c32[2 * i + 1] = -2.0;

        x0_c64[2 * i]     =  1.0;
        x0_c64[2 * i + 1] = -1.0;
        y0_c64[2 * i]     =  2.0;
        y0_c64[2 * i + 1] = -2.0;
    }

    c.bench_function("coral_sswap", |b| {
        b.iter_batched_ref(
            || (x0_f32.clone(), y0_f32.clone()),
            |(x, y)| {
                sswap(
                    black_box(n),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sswap", |b| {
        b.iter_batched_ref(
            || (x0_f32.clone(), y0_f32.clone()),
            |(x, y)| {
                unsafe {
                    cblas_sswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(inc),
                        black_box(y.as_mut_ptr()),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dswap", |b| {
        b.iter_batched_ref(
            || (x0_f64.clone(), y0_f64.clone()),
            |(x, y)| {
                dswap(
                    black_box(n),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dswap", |b| {
        b.iter_batched_ref(
            || (x0_f64.clone(), y0_f64.clone()),
            |(x, y)| {
                unsafe {
                    cblas_dswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(inc),
                        black_box(y.as_mut_ptr()),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cswap", |b| {
        b.iter_batched_ref(
            || (x0_c32.clone(), y0_c32.clone()),
            |(x, y)| {
                cswap(
                    black_box(n),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cswap", |b| {
        b.iter_batched_ref(
            || (x0_c32.clone(), y0_c32.clone()),
            |(x, y)| {
                unsafe {
                    cblas_cswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(inc),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(inc),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zswap", |b| {
        b.iter_batched_ref(
            || (x0_c64.clone(), y0_c64.clone()),
            |(x, y)| {
                zswap(
                    black_box(n),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zswap", |b| {
        b.iter_batched_ref(
            || (x0_c64.clone(), y0_c64.clone()),
            |(x, y)| {
                unsafe {
                    cblas_zswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr() as *mut _),
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

criterion_group!(benches, bench_swap);
criterion_main!(benches);

