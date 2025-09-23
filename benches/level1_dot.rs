use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level1::{
    sdot::sdot,
    ddot::ddot,
    cdotu::cdotu,
    cdotc::cdotc,
    zdotu::zdotu,
    zdotc::zdotc,
};

use cblas_sys::{
    cblas_sdot,
    cblas_ddot,
    cblas_cdotu_sub,
    cblas_cdotc_sub,
    cblas_zdotu_sub,
    cblas_zdotc_sub,
};

pub fn bench_dot(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    // real
    let xr_f32 = vec![1.0f32; n];
    let yr_f32 = vec![1.0f32; n];
    let xr_f64 = vec![1.0f64; n];
    let yr_f64 = vec![1.0f64; n];

    // complex
    let mut xc_f32 = vec![0.0f32; 2 * n];
    let mut yc_f32 = vec![0.0f32; 2 * n];
    let mut xc_f64 = vec![0.0f64; 2 * n];
    let mut yc_f64 = vec![0.0f64; 2 * n];
    for i in 0..n {
        xc_f32[2 * i]     =  1.0;
        xc_f32[2 * i + 1] = -1.0;
        yc_f32[2 * i]     =  1.0;
        yc_f32[2 * i + 1] =  1.0;

        xc_f64[2 * i]     =  1.0;
        xc_f64[2 * i + 1] = -1.0;
        yc_f64[2 * i]     =  1.0;
        yc_f64[2 * i + 1] =  1.0;
    }

    c.bench_function("coral_sdot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = sdot(
                    black_box(n),
                    black_box(&xr_f32),
                    black_box(1usize),
                    black_box(&yr_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sdot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_sdot(
                        black_box(n as i32),
                        black_box(xr_f32.as_ptr()),
                        black_box(inc),
                        black_box(yr_f32.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_ddot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = ddot(
                    black_box(n),
                    black_box(&xr_f64),
                    black_box(1usize),
                    black_box(&yr_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ddot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_ddot(
                        black_box(n as i32),
                        black_box(xr_f64.as_ptr()),
                        black_box(inc),
                        black_box(yr_f64.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = cdotu(
                    black_box(n),
                    black_box(&xc_f32),
                    black_box(1usize),
                    black_box(&yc_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f32; 2];
                unsafe {
                    cblas_cdotu_sub(
                        black_box(n as i32),
                        black_box(xc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = cdotc(
                    black_box(n),
                    black_box(&xc_f32),
                    black_box(1usize),
                    black_box(&yc_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f32; 2];
                unsafe {
                    cblas_cdotc_sub(
                        black_box(n as i32),
                        black_box(xc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = zdotu(
                    black_box(n),
                    black_box(&xc_f64),
                    black_box(1usize),
                    black_box(&yc_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f64; 2];
                unsafe {
                    cblas_zdotu_sub(
                        black_box(n as i32),
                        black_box(xc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = zdotc(
                    black_box(n),
                    black_box(&xc_f64),
                    black_box(1usize),
                    black_box(&yc_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f64; 2];
                unsafe {
                    cblas_zdotc_sub(
                        black_box(n as i32),
                        black_box(xc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_dot);
criterion_main!(benches);

