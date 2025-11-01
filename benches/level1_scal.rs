use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId, black_box};

use coral::level1::{
    sscal,
    dscal,
    cscal,
    zscal,
    csscal,
    zdscal,
};

use cblas_sys::{
    cblas_sscal,
    cblas_dscal,
    cblas_cscal,
    cblas_zscal,
    cblas_csscal,
    cblas_zdscal,
};

pub fn bench_scal(c: &mut Criterion) {
    let n:   usize = 1_000_000;

    let alpha_f32: f32 = 1.000123;
    let alpha_f64: f64 = 1.000123;
    let alpha_c32: [f32; 2] = [1.000123, -0.999877];
    let alpha_c64: [f64; 2] = [1.000123, -0.999877];

    let x0_f32 = vec![1.0f32; n];
    let x0_f64 = vec![1.0f64; n];

    let mut x0_c32 = vec![0.0f32; 2 * n];
    let mut x0_c64 = vec![0.0f64; 2 * n];
    for i in 0..n {
        x0_c32[2 * i]     =  1.0;
        x0_c32[2 * i + 1] = -1.0;
        x0_c64[2 * i]     =  1.0;
        x0_c64[2 * i + 1] = -1.0;
    }

    c.bench_function("coral_sscal", |b| {
        b.iter_batched_ref(
            || x0_f32.clone(),
            |x| {
                sscal(
                    black_box(n),
                    black_box(alpha_f32),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sscal", |b| {
        b.iter_batched_ref(
            || x0_f32.clone(),
            |x| {
                unsafe {
                    cblas_sscal(
                        black_box(n as i32),
                        black_box(alpha_f32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_dscal", |b| {
        b.iter_batched_ref(
            || x0_f64.clone(),
            |x| {
                dscal(
                    black_box(n),
                    black_box(alpha_f64),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dscal", |b| {
        b.iter_batched_ref(
            || x0_f64.clone(),
            |x| {
                unsafe {
                    cblas_dscal(
                        black_box(n as i32),
                        black_box(alpha_f64),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cscal", |b| {
        b.iter_batched_ref(
            || x0_c32.clone(),
            |x| {
                cscal(
                    black_box(n),
                    black_box(alpha_c32),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cscal", |b| {
        b.iter_batched_ref(
            || x0_c32.clone(),
            |x| {
                unsafe {
                    cblas_cscal(
                        black_box(n as i32),
                        black_box(alpha_c32.as_ptr() as *const _),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zscal", |b| {
        b.iter_batched_ref(
            || x0_c64.clone(),
            |x| {
                zscal(
                    black_box(n),
                    black_box(alpha_c64),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zscal", |b| {
        b.iter_batched_ref(
            || x0_c64.clone(),
            |x| {
                unsafe {
                    cblas_zscal(
                        black_box(n as i32),
                        black_box(alpha_c64.as_ptr() as *const _),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_csscal", |b| {
        b.iter_batched_ref(
            || x0_c32.clone(),
            |x| {
                csscal(
                    black_box(n),
                    black_box(alpha_f32),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_csscal", |b| {
        b.iter_batched_ref(
            || x0_c32.clone(),
            |x| {
                unsafe {
                    cblas_csscal(
                        black_box(n as i32),
                        black_box(alpha_f32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zdscal", |b| {
        b.iter_batched_ref(
            || x0_c64.clone(),
            |x| {
                zdscal(
                    black_box(n),
                    black_box(alpha_f64),
                    black_box(x.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zdscal", |b| {
        b.iter_batched_ref(
            || x0_c64.clone(),
            |x| {
                unsafe {
                    cblas_zdscal(
                        black_box(n as i32),
                        black_box(alpha_f64),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sscal_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: f32 = 1.000123;

    let mut group = c.benchmark_group("sscal");

    for &n in &sizes {
        let x0 = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| {
                    sscal(
                        black_box(n),
                        black_box(alpha),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_sscal(
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_dscal_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: f64 = 1.000123_f64;

    let mut group = c.benchmark_group("dscal");

    for &n in &sizes {
        let x0 = vec![1.0f64; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| {
                    dscal(
                        black_box(n),
                        black_box(alpha),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_dscal(
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_cscal_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: [f32; 2] = [1.000123, -0.999877];

    let mut group = c.benchmark_group("cscal");

    for &n in &sizes {
        let mut x0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| {
                    cscal(
                        black_box(n),
                        black_box(alpha),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_cscal(
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_zscal_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: [f64; 2] = [1.000123_f64, -0.999877_f64];

    let mut group = c.benchmark_group("zscal");

    for &n in &sizes {
        let mut x0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| {
                    zscal(
                        black_box(n),
                        black_box(alpha),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_zscal(
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scal, bench_sscal_sweep, bench_dscal_sweep, bench_cscal_sweep, bench_zscal_sweep);
criterion_main!(benches);

