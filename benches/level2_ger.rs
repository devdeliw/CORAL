use cblas_sys::{CBLAS_LAYOUT, cblas_sger, cblas_dger};
use rusty_blas::level2::{sger::sger, dger::dger};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

fn bench_ger(c: &mut Criterion) {
    let m: usize = 1024;
    let n: usize = 1024;
    let lda: usize = m + 8;

    let alpha_f32: f32 = 1.000123_f32;
    let alpha_f64: f64 = 1.000123_f64;

    c.bench_function("rusty_sger", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                for j in 0..n {
                    for i in 0..m {
                        a[j * lda + i] = 0.0;
                    }
                }

                let mut x = vec![0.0f32; m];
                for i in 0..m {
                    x[i] = (i as f32).mul_add(0.001, -0.5);
                }

                let mut y = vec![0.0f32; n];
                for j in 0..n {
                    y[j] = (j as f32).mul_add(0.002, 0.25);
                }

                (a, x, y)
            },
            |(mut a, x, y)| {
                sger(
                    black_box(m), black_box(n),
                    black_box(alpha_f32),
                    &x, 1,
                    &y, 1,
                    &mut a, 1, lda as isize,
                );
                black_box(&a);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_sger", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                for j in 0..n {
                    for i in 0..m {
                        a[j * lda + i] = 0.0;
                    }
                }

                let mut x = vec![0.0f32; m];
                for i in 0..m {
                    x[i] = (i as f32).mul_add(0.001, -0.5);
                }

                let mut y = vec![0.0f32; n];
                for j in 0..n {
                    y[j] = (j as f32).mul_add(0.002, 0.25);
                }

                (a, x, y)
            },
            |(mut a, x, y)| {
                unsafe {
                    cblas_sger(
                        CBLAS_LAYOUT::CblasColMajor,
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha_f32),
                        x.as_ptr(), 1,
                        y.as_ptr(), 1,
                        a.as_mut_ptr(), lda as i32,
                    );
                }
                black_box(&a);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_dger", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                for j in 0..n {
                    for i in 0..m {
                        a[j * lda + i] = 0.0;
                    }
                }

                let mut x = vec![0.0f64; m];
                for i in 0..m {
                    x[i] = (i as f64).mul_add(0.001, -0.5);
                }

                let mut y = vec![0.0f64; n];
                for j in 0..n {
                    y[j] = (j as f64).mul_add(0.002, 0.25);
                }

                (a, x, y)
            },
            |(mut a, x, y)| {
                dger(
                    black_box(m), black_box(n),
                    black_box(alpha_f64),
                    &x, 1,
                    &y, 1,
                    &mut a, 1, lda as isize,
                );
                black_box(&a);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_dger", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                for j in 0..n {
                    for i in 0..m {
                        a[j * lda + i] = 0.0;
                    }
                }

                let mut x = vec![0.0f64; m];
                for i in 0..m {
                    x[i] = (i as f64).mul_add(0.001, -0.5);
                }

                let mut y = vec![0.0f64; n];
                for j in 0..n {
                    y[j] = (j as f64).mul_add(0.002, 0.25);
                }

                (a, x, y)
            },
            |(mut a, x, y)| {
                unsafe {
                    cblas_dger(
                        CBLAS_LAYOUT::CblasColMajor,
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha_f64),
                        x.as_ptr(), 1,
                        y.as_ptr(), 1,
                        a.as_mut_ptr(), lda as i32,
                    );
                }
                black_box(&a);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_ger);
criterion_main!(benches);

