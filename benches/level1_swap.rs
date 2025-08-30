use rusty_blas::level1::{ 
    sswap::sswap, 
    dswap::dswap, 
    cswap::cswap, 
    zswap::zswap, 
}; 
use cblas_sys::{cblas_sswap, cblas_dswap, cblas_cswap, cblas_zswap};
use criterion::{criterion_group, criterion_main, Criterion, black_box, BatchSize};

fn bench_swap(c: &mut Criterion) {
    let n: usize = 10_000;

    c.bench_function("rusty_sswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let y: Vec<f32> = (0..n).map(|i| -(i as isize) as f32).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                sswap(
                    black_box(n),
                    black_box(&mut x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_sswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let y: Vec<f32> = (0..n).map(|i| -(i as isize) as f32).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                unsafe {
                    cblas_sswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                }
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_dswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let y: Vec<f64> = (0..n).map(|i| -(i as isize) as f64).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                dswap(
                    black_box(n),
                    black_box(&mut x),
                    black_box(1isize),
                    black_box(&mut y),
                    black_box(1isize),
                );
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_dswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let y: Vec<f64> = (0..n).map(|i| -(i as isize) as f64).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                unsafe {
                    cblas_dswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                }
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_cswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<[f32; 2]> = (0..n).map(|i| [i as f32, -(i as f32)]).collect();
                let y: Vec<[f32; 2]> = (0..n).map(|i| [-(i as f32), i as f32]).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                let x_flat: &mut [f32] = unsafe {
                    core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, 2 * n)
                };
                let y_flat: &mut [f32] = unsafe {
                    core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, 2 * n)
                };
                cswap(black_box(n), black_box(x_flat), 1, black_box(y_flat), 1);
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_cswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<[f32; 2]> = (0..n).map(|i| [i as f32, -(i as f32)]).collect();
                let y: Vec<[f32; 2]> = (0..n).map(|i| [-(i as f32), i as f32]).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                unsafe {
                    cblas_cswap(
                        black_box(n as i32), 
                        black_box(x.as_mut_ptr()), 
                        1, 
                        black_box(y.as_mut_ptr()), 
                        1
                    );
                }
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_zswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, -(i as f64)]).collect();
                let y: Vec<[f64; 2]> = (0..n).map(|i| [-(i as f64), i as f64]).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                let x_flat: &mut [f64] = unsafe {
                    core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, 2 * n)
                };
                let y_flat: &mut [f64] = unsafe {
                    core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, 2 * n)
                };
                zswap(black_box(n), black_box(x_flat), 1, black_box(y_flat), 1);
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_zswap", |b| {
        b.iter_batched(
            || {
                let x: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, -(i as f64)]).collect();
                let y: Vec<[f64; 2]> = (0..n).map(|i| [-(i as f64), i as f64]).collect();
                (x, y)
            },
            |(mut x, mut y)| {
                unsafe {
                    cblas_zswap(
                        black_box(n as i32), 
                        black_box(x.as_mut_ptr()), 
                        1, 
                        black_box(y.as_mut_ptr()), 
                        1
                    );
                }
                black_box((x, y));
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_swap);
criterion_main!(benches);

