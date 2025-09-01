use rusty_blas::level1::{
    sasum::sasum, 
    dasum::dasum, 
    scasum::scasum,
    dzasum::dzasum, 
};
use cblas_sys::{
    cblas_sasum, 
    cblas_dasum, 
    cblas_scasum, 
    cblas_dzasum
};
use criterion::{criterion_group, black_box, criterion_main, Criterion};

fn bench_asum(c: &mut Criterion) {
    let n: usize = 10_000;

    let data_f32: Vec<f32> = vec![1.0; n];
    let data_f64: Vec<f64> = vec![1.0; n];

    let data_c32: Vec<f32> = vec![1.0; 2 * n];
    let data_c64: Vec<f64> = vec![1.0; 2 * n];

    c.bench_function("rusty_sasum", |b| {
        b.iter(|| {
            let s = sasum(
                black_box(n),
                black_box(&data_f32),
                black_box(1),
            );
            black_box(s);
        })
    });

    c.bench_function("cblas_sasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_sasum(
                    black_box(n as i32),
                    black_box(data_f32.as_ptr()),
                    black_box(1),
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_dasum", |b| {
        b.iter(|| {
            let s = dasum(
                black_box(n),
                black_box(&data_f64),
                black_box(1),
            );
            black_box(s);
        })
    });

    c.bench_function("cblas_dasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dasum(
                    black_box(n as i32),
                    black_box(data_f64.as_ptr()),
                    black_box(1),
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_scasum", |b| {
        b.iter(|| {
            let s = scasum(
                black_box(n),
                black_box(&data_c32),
                black_box(1),
            );
            black_box(s);
        })
    });

    c.bench_function("cblas_scasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_scasum(
                    black_box(n as i32),
                    black_box(data_c32.as_ptr() as *const _),
                    black_box(1),
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_dzasum", |b| {
        b.iter(|| {
            let s = dzasum(
                black_box(n),
                black_box(&data_c64),
                black_box(1),
            );
            black_box(s);
        })
    });

    c.bench_function("cblas_dzasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dzasum(
                    black_box(n as i32),
                    black_box(data_c64.as_ptr() as *const _),
                    black_box(1),
                )
            };
            black_box(s);
        })
    });
}

criterion_group!(benches, bench_asum);
criterion_main!(benches);

