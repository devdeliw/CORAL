use rusty_blas::level1::{
    sasum::sasum, 
    dasum::dasum, 
    scasum::scasum,
    dzasum::dzasum, 
};
use cblas_sys::{cblas_sasum, cblas_dasum, cblas_scasum, cblas_dzasum};
use criterion::{criterion_group, black_box, criterion_main, Criterion};

fn bench_asum(c: &mut Criterion) {
    let n: usize = 10_000;

    let data_f32: Vec<f32> = vec![1.0; n];
    let data_f64: Vec<f64> = vec![1.0; n];

    let data_c32: Vec<f32> = vec![1.0; 2 * n];
    let data_c64: Vec<f64> = vec![1.0; 2 * n];

    c.bench_function("rusty_sasum", |b| {
        b.iter(|| {
            let s = sasum(n, &data_f32, 1);
            black_box(s);
        })
    });
    c.bench_function("cblas_sasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_sasum(
                    n as i32,
                    data_f32.as_ptr(),
                    1,
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_dasum", |b| {
        b.iter(|| {
            let s = dasum(n, &data_f64, 1);
            black_box(s);
        })
    });
    c.bench_function("cblas_dasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dasum(
                    n as i32,
                    data_f64.as_ptr(),
                    1,
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_scasum", |b| {
        b.iter(|| {
            let s = scasum(n, &data_c32, 1);
            black_box(s);
        })
    });
    c.bench_function("cblas_scasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_scasum(
                    n as i32,
                    data_c32.as_ptr() as *const _,
                    1,
                )
            };
            black_box(s);
        })
    });

    c.bench_function("rusty_dzasum", |b| {
        b.iter(|| {
            let s = dzasum(n, &data_c64, 1);
            black_box(s);
        })
    });
    c.bench_function("cblas_dzasum", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dzasum(
                    n as i32,
                    data_c64.as_ptr() as *const _,
                    1,
                )
            };
            black_box(s);
        })
    });
}

criterion_group!(benches, bench_asum);
criterion_main!(benches);

