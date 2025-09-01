use rusty_blas::level1::{ 
    isamax::isamax, 
    idamax::idamax, 
    icamax::icamax, 
    izamax::izamax,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cblas_sys::{
    cblas_isamax, 
    cblas_idamax, 
    cblas_icamax, 
    cblas_izamax
};

fn bench_iamax(c: &mut Criterion) {
    let n: usize = 10_000;

    let mut data_f32: Vec<f32> = vec![1.0; n];
    data_f32[n - 1] = 3.0;

    let mut data_f64: Vec<f64> = vec![1.0; n];
    data_f64[n - 1] = 3.0;

    let mut data_c32: Vec<f32> = vec![1.0; 2 * n];
    data_c32[2 * (n - 1)] = 3.0; 

    let mut data_c64: Vec<f64> = vec![1.0; 2 * n];
    data_c64[2 * (n - 1)] = 3.0; 

    c.bench_function("rusty_isamax", |b| {
        b.iter(|| {
            let idx = isamax(
                black_box(n),
                black_box(&data_f32),
                black_box(1isize),
            );
            black_box(idx);
        })
    });

    c.bench_function("cblas_isamax", |b| {
        b.iter(|| {
            let idx = unsafe { 
                cblas_isamax(
                    black_box(n as i32),
                    black_box(data_f32.as_ptr()),
                    black_box(1),
                ) 
            };
            black_box(idx);
        })
    });

    c.bench_function("rusty_idamax", |b| {
        b.iter(|| {
            let idx = idamax(
                black_box(n),
                black_box(&data_f64),
                black_box(1isize),
            );
            black_box(idx);
        })
    });

    c.bench_function("cblas_idamax", |b| {
        b.iter(|| {
            let idx = unsafe { 
                cblas_idamax(
                    black_box(n as i32),
                    black_box(data_f64.as_ptr()),
                    black_box(1),
                ) 
            };
            black_box(idx);
        })
    });

    c.bench_function("rusty_icamax", |b| {
        b.iter(|| {
            let idx = icamax(
                black_box(n),
                black_box(&data_c32),
                black_box(1isize),
            );
            black_box(idx);
        })
    });

    c.bench_function("cblas_icamax", |b| {
        b.iter(|| {
            let idx = unsafe {
                cblas_icamax(
                    black_box(n as i32),
                    black_box(data_c32.as_ptr() as *const [f32; 2]),
                    black_box(1),
                )
            };
            black_box(idx);
        })
    });

    c.bench_function("rusty_izamax", |b| {
        b.iter(|| {
            let idx = izamax(
                black_box(n),
                black_box(&data_c64),
                black_box(1isize),
            );
            black_box(idx);
        })
    });

    c.bench_function("cblas_izamax", |b| {
        b.iter(|| {
            let idx = unsafe {
                cblas_izamax(
                    black_box(n as i32),
                    black_box(data_c64.as_ptr() as *const [f64; 2]),
                    black_box(1),
                )
            };
            black_box(idx);
        })
    });
}

criterion_group!(benches, bench_iamax);
criterion_main!(benches);

