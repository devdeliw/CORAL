use rusty_blas::level1::nrm2::{snrm2, dnrm2, scnrm2, dznrm2};
use cblas_sys::{cblas_snrm2, cblas_dnrm2, cblas_scnrm2, cblas_dznrm2};
use criterion::{criterion_group, criterion_main, Criterion, black_box};


fn bench_nrm2(c: &mut Criterion) {
    let n: usize           = 10_000;
    let data_f32: Vec<f32> = vec![1.0; n];
    let data_f64: Vec<f64> = vec![1.0; n];

    let data_cf32: Vec<f32> = vec![1.0; 2 * n];
    let data_cf64: Vec<f64> = vec![1.0; 2 * n];

    // single precision
    c.bench_function("my_snrm2_f32", |b| {
        b.iter(|| {
            let s = snrm2(
                black_box(n as usize),
                black_box(&data_f32),
                black_box(1 as isize),
            );
            black_box(s);
        })
    });
    c.bench_function("cblas_snrm2", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_snrm2(
                    black_box(n as i32),
                    black_box(data_f32.as_ptr()),
                    black_box(1 as i32),
                )
            };
            black_box(s);
        })
    });

    // double precision 
    c.bench_function("my_dnrm2_f64", |b| {
        b.iter(|| {
            let s = dnrm2(
                black_box(n as usize),
                black_box(&data_f64),
                black_box(1 as isize),
            );
            black_box(s);
        })
    });
    c.bench_function("cblas_dnrm2", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dnrm2(
                    black_box(n as i32),
                    black_box(data_f64.as_ptr()),
                    black_box(1 as i32),
                )
            };
            black_box(s);
        })
    });

    // complex single precision 
    c.bench_function("my_scnrm2_cf32", |b| {
        b.iter(|| {
            let s = scnrm2(
                black_box(n as usize),
                black_box(&data_cf32),
                black_box(1 as isize),
            );
            black_box(s);
        })
    });
    c.bench_function("cblas_scnrm2", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_scnrm2(
                    black_box(n as i32),
                    black_box(data_cf32.as_ptr().cast::<[f32; 2]>()),
                    black_box(1 as i32),
                )
            };
            black_box(s);
        })
    });

    // complex double precision 
    c.bench_function("my_dznrm2_c64", |b| {
        b.iter(|| {
            let s = dznrm2(
                black_box(n as usize),
                black_box(&data_cf64),
                black_box(1 as isize),
            );
            black_box(s);
        })
    });
    c.bench_function("cblas_dznrm2", |b| {
        b.iter(|| {
            let s = unsafe {
                cblas_dznrm2(
                    black_box(n as i32),
                    black_box(data_cf64.as_ptr().cast::<[f64; 2]>()),
                    black_box(1 as i32),
                )
            };
            black_box(s);
        })
    });
}

criterion_group!(benches, bench_nrm2);
criterion_main!(benches);

