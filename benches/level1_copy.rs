use rusty_blas::level1::{
    scopy::scopy, 
    dcopy::dcopy, 
    ccopy::ccopy, 
    zcopy::zcopy, 
};
use cblas_sys::{
    cblas_scopy, 
    cblas_dcopy, 
    cblas_ccopy, 
    cblas_zcopy
};
use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn bench_copy(c: &mut Criterion) {
    let n: usize = 10_000;

    let data_f32: Vec<f32> = vec![1.0; n];
    let data_f64: Vec<f64> = vec![1.0; n];
    let mut out_f32: Vec<f32> = vec![0.0; n];
    let mut out_f64: Vec<f64> = vec![0.0; n];

    let data_c32: Vec<f32> = vec![1.0; 2 * n];
    let data_c64: Vec<f64> = vec![1.0; 2 * n];
    let mut out_c32: Vec<f32> = vec![0.0; 2 * n];
    let mut out_c64: Vec<f64> = vec![0.0; 2 * n];

    c.bench_function("rusty_scopy", |b| {
        b.iter(|| {
            scopy(
                black_box(n),
                black_box(&data_f32),
                black_box(1),
                black_box(&mut out_f32),
                black_box(1),
            );
            black_box(&out_f32);
        })
    });

    c.bench_function("cblas_scopy", |b| {
        b.iter(|| {
            unsafe {
                cblas_scopy(
                    black_box(n as i32),
                    black_box(data_f32.as_ptr()),
                    black_box(1),
                    black_box(out_f32.as_mut_ptr()),
                    black_box(1),
                );
            }
            black_box(&out_f32);
        })
    });

    c.bench_function("rusty_dcopy", |b| {
        b.iter(|| {
            dcopy(
                black_box(n),
                black_box(&data_f64),
                black_box(1),
                black_box(&mut out_f64),
                black_box(1),
            );
            black_box(&out_f64);
        })
    });

    c.bench_function("cblas_dcopy", |b| {
        b.iter(|| {
            unsafe {
                cblas_dcopy(
                    black_box(n as i32),
                    black_box(data_f64.as_ptr()),
                    black_box(1),
                    black_box(out_f64.as_mut_ptr()),
                    black_box(1),
                );
            }
            black_box(&out_f64);
        })
    });

    c.bench_function("rusty_ccopy", |b| {
        b.iter(|| {
            ccopy(
                black_box(n),
                black_box(&data_c32),
                black_box(1),
                black_box(&mut out_c32),
                black_box(1),
            );
            black_box(&out_c32);
        })
    });

    c.bench_function("cblas_ccopy", |b| {
        b.iter(|| {
            unsafe {
                cblas_ccopy(
                    black_box(n as i32),
                    black_box(data_c32.as_ptr() as *const _),
                    black_box(1),
                    black_box(out_c32.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            }
            black_box(&out_c32);
        })
    });

    c.bench_function("rusty_zcopy", |b| {
        b.iter(|| {
            zcopy(
                black_box(n),
                black_box(&data_c64),
                black_box(1),
                black_box(&mut out_c64),
                black_box(1),
            );
            black_box(&out_c64);
        })
    });

    c.bench_function("cblas_zcopy", |b| {
        b.iter(|| {
            unsafe {
                cblas_zcopy(
                    black_box(n as i32),
                    black_box(data_c64.as_ptr() as *const _),
                    black_box(1),
                    black_box(out_c64.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            }
            black_box(&out_c64);
        })
    });
}

criterion_group!(benches, bench_copy);
criterion_main!(benches);

