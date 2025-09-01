use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cblas_sys::{
    cblas_ddot,
    cblas_sdot, 
    cblas_cdotu_sub,
    cblas_cdotc_sub, 
    cblas_zdotu_sub,
    cblas_zdotc_sub,
};
use rusty_blas::level1::{
    sdot::sdot,
    ddot::ddot,
    cdotu::cdotu,
    cdotc::cdotc,
    zdotu::zdotu, 
    zdotc::zdotc,
};

fn bench_dot(c: &mut Criterion) {
    let n_real = 1_000_000usize;
    let n_cplx = 500_000usize;

    let x_f32: Vec<f32> = (0..n_real)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_f32: Vec<f32> = (0..n_real)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let x_f64: Vec<f64> = (0..n_real)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_f64: Vec<f64> = (0..n_real)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();

    let x_c32: Vec<f32> = {
        let mut v = vec![0.0f32; 2 * n_cplx];
        for i in 0..n_cplx {
            v[2 * i] = (i as f32) * 0.01 - 0.3;
            v[2 * i + 1] = (i % 5) as f32 * 0.02 - 0.05;
        }
        v
    };
    let y_c32: Vec<f32> = {
        let mut v = vec![0.0f32; 2 * n_cplx];
        for i in 0..n_cplx {
            v[2 * i] = 1.0 + (i % 7) as f32 * 0.03;
            v[2 * i + 1] = if i % 3 == 0 { -0.2 } else { 0.1 };
        }
        v
    };

    let x_c64: Vec<f64> = {
        let mut v = vec![0.0f64; 2 * n_cplx];
        for i in 0..n_cplx {
            v[2 * i] = (i as f64) * 0.01 - 0.3;
            v[2 * i + 1] = (i % 5) as f64 * 0.02 - 0.05;
        }
        v
    };
    let y_c64: Vec<f64> = {
        let mut v = vec![0.0f64; 2 * n_cplx];
        for i in 0..n_cplx {
            v[2 * i] = 1.0 + (i % 7) as f64 * 0.03;
            v[2 * i + 1] = if i % 3 == 0 { -0.2 } else { 0.1 };
        }
        v
    };

    c.bench_function("rusty_sdot", |b| {
        b.iter(|| {
            black_box(sdot(
                black_box(n_real),
                black_box(&x_f32),
                black_box(1isize),
                black_box(&y_f32),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_sdot", |b| {
        b.iter(|| unsafe {
            black_box(cblas_sdot(
                black_box(n_real as i32),
                black_box(x_f32.as_ptr()),
                black_box(1i32),
                black_box(y_f32.as_ptr()),
                black_box(1i32),
            ))
        })
    });

    c.bench_function("rusty_ddot", |b| {
        b.iter(|| {
            black_box(ddot(
                black_box(n_real),
                black_box(&x_f64),
                black_box(1isize),
                black_box(&y_f64),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_ddot", |b| {
        b.iter(|| unsafe {
            black_box(cblas_ddot(
                black_box(n_real as i32),
                black_box(x_f64.as_ptr()),
                black_box(1i32),
                black_box(y_f64.as_ptr()),
                black_box(1i32),
            ))
        })
    });

    c.bench_function("rusty_cdotu", |b| {
        b.iter(|| {
            black_box(cdotu(
                black_box(n_cplx),
                black_box(&x_c32),
                black_box(1isize),
                black_box(&y_c32),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_cdotu", |b| {
        b.iter(|| unsafe {
            let mut out: [f32; 2] = [0.0, 0.0];
            cblas_cdotu_sub(
                black_box(n_cplx as i32),
                black_box(x_c32.as_ptr() as *const [f32; 2]),
                black_box(1i32),
                black_box(y_c32.as_ptr() as *const [f32; 2]),
                black_box(1i32),
                black_box(&mut out as *mut [f32; 2]),
            );
            black_box(out)
        })
    });

    c.bench_function("rusty_cdotc", |b| {
        b.iter(|| {
            black_box(cdotc(
                black_box(n_cplx),
                black_box(&x_c32),
                black_box(1isize),
                black_box(&y_c32),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_cdotc", |b| {
        b.iter(|| unsafe {
            let mut out: [f32; 2] = [0.0, 0.0];
            cblas_cdotc_sub(
                black_box(n_cplx as i32),
                black_box(x_c32.as_ptr() as *const [f32; 2]),
                black_box(1i32),
                black_box(y_c32.as_ptr() as *const [f32; 2]),
                black_box(1i32),
                black_box(&mut out as *mut [f32; 2]),
            );
            black_box(out)
        })
    });

    c.bench_function("rusty_zdotu", |b| {
        b.iter(|| {
            black_box(zdotu(
                black_box(n_cplx),
                black_box(&x_c64),
                black_box(1isize),
                black_box(&y_c64),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_zdotu", |b| {
        b.iter(|| unsafe {
            let mut out: [f64; 2] = [0.0, 0.0];
            cblas_zdotu_sub(
                black_box(n_cplx as i32),
                black_box(x_c64.as_ptr() as *const [f64; 2]),
                black_box(1i32),
                black_box(y_c64.as_ptr() as *const [f64; 2]),
                black_box(1i32),
                black_box(&mut out as *mut [f64; 2]),
            );
            black_box(out)
        })
    });

    c.bench_function("rusty_zdotc", |b| {
        b.iter(|| {
            black_box(zdotc(
                black_box(n_cplx),
                black_box(&x_c64),
                black_box(1isize),
                black_box(&y_c64),
                black_box(1isize),
            ))
        })
    });

    c.bench_function("cblas_zdotc", |b| {
        b.iter(|| unsafe {
            let mut out: [f64; 2] = [0.0, 0.0];
            cblas_zdotc_sub(
                black_box(n_cplx as i32),
                black_box(x_c64.as_ptr() as *const [f64; 2]),
                black_box(1i32),
                black_box(y_c64.as_ptr() as *const [f64; 2]),
                black_box(1i32),
                black_box(&mut out as *mut [f64; 2]),
            );
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_dot);
criterion_main!(benches);

