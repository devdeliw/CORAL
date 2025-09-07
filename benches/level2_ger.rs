use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use cblas_sys::{CBLAS_LAYOUT, cblas_sger, cblas_dger};
use rusty_blas::level2::{sger::sger, dger::dger};

fn bench_ger(c: &mut Criterion) {
    let m: usize = 1024;
    let n: usize = 1024;
    let lda: usize = m + 8; 

    let alpha_f32: f32 = 1.000123_f32;
    let alpha_f64: f64 = 1.000123_f64;

    let a32_zero = vec![0.0f32; lda * n];
    let a64_zero = vec![0.0f64; lda * n];

    let x32: Vec<f32> = (0..m).map(|i| (i as f32).mul_add(0.001, -0.5)).collect();
    let y32: Vec<f32> = (0..n).map(|j| (j as f32).mul_add(0.002, 0.25)).collect();

    let x64: Vec<f64> = (0..m).map(|i| (i as f64).mul_add(0.001, -0.5)).collect();
    let y64: Vec<f64> = (0..n).map(|j| (j as f64).mul_add(0.002, 0.25)).collect();

    c.bench_function("rusty_sger", |b| {
        let a0 = a32_zero.clone();
        let mut sign = alpha_f32;
        b.iter_batched_ref(
            || { let s = sign; sign = -sign; (a0.clone(), s) },
            |state| {
                let (a, s) = state;
                sger(
                    m, n,
                    *s,
                    &x32, 1,
                    &y32, 1,
                    a.as_mut_slice(), 1, lda as isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_sger", |b| {
        let a0 = a32_zero.clone();
        let mut sign = alpha_f32;
        b.iter_batched_ref(
            || { let s = sign; sign = -sign; (a0.clone(), s) },
            |state| unsafe {
                let (a, s) = state;
                cblas_sger(
                    CBLAS_LAYOUT::CblasColMajor,
                    m as i32, n as i32,
                    *s,
                    x32.as_ptr(), 1,
                    y32.as_ptr(), 1,
                    a.as_mut_ptr(), lda as i32,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("rusty_dger", |b| {
        let a0 = a64_zero.clone();
        let mut sign = alpha_f64;
        b.iter_batched_ref(
            || { let s = sign; sign = -sign; (a0.clone(), s) },
            |state| {
                let (a, s) = state;
                dger(
                    m, n,
                    *s,
                    &x64, 1,
                    &y64, 1,
                    a.as_mut_slice(), 1, lda as isize,
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_dger", |b| {
        let a0 = a64_zero.clone();
        let mut sign = alpha_f64;
        b.iter_batched_ref(
            || { let s = sign; sign = -sign; (a0.clone(), s) },
            |state| unsafe {
                let (a, s) = state;
                cblas_dger(
                    CBLAS_LAYOUT::CblasColMajor,
                    m as i32, n as i32,
                    *s,
                    x64.as_ptr(), 1,
                    y64.as_ptr(), 1,
                    a.as_mut_ptr(), lda as i32,
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_ger);
criterion_main!(benches);

