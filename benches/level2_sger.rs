use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box};

use coral::level2::sger::sger;

use cblas_sys::{
    CBLAS_LAYOUT,
    cblas_sger,
};

pub fn bench_sger(c: &mut Criterion) {
    let m:   usize = 1024;
    let n:   usize = 1024;
    let lda: usize = m;

    let alpha: f32 = 1.000123;

    // start from zeros; 
    // just alpha * x * y^T 
    let a0 = vec![0.0f32; lda * n];

    let x = vec![1.0f32; m]; 
    let y = vec![1.0f32; n]; 

    c.bench_function("coral_sger", |b| {
        b.iter_batched_ref(
            || a0.clone(), 
            |a| {
                sger(
                    black_box(m),
                    black_box(n),
                    black_box(alpha),
                    black_box(&x),
                    black_box(1),
                    black_box(&y),
                    black_box(1),
                    black_box(a.as_mut_slice()),
                    black_box(lda),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cblas_sger", |b| {
        b.iter_batched_ref(
            || a0.clone(),
            |a| unsafe {
                cblas_sger(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(y.as_ptr()),
                    black_box(1),
                    black_box(a.as_mut_ptr()),
                    black_box(lda as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_sger);
criterion_main!(benches);

