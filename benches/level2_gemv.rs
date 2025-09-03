use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};

use rusty_blas::level2::sgemv::sgemv;

use cblas_sys::{cblas_sgemv, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

fn bench_gemv(c: &mut Criterion) {
    let m: usize = 1024;
    let n: usize = 768;

    let lda: usize = m + 8;

    let alpha: f32 = 1.000123_f32;
    let beta:  f32 = 0.000321_f32;

    c.bench_function("rusty_sgemv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                for v in &mut a { *v = 1.0; }

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                let mut y = vec![0.0f32; m];
                for v in &mut y { *v = 2.0; }

                (a, x, y)
            },
            |(a, x, mut y)| {
                sgemv(
                    black_box(m),
                    black_box(n),
                    black_box(alpha),
                    black_box(&a),
                    black_box(1isize),            
                    black_box(lda as isize),      
                    black_box(&x),
                    black_box(1isize),            
                    black_box(beta),
                    black_box(&mut y),
                    black_box(1isize),            
                );
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_sgemv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                for v in &mut a { *v = 1.0; }

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                let mut y = vec![0.0f32; m];
                for v in &mut y { *v = 2.0; }

                (a, x, y)
            },
            |(a, x, mut y)| {
                unsafe {
                    cblas_sgemv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_ptr()),
                        black_box(1i32), 
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1i32), 
                    );
                }
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_gemv);
criterion_main!(benches);
