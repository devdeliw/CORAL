use blas_src as _; 
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use coral::level3::dgemm_nn::dgemm_nn;

#[inline(always)]
fn make_matrix_colmajor(m: usize, n: usize, ld: usize, fill: f64) -> Vec<f64> {
    assert!(ld >= m);
    let mut a = vec![0.0f64; ld * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * ld] = fill;
        }
    }
    a
}

pub fn bench_dgemm_nn_fixed_1024(c: &mut Criterion) {
    let n: usize = 1024;
    let (m, k) = (n, n);

    let lda = m;
    let ldb = k;
    let ldc = m;

    let alpha: f64 = 1.000123;
    let beta:  f64 = 0.000321;

    let a  = make_matrix_colmajor(m, k, lda, 1.0);
    let b  = make_matrix_colmajor(k, n, ldb, 1.0);
    let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

    // coral
    c.bench_function("coral_dgemm_nn", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| {
                dgemm_nn(
                    black_box(m),
                    black_box(n),
                    black_box(k),
                    black_box(alpha),
                    black_box(a.as_ptr()),
                    black_box(lda),
                    black_box(b.as_ptr()),
                    black_box(ldb),
                    black_box(beta),
                    black_box(c_buf.as_mut_ptr()),
                    black_box(ldc),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_dgemm_nn", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| unsafe {
                cblas_dgemm(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(k as i32),
                    black_box(alpha),
                    black_box(a.as_ptr()),
                    black_box(lda as i32),
                    black_box(b.as_ptr()),
                    black_box(ldb as i32),
                    black_box(beta),
                    black_box(c_buf.as_mut_ptr()),
                    black_box(ldc as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_dgemm_nn_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("dgemm_nn");

    for &n in &sizes {
        let (m, k) = (n, n);
        let lda = m;
        let ldb = k;
        let ldc = m;

        let alpha: f64 = 1.000123;
        let beta:  f64 = 0.000321;

        let a  = make_matrix_colmajor(m, k, lda, 1.0);
        let b  = make_matrix_colmajor(k, n, ldb, 1.0);
        let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

        // coral
        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| {
                    dgemm_nn(
                        black_box(m),
                        black_box(n),
                        black_box(k),
                        black_box(alpha),
                        black_box(a.as_ptr()),
                        black_box(lda),
                        black_box(b.as_ptr()),
                        black_box(ldb),
                        black_box(beta),
                        black_box(c_buf.as_mut_ptr()),
                        black_box(ldc),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        // cblas
        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| unsafe {
                    cblas_dgemm(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(k as i32),
                        black_box(alpha),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(b.as_ptr()),
                        black_box(ldb as i32),
                        black_box(beta),
                        black_box(c_buf.as_mut_ptr()),
                        black_box(ldc as i32),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dgemm_nn_fixed_1024, bench_dgemm_nn_sweep);
criterion_main!(benches);

