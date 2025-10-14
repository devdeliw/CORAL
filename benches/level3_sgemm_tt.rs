use blas_src as _;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use coral::level3::sgemm::sgemm;
use coral::enums::CoralTranspose;

#[inline(always)]
fn make_matrix_colmajor(m: usize, n: usize, ld: usize, fill: f32) -> Vec<f32> {
    assert!(ld >= m);
    let mut a = vec![0.0f32; ld * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * ld] = fill;
        }
    }
    a
}

pub fn bench_sgemm_tt_fixed(c: &mut Criterion) {
    let n: usize = 1024;
    let (m, k) = (n, n);

    let lda = k;
    let ldb = n;
    let ldc = m;

    let alpha: f32 = 1.000123;
    let beta:  f32 = 0.000321;

    let a  = make_matrix_colmajor(k, m, lda, 1.0);
    let b  = make_matrix_colmajor(n, k, ldb, 1.0);
    let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

    c.bench_function("coral_sgemm_tt", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| {
                sgemm(
                    CoralTranspose::Transpose,
                    CoralTranspose::Transpose,
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

    c.bench_function("blas_sgemm_tt", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| unsafe {
                cblas_sgemm(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
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

pub fn bench_sgemm_tt_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("sgemm_tt");

    for &n in &sizes {
        let (m, k) = (n, n);
        let lda = k;
        let ldb = n;
        let ldc = m;

        let alpha: f32 = 1.000123;
        let beta:  f32 = 0.000321;

        let a  = make_matrix_colmajor(k, m, lda, 1.0);
        let b  = make_matrix_colmajor(n, k, ldb, 1.0);
        let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| {
                    sgemm(
                        CoralTranspose::Transpose,
                        CoralTranspose::Transpose,
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

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| unsafe {
                    cblas_sgemm(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
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

criterion_group!(benches, bench_sgemm_tt_fixed, bench_sgemm_tt_sweep);
criterion_main!(benches);

