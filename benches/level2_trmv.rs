use cblas_sys::{
    CBLAS_DIAG,
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
    CBLAS_UPLO,
    cblas_strmv,
    cblas_dtrmv,
};
use rusty_blas::level2::{
    enums::{ Trans, Diag, UpLo },
    strmv::strmv,
    dtrmv::dtrmv,
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

// pseudo-random float in [-0.5, 0.5]
fn f32_from_u32_f32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

// pseudo-random double in [-0.5, 0.5]
fn f64_from_u32_f64(u: u32) -> f64 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f64 / 65536.0) - 0.5
}

fn fill_upper_colmajor_f32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        let base = j * lda;
        for i in 0..n {
            a[base + i] = f32_from_u32_f32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        let base = j * lda;
        for i in (j + 1)..n {
            a[base + i] = 0.0;
        }
    }
}

fn fill_lower_colmajor_f32(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        let base = j * lda;
        for i in 0..n {
            a[base + i] = f32_from_u32_f32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        let base = j * lda;
        for i in 0..j {
            a[base + i] = 0.0;
        }
    }
}

fn fill_upper_colmajor_f64(a: &mut [f64], n: usize, lda: usize) {
    for j in 0..n {
        let base = j * lda;
        for i in 0..n {
            a[base + i] = f64_from_u32_f64(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        let base = j * lda;
        for i in (j + 1)..n {
            a[base + i] = 0.0;
        }
    }
}

fn fill_lower_colmajor_f64(a: &mut [f64], n: usize, lda: usize) {
    for j in 0..n {
        let base = j * lda;
        for i in 0..n {
            a[base + i] = f64_from_u32_f64(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        let base = j * lda;
        for i in 0..j {
            a[base + i] = 0.0;
        }
    }
}

unsafe fn cblas_strmv_upper_notrans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n, a, lda, x, incx,
    );
}}
unsafe fn cblas_strmv_lower_notrans_nonunit_f32(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
) { unsafe {
    cblas_strmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n, a, lda, x, incx,
    );
}}

// cblas wrappers (f64)
unsafe fn cblas_dtrmv_upper_notrans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n, a, lda, x, incx,
    );
}}
unsafe fn cblas_dtrmv_lower_notrans_nonunit_f64(
    n: i32,
    a: *const f64, lda: i32,
    x: *mut f64, incx: i32,
) { unsafe {
    cblas_dtrmv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n, a, lda, x, incx,
    );
}}

fn bench_trmv(c: &mut Criterion) {
    let n: usize   = 2048;
    let lda: usize = n + 8;

    c.bench_function("rusty_strumv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_upper_colmajor_f32(&mut a, n, lda);
                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                strmv(
                    black_box(UpLo::UpperTriangular),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnitDiag),
                    black_box(n),
                    black_box(&a),
                    black_box(1isize),
                    black_box(lda as isize),
                    black_box(&mut x),
                    black_box(1isize),
                );
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_strumv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_upper_colmajor_f32(&mut a, n, lda);
                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_strmv_upper_notrans_nonunit_f32(
                        black_box(n as i32),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_strlmv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_lower_colmajor_f32(&mut a, n, lda);
                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                strmv(
                    black_box(UpLo::LowerTriangular),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnitDiag),
                    black_box(n),
                    black_box(&a),
                    black_box(1isize),
                    black_box(lda as isize),
                    black_box(&mut x),
                    black_box(1isize),
                );
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_strlmv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_lower_colmajor_f32(&mut a, n, lda);
                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_strmv_lower_notrans_nonunit_f32(
                        black_box(n as i32),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_dtrumv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                fill_upper_colmajor_f64(&mut a, n, lda);
                let mut x = vec![0.0f64; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                dtrmv(
                    black_box(UpLo::UpperTriangular),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnitDiag),
                    black_box(n),
                    black_box(&a),
                    black_box(1isize),
                    black_box(lda as isize),
                    black_box(&mut x),
                    black_box(1isize),
                );
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_dtrumv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                fill_upper_colmajor_f64(&mut a, n, lda);
                let mut x = vec![0.0f64; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_dtrmv_upper_notrans_nonunit_f64(
                        black_box(n as i32),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_dtrlmv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                fill_lower_colmajor_f64(&mut a, n, lda);
                let mut x = vec![0.0f64; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                dtrmv(
                    black_box(UpLo::LowerTriangular),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnitDiag),
                    black_box(n),
                    black_box(&a),
                    black_box(1isize),
                    black_box(lda as isize),
                    black_box(&mut x),
                    black_box(1isize),
                );
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("cblas_dtrlmv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f64; lda * n];
                fill_lower_colmajor_f64(&mut a, n, lda);
                let mut x = vec![0.0f64; n];
                for v in &mut x { *v = 1.0; }
                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_dtrmv_lower_notrans_nonunit_f64(
                        black_box(n as i32),
                        black_box(a.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1i32),
                    );
                }
                black_box(&x);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_trmv);
criterion_main!(benches);

