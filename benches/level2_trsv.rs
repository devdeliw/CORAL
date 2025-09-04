use cblas_sys::{
    CBLAS_DIAG, 
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE, 
    CBLAS_UPLO,
    cblas_strsv,
};
use rusty_blas::level2::{ 
    strusv::strusv, 
    strlsv::strlsv, 
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

// pseudo-random float in [-0.5, 0.5] 
fn f32_from_u32(u: u32) -> f32 {
    let v = (u.wrapping_mul(2654435761)) >> 8;
    ((v & 0xFFFF) as f32 / 65536.0) - 0.5
}

// generate col major upper triangular matrix 
fn fill_upper_colmajor(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in (j + 1)..n {
            a[j * lda + i] = 0.0;
        }
    }
}

// generate col major lower triangular matrix 
fn fill_lower_colmajor(a: &mut [f32], n: usize, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            a[j * lda + i] = f32_from_u32(((i as u32) << 16) ^ (j as u32));
        }
    }
    for i in 0..n {
        let idx = i * lda + i;
        a[idx] = 1.0 + a[idx].abs();
    }
    for j in 0..n {
        for i in 0..j {
            a[j * lda + i] = 0.0;
        }
    }
}

// cblas wrappers
unsafe fn cblas_strsv_upper_notrans_nonunit(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
)  { unsafe { 
    cblas_strsv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasUpper,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda,
        x, incx,
    );
}}
unsafe fn cblas_strsv_lower_notrans_nonunit(
    n: i32,
    a: *const f32, lda: i32,
    x: *mut f32, incx: i32,
)  { unsafe { 
    cblas_strsv(
        CBLAS_LAYOUT::CblasColMajor,
        CBLAS_UPLO::CblasLower,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_DIAG::CblasNonUnit,
        n,
        a, lda,
        x, incx,
    );
}}


fn bench_strusv(c: &mut Criterion) {
    let n: usize = 2048;
    let lda: usize = n + 8;

    c.bench_function("rusty_strusv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_upper_colmajor(&mut a, n, lda);

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                (a, x)
            },
            |(a, mut x)| {
                strusv(
                    black_box(n),
                    black_box(false),         
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

    c.bench_function("cblas_strusv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_upper_colmajor(&mut a, n, lda);

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_strsv_upper_notrans_nonunit(
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

    c.bench_function("rusty_strlsv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_lower_colmajor(&mut a, n, lda);

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                (a, x)
            },
            |(a, mut x)| {
                strlsv(
                    black_box(n),
                    black_box(false),         
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

    c.bench_function("cblas_strlsv", |b| {
        b.iter_batched(
            || {
                let mut a = vec![0.0f32; lda * n];
                fill_lower_colmajor(&mut a, n, lda);

                let mut x = vec![0.0f32; n];
                for v in &mut x { *v = 1.0; }

                (a, x)
            },
            |(a, mut x)| {
                unsafe {
                    cblas_strsv_lower_notrans_nonunit(
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

criterion_group!(benches, bench_strusv);
criterion_main!(benches);
