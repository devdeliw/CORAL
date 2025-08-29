use rusty_blas::level1::asum::{sasum, dasum}; 
use cblas_sys::{cblas_sasum, cblas_dasum}; 
use criterion::{criterion_group, black_box, criterion_main, Criterion}; 

// only single stride 
fn bench_asum(c: &mut Criterion) { 
    let n: usize           = 10_000; 
    let data_f32: Vec<f32> = vec![1.0; n]; 
    let data_f64: Vec<f64> = vec![1.0; n]; 

    // single precision 
    c.bench_function("my_blas_f32", |b| { 
        b.iter(|| { 
            let s = sasum(
                black_box(n as usize), 
                black_box(&data_f32), 
                black_box(1 as isize)
            ); 
            black_box(s); 
        })
    });
    c.bench_function("cblas_sasum", |b| {
        b.iter(|| {
            let s = unsafe { 
                cblas_sasum(
                    black_box(n as i32), 
                    black_box(data_f32.as_ptr()), 
                    black_box(1 as i32)
                ) 
            };
            black_box(s);
        }) 
    });

    // double precision
    c.bench_function("my_blas_f64", |b| { 
        b.iter(|| { 
            let s = dasum(
                black_box(n as usize), 
                black_box(&data_f64), 
                black_box(1 as isize) 
            ); 
            black_box(s); 
        })
    });
    c.bench_function("cblas_dasum", |b| {
        b.iter(|| {
            let s = unsafe { 
                cblas_dasum(
                    black_box(n as i32), 
                    black_box(data_f64.as_ptr()), 
                    black_box(1 as i32)
                ) 
            };
            black_box(s);
        }) 
    });
}

criterion_group!(benches, bench_asum);
criterion_main!(benches);


