use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use cblas_sys::{cblas_srot, cblas_drot, cblas_srotg, cblas_drotg, cblas_srotmg, cblas_drotmg};
use rusty_blas::level1::{
    srot::srot,
    drot::drot,
    srotg::srotg,
    drotg::drotg,
    srotmg::srotmg,
    drotmg::drotmg
};

fn bench_rot(c: &mut Criterion) {
    let n = 1_000_000usize;

    let x_f32: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_f32: Vec<f32> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f32) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();
    let c_f32: f32 = 0.8;
    let s_f32: f32 = 0.6;

    let x_f64: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 0.001 + if i & 1 == 0 { 0.25 } else { -0.25 })
        .collect();
    let y_f64: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i % 7) as f64) * 0.03125 - if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();
    let c_f64: f64 = 0.8;
    let s_f64: f64 = 0.6;

    c.bench_function("rusty_srot_inc1", |b| {
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                srot(n, &mut x, 1, &mut y, 1, c_f32, s_f32);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_srot_inc1", |b| {
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_srot(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c_f32, s_f32) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_drot_inc1", |b| {
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                drot(n, &mut x, 1, &mut y, 1, c_f64, s_f64);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_drot_inc1", |b| {
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_drot(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c_f64, s_f64) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_srotg", |bch| {
        bch.iter(|| {
            let mut a = black_box(3.25f32);
            let mut bval = black_box(-4.75f32);
            let mut cval = 0.0f32;
            let mut sval = 0.0f32;
            srotg(&mut a, &mut bval, &mut cval, &mut sval);
            black_box((a, bval, cval, sval))
        })
    });
    c.bench_function("cblas_srotg", |bch| {
        bch.iter(|| unsafe {
            let mut a = black_box(3.25f32);
            let mut bval = black_box(-4.75f32);
            let mut cval = 0.0f32;
            let mut sval = 0.0f32;
            cblas_srotg(&mut a, &mut bval, &mut cval, &mut sval);
            black_box((a, bval, cval, sval))
        })
    });

    c.bench_function("rusty_drotg", |bch| {
        bch.iter(|| {
            let mut a = black_box(6.0f64);
            let mut bval = black_box(-2.0f64);
            let mut cval = 0.0f64;
            let mut sval = 0.0f64;
            drotg(&mut a, &mut bval, &mut cval, &mut sval);
            black_box((a, bval, cval, sval))
        })
    });
    c.bench_function("cblas_drotg", |bch| {
        bch.iter(|| unsafe {
            let mut a = black_box(6.0f64);
            let mut bval = black_box(-2.0f64);
            let mut cval = 0.0f64;
            let mut sval = 0.0f64;
            cblas_drotg(&mut a, &mut bval, &mut cval, &mut sval);
            black_box((a, bval, cval, sval))
        })
    });

    c.bench_function("rusty_srotm_inc1_flag_m1", |b| {
        let param_m1_f32: [f32; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2]; 
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                rusty_blas::level1::srotm::srotm(n, &mut x, 1, &mut y, 1, &param_m1_f32);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_srotm_inc1_flag_m1", |b| {
        let param_m1_f32: [f32; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2];
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_sys::cblas_srotm(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, param_m1_f32.as_ptr()) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_srotm_inc1_flag_p1", |b| {
        let param_p1_f32: [f32; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                rusty_blas::level1::srotm::srotm(n, &mut x, 1, &mut y, 1, &param_p1_f32);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_srotm_inc1_flag_p1", |b| {
        let param_p1_f32: [f32; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];
        b.iter_batched(
            || (x_f32.clone(), y_f32.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_sys::cblas_srotm(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, param_p1_f32.as_ptr()) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_drotm_inc1_flag_m1", |b| {
        let param_m1_f64: [f64; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2]; // full 2x2
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                rusty_blas::level1::drotm::drotm(n, &mut x, 1, &mut y, 1, &param_m1_f64);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_drotm_inc1_flag_m1", |b| {
        let param_m1_f64: [f64; 5] = [-1.0, 0.9, -0.4, 0.3, 1.2];
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_sys::cblas_drotm(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, param_m1_f64.as_ptr()) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_drotm_inc1_flag_p1", |b| {
        let param_p1_f64: [f64; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                rusty_blas::level1::drotm::drotm(n, &mut x, 1, &mut y, 1, &param_p1_f64);
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("cblas_drotm_inc1_flag_p1", |b| {
        let param_p1_f64: [f64; 5] = [1.0, 0.8, 0.0, 0.0, 1.1];
        b.iter_batched(
            || (x_f64.clone(), y_f64.clone()),
            |(mut x, mut y)| {
                unsafe { cblas_sys::cblas_drotm(n as i32, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, param_p1_f64.as_ptr()) };
                black_box(&x);
                black_box(&y);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rusty_srotmg_flag_m2", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(1.5f32);
            let mut sd2 = black_box(2.3f32);
            let mut sx1 = black_box(0.7f32);
            let sy1 = black_box(0.0f32); // sp2 = sd2*sy1 = 0 => flag -2
            let mut param = [0.0f32; 5];
            srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_srotmg_flag_m2", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(1.5f32);
            let mut sd2 = black_box(2.3f32);
            let mut sx1 = black_box(0.7f32);
            let sy1 = black_box(0.0f32);
            let mut param = [0.0f32; 5];
            cblas_srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_srotmg_flag_m1", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(-1.0f32);
            let mut sd2 = black_box(2.0f32);
            let mut sx1 = black_box(3.0f32);
            let sy1 = black_box(1.25f32);
            let mut param = [0.0f32; 5];
            srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_srotmg_flag_m1", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(-1.0f32);
            let mut sd2 = black_box(2.0f32);
            let mut sx1 = black_box(3.0f32);
            let sy1 = black_box(1.25f32);
            let mut param = [0.0f32; 5];
            cblas_srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_srotmg_flag_0", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(2.0f32);
            let mut sd2 = black_box(0.5f32);
            let mut sx1 = black_box(3.0f32);
            let sy1 = black_box(0.25f32);
            let mut param = [0.0f32; 5];
            srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_srotmg_flag_0", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(2.0f32);
            let mut sd2 = black_box(0.5f32);
            let mut sx1 = black_box(3.0f32);
            let sy1 = black_box(0.25f32);
            let mut param = [0.0f32; 5];
            cblas_srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_srotmg_flag_p1", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(0.5f32);
            let mut sd2 = black_box(2.0f32);
            let mut sx1 = black_box(0.5f32);
            let sy1 = black_box(3.0f32);
            let mut param = [0.0f32; 5];
            srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_srotmg_flag_p1", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(0.5f32);
            let mut sd2 = black_box(2.0f32);
            let mut sx1 = black_box(0.5f32);
            let sy1 = black_box(3.0f32);
            let mut param = [0.0f32; 5];
            cblas_srotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    }); 

    c.bench_function("rusty_drotmg_flag_m2", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(1.5f64);
            let mut sd2 = black_box(2.3f64);
            let mut sx1 = black_box(0.7f64);
            let sy1 = black_box(0.0f64); // sp2 = sd2*sy1 = 0 => flag -2
            let mut param = [0.0f64; 5];
            drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_drotmg_flag_m2", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(1.5f64);
            let mut sd2 = black_box(2.3f64);
            let mut sx1 = black_box(0.7f64);
            let sy1 = black_box(0.0f64);
            let mut param = [0.0f64; 5];
            cblas_drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_drotmg_flag_m1", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(-1.0f64);
            let mut sd2 = black_box(2.0f64);
            let mut sx1 = black_box(3.0f64);
            let sy1 = black_box(1.25f64);
            let mut param = [0.0f64; 5];
            drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_drotmg_flag_m1", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(-1.0f64);
            let mut sd2 = black_box(2.0f64);
            let mut sx1 = black_box(3.0f64);
            let sy1 = black_box(1.25f64);
            let mut param = [0.0f64; 5];
            cblas_drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_drotmg_flag_0", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(2.0f64);
            let mut sd2 = black_box(0.5f64);
            let mut sx1 = black_box(3.0f64);
            let sy1 = black_box(0.25f64);
            let mut param = [0.0f64; 5];
            drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_drotmg_flag_0", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(2.0f64);
            let mut sd2 = black_box(0.5f64);
            let mut sx1 = black_box(3.0f64);
            let sy1 = black_box(0.25f64);
            let mut param = [0.0f64; 5];
            cblas_drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });

    c.bench_function("rusty_drotmg_flag_p1", |bch| {
        bch.iter(|| {
            let mut sd1 = black_box(0.5f64);
            let mut sd2 = black_box(2.0f64);
            let mut sx1 = black_box(0.5f64);
            let sy1 = black_box(3.0f64);
            let mut param = [0.0f64; 5];
            drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, &mut param);
            black_box((sd1, sd2, sx1, param))
        })
    });
    c.bench_function("cblas_drotmg_flag_p1", |bch| {
        bch.iter(|| unsafe {
            let mut sd1 = black_box(0.5f64);
            let mut sd2 = black_box(2.0f64);
            let mut sx1 = black_box(0.5f64);
            let sy1 = black_box(3.0f64);
            let mut param = [0.0f64; 5];
            cblas_drotmg(&mut sd1, &mut sd2, &mut sx1, sy1, param.as_mut_ptr());
            black_box((sd1, sd2, sx1, param))
        })
    });
}

criterion_group!(benches, bench_rot);
criterion_main!(benches);
