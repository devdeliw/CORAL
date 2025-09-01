use rusty_blas::level1::{ 
    sswap::sswap, 
    dswap::dswap, 
    cswap::cswap, 
    zswap::zswap
}; 
use cblas_sys::{cblas_sswap, cblas_dswap, cblas_cswap, cblas_zswap};


#[inline]
fn assert_bits_eq_f32(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(x.to_bits() == y.to_bits(), "f32 mismatch at {i}: {x:?} vs {y:?}");
    }
}

#[inline]
fn assert_bits_eq_f64(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(x.to_bits() == y.to_bits(), "f64 mismatch at {i}: {x:?} vs {y:?}");
    }
}


#[test]
fn sswap_matches_cblas() {
    let sizes = [0usize, 1, 2, 7, 64, 123];

    for &n in &sizes {
        let mut x_mine = vec![0.0f32; n];
        let mut y_mine = vec![0.0f32; n];
        let mut x_blas = vec![0.0f32; n];
        let mut y_blas = vec![0.0f32; n];

        for i in 0..n {
            x_mine[i] = i as f32 * 0.5 - 2.0;
            y_mine[i] = -(i as f32) * 0.25 + 1.0;
            x_blas[i] = x_mine[i];
            y_blas[i] = y_mine[i];
        }

        sswap(n, &mut x_mine, 1, &mut y_mine, 1);
        unsafe {
            cblas_sswap(
                n as i32, 
                x_blas.as_mut_ptr(), 
                1, 
                y_blas.as_mut_ptr(), 
                1
            );
        }

        assert_bits_eq_f32(&x_mine[..n], &x_blas[..n]);
        assert_bits_eq_f32(&y_mine[..n], &y_blas[..n]);
    }
}

#[test]
fn sswap_matches_cblas_stride() {
    let n = 50usize;
    let incx = 2isize;
    let incy = 3isize;

    let len_x = 1 + (n - 1) * (incx as usize);
    let len_y = 1 + (n - 1) * (incy as usize);

    let mut x_mine = vec![0.0f32; len_x];
    let mut y_mine = vec![0.0f32; len_y];
    let mut x_blas = vec![0.0f32; len_x];
    let mut y_blas = vec![0.0f32; len_y];

    for i in 0..n {
        x_mine[i * (incx as usize)] = i as f32 * 0.3 - 1.0;
        y_mine[i * (incy as usize)] = -(i as f32) * 0.2 + 2.0;
    }
    x_blas.copy_from_slice(&x_mine);
    y_blas.copy_from_slice(&y_mine);

    sswap(n, &mut x_mine, incx, &mut y_mine, incy);
    unsafe { 
        cblas_sswap(
            n as i32,
            x_blas.as_mut_ptr(), 
            incx as i32, 
            y_blas.as_mut_ptr(), 
            incy as i32
        ); 
    }

    assert_bits_eq_f32(&x_mine, &x_blas);
    assert_bits_eq_f32(&y_mine, &y_blas);
}

#[test]
fn dswap_matches_cblas() {
    let n = 129usize;

    let mut x_mine = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect::<Vec<_>>();
    let mut y_mine = (0..n).map(|i| -(i as f64) * 0.25 + 2.0).collect::<Vec<_>>();
    let mut x_blas = x_mine.clone();
    let mut y_blas = y_mine.clone();

    dswap(n, &mut x_mine, 1, &mut y_mine, 1);
    unsafe { cblas_dswap(n as i32, x_blas.as_mut_ptr(), 1, y_blas.as_mut_ptr(), 1); }

    assert_bits_eq_f64(&x_mine, &x_blas);
    assert_bits_eq_f64(&y_mine, &y_blas);
}

#[test]
fn dswap_matches_cblas_stride() {
    let n = 64usize;
    let incx = 3isize;
    let incy = 2isize;

    let len_x = 1 + (n - 1) * (incx as usize);
    let len_y = 1 + (n - 1) * (incy as usize);

    let mut x_mine = vec![0.0f64; len_x];
    let mut y_mine = vec![0.0f64; len_y];
    let mut x_blas = vec![0.0f64; len_x];
    let mut y_blas = vec![0.0f64; len_y];

    for i in 0..n {
        x_mine[i * (incx as usize)] = (i as f64 * 0.31).sin();
        y_mine[i * (incy as usize)] = (i as f64 * 0.17).cos();
    }
    x_blas.copy_from_slice(&x_mine);
    y_blas.copy_from_slice(&y_mine);

    dswap(n, &mut x_mine, incx, &mut y_mine, incy);
    unsafe {
        cblas_dswap(
            n as i32,
            x_blas.as_mut_ptr(),
            incx as i32, 
            y_blas.as_mut_ptr(), 
            incy as i32
        ); 
    }

    assert_bits_eq_f64(&x_mine, &x_blas);
    assert_bits_eq_f64(&y_mine, &y_blas);
}

#[test]
fn cswap_matches_cblas() {
    let n = 77usize;

    let mut x_pairs: Vec<[f32; 2]> = (0..n)
        .map(|i| [i as f32 * 0.5, i as f32 * 0.5 + 0.1])
        .collect();
    let mut y_pairs: Vec<[f32; 2]> = (0..n)
        .map(|i| [-(i as f32) * 0.2, -(i as f32) * 0.2 + 0.3])
        .collect();

    let mut x_pairs_blas = x_pairs.clone();
    let mut y_pairs_blas = y_pairs.clone();

    let x_flat: &mut [f32] = unsafe { 
        core::slice::from_raw_parts_mut(x_pairs.as_mut_ptr() as *mut f32, 2*n)
    };
    let y_flat: &mut [f32] = unsafe {
        core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f32, 2*n) 
    };

    cswap(n, x_flat, 1, y_flat, 1);
    unsafe { 
        cblas_cswap(
            n as i32, 
            x_pairs_blas.as_mut_ptr(), 
            1,
            y_pairs_blas.as_mut_ptr(), 
            1
        ); 
    }

    let x_blas_flat: &[f32] = unsafe { 
        core::slice::from_raw_parts(x_pairs_blas.as_ptr() as *const f32, 2*n)
    };
    let y_blas_flat: &[f32] = unsafe { 
        core::slice::from_raw_parts(y_pairs_blas.as_ptr() as *const f32, 2*n) 
    };

    assert_bits_eq_f32(x_flat, x_blas_flat);
    assert_bits_eq_f32(y_flat, y_blas_flat);
}

#[test]
fn cswap_matches_cblas_stride() {
    let n = 60usize;
    let inc = 2isize;
    let step = inc as usize;
    let len = 1 + (n - 1) * step;

    let mut x_pairs: Vec<[f32; 2]> = vec![[0.0; 2]; len];
    let mut y_pairs: Vec<[f32; 2]> = vec![[0.0; 2]; len];
    for i in 0..n {
        let k = i * step;
        x_pairs[k] = [i as f32 * 0.7, i as f32 * 0.7 + 0.05];
        y_pairs[k] = [-(i as f32) * 0.4, -(i as f32) * 0.4 + 0.2];
    }
    let mut xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &mut [f32] = unsafe {
        core::slice::from_raw_parts_mut(x_pairs.as_mut_ptr() as *mut f32, 2*len) 
    };
    let yf: &mut [f32] = unsafe { 
        core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f32, 2*len) 
    };

    cswap(n, xf, inc, yf, inc);
    unsafe { 
        cblas_cswap(
            n as i32,
            xb.as_mut_ptr(),
            inc as i32,
            yb.as_mut_ptr(), 
            inc as i32
        ); 
    }

    let xb_flat: &[f32] = unsafe { 
        core::slice::from_raw_parts(xb.as_ptr() as *const f32, 2*len) 
    };
    let yb_flat: &[f32] = unsafe {
        core::slice::from_raw_parts(yb.as_ptr() as *const f32, 2*len) 
    };

    assert_bits_eq_f32(xf, xb_flat);
    assert_bits_eq_f32(yf, yb_flat);
}

#[test]
fn zswap_matches_cblas() {
    let n = 66usize;

    let mut x_pairs: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64 * 0.8, i as f64 * 0.8 + 0.2])
        .collect();
    let mut y_pairs: Vec<[f64; 2]> = (0..n)
        .map(|i| [-(i as f64) * 0.5, -(i as f64) * 0.5 + 0.4])
        .collect();
    let mut xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(x_pairs.as_mut_ptr() as *mut f64, 2*n) 
    };
    let yf: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f64, 2*n) 
    };

    zswap(n, xf, 1, yf, 1);
    unsafe { 
        cblas_zswap(
            n as i32,
            xb.as_mut_ptr(), 
            1, 
            yb.as_mut_ptr(), 
            1
        ); 
    }

    let xb_flat: &[f64] = unsafe { 
        core::slice::from_raw_parts(xb.as_ptr() as *const f64, 2*n) 
    };
    let yb_flat: &[f64] = unsafe { 
        core::slice::from_raw_parts(yb.as_ptr() as *const f64, 2*n) 
    };

    assert_bits_eq_f64(xf, xb_flat);
    assert_bits_eq_f64(yf, yb_flat);
}

#[test]
fn zswap_matches_cblas_stride() {
    let n = 48usize;
    let inc = 3isize;
    let step = inc as usize;
    let len = 1 + (n - 1) * step;

    let mut x_pairs: Vec<[f64; 2]> = vec![[0.0; 2]; len];
    let mut y_pairs: Vec<[f64; 2]> = vec![[0.0; 2]; len];
    for i in 0..n {
        let k = i * step;
        x_pairs[k] = [i as f64 * 0.21, i as f64 * 0.21 + 0.03];
        y_pairs[k] = [-(i as f64) * 0.18, -(i as f64) * 0.18 + 0.07];
    }
    let mut xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(x_pairs.as_mut_ptr() as *mut f64, 2*len) 
    };
    let yf: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f64, 2*len) 
    };

    zswap(n, xf, inc, yf, inc);
    unsafe { 
        cblas_zswap(
            n as i32, 
            xb.as_mut_ptr(),
            inc as i32, 
            yb.as_mut_ptr(), 
            inc as i32
        ); 
    }

    let xb_flat: &[f64] = unsafe { 
        core::slice::from_raw_parts(xb.as_ptr() as *const f64, 2*len) 
    };
    let yb_flat: &[f64] = unsafe { 
        core::slice::from_raw_parts(yb.as_ptr() as *const f64, 2*len)
    };

    assert_bits_eq_f64(xf, xb_flat);
    assert_bits_eq_f64(yf, yb_flat);
}

