use rusty_blas::level1::{
    saxpy::saxpy,
    daxpy::daxpy,
    caxpy::caxpy,
    zaxpy::zaxpy,
};
use cblas_sys::{cblas_saxpy, cblas_daxpy, cblas_caxpy, cblas_zaxpy};

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

#[inline]
fn assert_approx_eq_f32(a: &[f32], b: &[f32], rel: f32, abs: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= abs + rel * x.abs().max(y.abs()),
            "f32 approx mismatch at {i}: {x:?} vs {y:?}, diff={diff}"
        );
    }
}

#[inline]
fn assert_approx_eq_f64(a: &[f64], b: &[f64], rel: f64, abs: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= abs + rel * x.abs().max(y.abs()),
            "f64 approx mismatch at {i}: {x:?} vs {y:?}, diff={diff}"
        );
    }
}

#[test]
fn saxpy_matches_cblas_unit_stride() {
    let sizes = [0usize, 1, 2, 7, 64, 123];
    let alpha: f32 = 1.2345;

    for &n in &sizes {
        let x_mine = (0..n).map(|i| i as f32 * 0.5 - 2.0).collect::<Vec<_>>();
        let mut y_mine = (0..n).map(|i| -(i as f32) * 0.25 + 1.0).collect::<Vec<_>>();
        let x_blas = x_mine.clone();
        let mut y_blas = y_mine.clone();

        saxpy(n, alpha, &x_mine, 1, &mut y_mine, 1);
        unsafe { cblas_saxpy(n as i32, alpha, x_blas.as_ptr(), 1, y_blas.as_mut_ptr(), 1); }

        assert_bits_eq_f32(&x_mine, &x_blas);
        assert_approx_eq_f32(&y_mine, &y_blas, 1e-6, 1e-7);
    }
}

#[test]
fn saxpy_matches_cblas_nonunit_stride() {
    let n = 50usize;
    let alpha: f32 = -0.75;
    let incx = 2isize;
    let incy = 3isize;

    let len_x = 1 + (n - 1) * (incx as usize);
    let len_y = 1 + (n - 1) * (incy as usize);

    let mut x_mine = vec![0.0f32; len_x];
    let mut y_mine = vec![0.0f32; len_y];
    for i in 0..n {
        x_mine[i * (incx as usize)] = i as f32 * 0.31 - 1.0;
        y_mine[i * (incy as usize)] = (i as f32 * 0.17).cos();
    }
    let x_blas = x_mine.clone();
    let mut y_blas = y_mine.clone();

    saxpy(n, alpha, &x_mine, incx, &mut y_mine, incy);
    unsafe { cblas_saxpy(n as i32, alpha, x_blas.as_ptr(), incx as i32, y_blas.as_mut_ptr(), incy as i32); }

    assert_bits_eq_f32(&x_mine, &x_blas);
    assert_approx_eq_f32(&y_mine, &y_blas, 1e-6, 1e-7);
}

#[test]
fn daxpy_matches_cblas_unit_stride() {
    let n = 129usize;
    let alpha: f64 = 0.987654321;

    let x_mine = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect::<Vec<_>>();
    let mut y_mine = (0..n).map(|i| -(i as f64) * 0.25 + 2.0).collect::<Vec<_>>();
    let x_blas = x_mine.clone();
    let mut y_blas = y_mine.clone();

    daxpy(n, alpha, &x_mine, 1, &mut y_mine, 1);
    unsafe { cblas_daxpy(n as i32, alpha, x_blas.as_ptr(), 1, y_blas.as_mut_ptr(), 1); }

    assert_bits_eq_f64(&x_mine, &x_blas);
    assert_approx_eq_f64(&y_mine, &y_blas, 1e-13, 1e-15);
}

#[test]
fn daxpy_matches_cblas_nonunit_stride() {
    let n = 64usize;
    let alpha: f64 = -1.25;
    let incx = 3isize;
    let incy = 2isize;

    let len_x = 1 + (n - 1) * (incx as usize);
    let len_y = 1 + (n - 1) * (incy as usize);

    let mut x_mine = vec![0.0f64; len_x];
    let mut y_mine = vec![0.0f64; len_y];
    for i in 0..n {
        x_mine[i * (incx as usize)] = (i as f64 * 0.31).sin();
        y_mine[i * (incy as usize)] = (i as f64 * 0.17).cos();
    }
    let x_blas = x_mine.clone();
    let mut y_blas = y_mine.clone();

    daxpy(n, alpha, &x_mine, incx, &mut y_mine, incy);
    unsafe { cblas_daxpy(n as i32, alpha, x_blas.as_ptr(), incx as i32, y_blas.as_mut_ptr(), incy as i32); }

    assert_bits_eq_f64(&x_mine, &x_blas);
    assert_approx_eq_f64(&y_mine, &y_blas, 1e-13, 1e-15);
}

#[test]
fn caxpy_matches_cblas_unit_stride() {
    let n = 77usize;
    let alpha = [0.9f32, -0.3f32]; 

    let x_pairs: Vec<[f32; 2]> = (0..n).map(|i| [i as f32 * 0.5, i as f32 * 0.5 + 0.1]).collect();
    let mut y_pairs: Vec<[f32; 2]> = (0..n).map(|i| [-(i as f32) * 0.2, -(i as f32) * 0.2 + 0.3]).collect();
    let x_pairs_blas = x_pairs.clone();
    let mut y_pairs_blas = y_pairs.clone();

    let xf: & [f32] = unsafe { core::slice::from_raw_parts(x_pairs.as_ptr() as *const f32, 2*n) };
    let yf: &mut [f32] = unsafe { core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f32, 2*n) };

    caxpy(n, alpha, xf, 1, yf, 1);
    unsafe { cblas_caxpy(n as i32, alpha.as_ptr() as *const _, x_pairs_blas.as_ptr(), 1, y_pairs_blas.as_mut_ptr(), 1); }

    let yb_flat: &[f32] = unsafe { core::slice::from_raw_parts(y_pairs_blas.as_ptr() as *const f32, 2*n) };

    let xf_after: &[f32] = unsafe { core::slice::from_raw_parts(x_pairs.as_ptr() as *const f32, 2*n) };
    assert_bits_eq_f32(xf_after, unsafe { core::slice::from_raw_parts(x_pairs_blas.as_ptr() as *const f32, 2*n) });
    assert_approx_eq_f32(yf, yb_flat, 1e-6, 1e-7);
}

#[test]
fn caxpy_matches_cblas_nonunit_stride() {
    let n = 60usize;
    let incx = 2isize;
    let incy = 3isize;
    let step_x = incx as usize;
    let step_y = incy as usize;
    let len_x = 1 + (n - 1) * step_x;
    let len_y = 1 + (n - 1) * step_y;
    let alpha = [-0.4f32, 0.7f32];

    let mut x_pairs: Vec<[f32; 2]> = vec![[0.0; 2]; len_x];
    let mut y_pairs: Vec<[f32; 2]> = vec![[0.0; 2]; len_y];
    for i in 0..n {
        let kx = i * step_x;
        let ky = i * step_y;
        x_pairs[kx] = [i as f32 * 0.7, i as f32 * 0.7 + 0.05];
        y_pairs[ky] = [-(i as f32) * 0.4, -(i as f32) * 0.4 + 0.2];
    }
    let xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &[f32]  = unsafe { core::slice::from_raw_parts(x_pairs.as_ptr() as *const f32, 2*len_x) };
    let yf: &mut [f32] = unsafe { core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f32, 2*len_y) };

    caxpy(n, alpha, xf, incx, yf, incy);
    unsafe { cblas_caxpy(n as i32, alpha.as_ptr() as *const _, xb.as_ptr(), incx as i32, yb.as_mut_ptr(), incy as i32); }

    let xb_flat: &[f32] = unsafe { core::slice::from_raw_parts(xb.as_ptr() as *const f32, 2*len_x) };
    let yb_flat: &[f32] = unsafe { core::slice::from_raw_parts(yb.as_ptr() as *const f32, 2*len_y) };

    assert_bits_eq_f32(xf, xb_flat);
    assert_approx_eq_f32(yf, yb_flat, 1e-6, 1e-7);
}

#[test]
fn zaxpy_matches_cblas_unit_stride() {
    let n = 66usize;
    let alpha = [0.25f64, -0.11f64];

    let x_pairs: Vec<[f64; 2]> = (0..n).map(|i| [i as f64 * 0.8, i as f64 * 0.8 + 0.2]).collect();
    let mut y_pairs: Vec<[f64; 2]> = (0..n).map(|i| [-(i as f64) * 0.5, -(i as f64) * 0.5 + 0.4]).collect();
    let xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &[f64] = unsafe { core::slice::from_raw_parts(x_pairs.as_ptr() as *const f64, 2*n) };
    let yf: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f64, 2*n) };

    zaxpy(n, alpha, xf, 1, yf, 1);
    unsafe { cblas_zaxpy(n as i32, alpha.as_ptr() as *const _, xb.as_ptr(), 1, yb.as_mut_ptr(), 1); }

    let xb_flat: &[f64] = unsafe { core::slice::from_raw_parts(xb.as_ptr() as *const f64, 2*n) };
    let yb_flat: &[f64] = unsafe { core::slice::from_raw_parts(yb.as_ptr() as *const f64, 2*n) };

    assert_bits_eq_f64(xf, xb_flat);
    assert_approx_eq_f64(yf, yb_flat, 1e-13, 1e-15);
}

#[test]
fn zaxpy_matches_cblas_nonunit_stride() {
    let n = 48usize;
    let incx = 3isize;
    let incy = 2isize;
    let step_x = incx as usize;
    let step_y = incy as usize;
    let len_x = 1 + (n - 1) * step_x;
    let len_y = 1 + (n - 1) * step_y;
    let alpha = [-0.33f64, 0.125f64];

    let mut x_pairs: Vec<[f64; 2]> = vec![[0.0; 2]; len_x];
    let mut y_pairs: Vec<[f64; 2]> = vec![[0.0; 2]; len_y];
    for i in 0..n {
        let kx = i * step_x;
        let ky = i * step_y;
        x_pairs[kx] = [i as f64 * 0.21, i as f64 * 0.21 + 0.03];
        y_pairs[ky] = [-(i as f64) * 0.18, -(i as f64) * 0.18 + 0.07];
    }
    let xb = x_pairs.clone();
    let mut yb = y_pairs.clone();

    let xf: &[f64] = unsafe { core::slice::from_raw_parts(x_pairs.as_ptr() as *const f64, 2*len_x) };
    let yf: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(y_pairs.as_mut_ptr() as *mut f64, 2*len_y) };

    zaxpy(n, alpha, xf, incx, yf, incy);
    unsafe { cblas_zaxpy(n as i32, alpha.as_ptr() as *const _, xb.as_ptr(), incx as i32, yb.as_mut_ptr(), incy as i32); }

    let xb_flat: &[f64] = unsafe { core::slice::from_raw_parts(xb.as_ptr() as *const f64, 2*len_x) };
    let yb_flat: &[f64] = unsafe { core::slice::from_raw_parts(yb.as_ptr() as *const f64, 2*len_y) };

    assert_bits_eq_f64(xf, xb_flat);
    assert_approx_eq_f64(yf, yb_flat, 1e-13, 1e-15);
}

