use rusty_blas::level1::{ 
    scopy::scopy, 
    dcopy::dcopy, 
    ccopy::ccopy, 
    zcopy::zcopy, 
};
use cblas_sys::{
    cblas_scopy, 
    cblas_dcopy, 
    cblas_ccopy, 
    cblas_zcopy
};


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

// unit stride tests 

#[test]
fn scopy_matches_cblas() {
    let sizes = [0usize, 1, 2, 7, 64, 123];

    for &n in &sizes {
        let mut x = vec![0.0f32; n];
        for i in 0..n {
            x[i] = i as f32 * 0.5 - 2.0;
        }

        let mut y_mine = vec![0.0f32; n];
        let mut y_blas = vec![0.0f32; n];

        scopy(n, &x, 1, &mut y_mine, 1);
        unsafe {
            cblas_scopy(
                n as i32, 
                x.as_ptr(), 
                1, 
                y_blas.as_mut_ptr(),
                1
            );
        }

        assert_bits_eq_f32(&y_mine[..n], &y_blas[..n]);
    }
}


#[test]
fn ccopy_matches_cblas() {
    let sizes = [0usize, 1, 2, 5, 32, 77];

    for &n in &sizes {
        let mut x = vec![0.0f32; 2 * n];
        for k in 0..n {
            x[2 * k]     = k as f32;
            x[2 * k + 1] = k as f32 + 0.5;
        }

        let mut y_mine = vec![0.0f32; 2 * n];
        let mut y_blas = vec![0.0f32; 2 * n];

        ccopy(n, &x, 1, &mut y_mine, 1);
        unsafe {
            cblas_ccopy(
                n as i32,
                x.as_ptr() as *const _,
                1,
                y_blas.as_mut_ptr() as *mut _,
                1,
            );
        }

        assert_bits_eq_f32(&y_mine[..2 * n], &y_blas[..2 * n]);
    }
}

#[test]
fn zcopy_matches_cblas() {
    let sizes = [0usize, 1, 2, 5, 32, 77];

    for &n in &sizes {
        let mut x = vec![0.0f64; 2 * n];
        for k in 0..n {
            x[2 * k]     = k as f64;
            x[2 * k + 1] = k as f64 + 0.5;
        }

        let mut y_mine = vec![0.0f64; 2 * n];
        let mut y_blas = vec![0.0f64; 2 * n];

        zcopy(n, &x, 1, &mut y_mine, 1);
        unsafe {
            cblas_zcopy(
                n as i32,
                x.as_ptr() as *const _,
                1,
                y_blas.as_mut_ptr() as *mut _,
                1,
            );
        }

        assert_bits_eq_f64(&y_mine[..2 * n], &y_blas[..2 * n]);
    }
}

// non unit stride tests below 

#[test]
fn scopy_matches_cblas_stride() {
    let n = 50usize;
    let incx = 2isize;
    let incy = 3isize;

    let len_x = 1 + (n.saturating_sub(1)) * (incx as usize);
    let len_y = 1 + (n.saturating_sub(1)) * (incy as usize);

    let mut x = vec![0.0f32; len_x];
    for i in 0..n { x[i * (incx as usize)] = i as f32 * 0.25 - 1.0; }

    let mut y_mine = vec![-9.0f32; len_y];
    let mut y_blas = vec![-9.0f32; len_y];

    scopy(n, &x, incx, &mut y_mine, incy);
    unsafe {
        cblas_scopy(
            n as i32, 
            x.as_ptr(), 
            incx as i32, 
            y_blas.as_mut_ptr(), 
            incy as i32
        );
    }

    assert_bits_eq_f32(&y_mine[..len_y], &y_blas[..len_y]);
}

#[test]
fn dcopy_matches_cblas_stride() {
    let n = 61usize;
    let incx = 3isize;
    let incy = 2isize;

    let len_x = 1 + (n.saturating_sub(1)) * (incx as usize);
    let len_y = 1 + (n.saturating_sub(1)) * (incy as usize);

    let mut x = vec![0.0f64; len_x];
    for i in 0..n { x[i * (incx as usize)] = (i as f64 * 0.31).sin(); }

    let mut y_mine = vec![7.7f64; len_y];
    let mut y_blas = vec![7.7f64; len_y];

    dcopy(n, &x, incx, &mut y_mine, incy);
    unsafe {
        cblas_dcopy(
            n as i32, 
            x.as_ptr(), 
            incx as i32, 
            y_blas.as_mut_ptr(), 
            incy as i32
        );
    }

    assert_bits_eq_f64(&y_mine[..len_y], &y_blas[..len_y]);
}

#[test]
fn ccopy_matches_cblas_stride() {
    let n = 40usize;
    let inc = 2isize; 

    let len = 1 + (n.saturating_sub(1)) * (inc as usize);
    let mut x = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        x[2*k]     = i as f32 * 0.5;
        x[2*k + 1] = i as f32 * 0.5 + 0.25;
    }

    let mut y_mine = vec![-3.0f32; 2 * len];
    let mut y_blas = vec![-3.0f32; 2 * len];

    ccopy(n, &x, inc, &mut y_mine, inc);
    unsafe {
        cblas_ccopy(
            n as i32, 
            x.as_ptr() as *const _, 
            inc as i32,
            y_blas.as_mut_ptr() as *mut _,
            inc as i32
        );
    }

    assert_bits_eq_f32(&y_mine[..2 * len], &y_blas[..2 * len]);
}

#[test]
fn zcopy_matches_cblas_stride() {
    let n = 33usize;
    let inc = 3isize;

    let len = 1 + (n.saturating_sub(1)) * (inc as usize);
    let mut x = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * (inc as usize);
        x[2*k]     = i as f64 * 0.7;
        x[2*k + 1] = i as f64 * 0.7 + 0.5;
    }

    let mut y_mine = vec![4.4f64; 2 * len];
    let mut y_blas = vec![4.4f64; 2 * len];

    zcopy(n, &x, inc, &mut y_mine, inc);
    unsafe {
        cblas_zcopy(
            n as i32, 
            x.as_ptr() as *const _, 
            inc as i32,
            y_blas.as_mut_ptr() as *mut _, 
            inc as i32);
    }

    assert_bits_eq_f64(&y_mine[..2 * len], &y_blas[..2 * len]);
}


