use cblas_sys::{cblas_sger, CBLAS_LAYOUT};
use coral::level2::sger::sger;

// cblas wrapper
fn cblas_sger_colmajor(
    m: i32,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    a: *mut f32,
    lda: i32,
) {
    unsafe {
        cblas_sger(
            CBLAS_LAYOUT::CblasColMajor,
            m,
            n,
            alpha,
            x,
            incx,
            y,
            incy,
            a,
            lda,
        );
    }
}

// helpers 
fn make_col_major_matrix(m: usize, n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..m {
            // deterministic pattern
            a[i + j * lda] = 0.1 + (i as f32) * 0.5 + (j as f32) * 0.25;
        }
    }
    a
}

fn make_strided_vec(len_logical: usize, inc: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut v = vec![0.0f32; (len_logical - 1) * inc + 1];
    let mut idx = 0usize;
    for k in 0..len_logical {
        v[idx] = f(k);
        idx += inc;
    }
    v
}

fn assert_allclose(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "mismatch at {}: got {} vs {} (diff = {}, tol = {})",
            i, x, y, diff, tol
        );
    }
}

#[test]
fn unit_strides() {
    let m = 7usize;
    let n = 5usize;
    let lda = m;

    let alpha = 1.25f32;

    let a0 = make_col_major_matrix(m, n, lda);
    let x = (0..m).map(|i| 0.2 + i as f32 * 0.1).collect::<Vec<_>>();
    let y = (0..n).map(|j| -0.3 + j as f32 * 0.05).collect::<Vec<_>>();

    // coral
    let mut a_coral = a0.clone();
    sger(
        m, n, alpha,
        &x, 1,
        &y, 1,
        &mut a_coral, lda,
    );

    // cblas
    let mut a_ref = a0.clone();
    cblas_sger_colmajor(
        m as i32, n as i32, alpha,
        x.as_ptr(), 1,
        y.as_ptr(), 1,
        a_ref.as_mut_ptr(), lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, 1e-4);
}

#[test]
fn strided_padded_lda() {
    let m = 9usize;
    let n = 4usize;
    let lda = m + 3; // padded lda

    let alpha = -0.85f32;

    let a0 = make_col_major_matrix(m, n, lda);

    // non-unit strides
    let incx = 2usize;
    let incy = 3usize;

    let x = make_strided_vec(m, incx, |i| 0.05 + 0.03 * i as f32);
    let y = make_strided_vec(n, incy, |j| 0.4 - 0.02 * j as f32);

    // coral
    let mut a_coral = a0.clone();
    sger(
        m, n, alpha,
        &x, incx,
        &y, incy,
        &mut a_coral, lda,
    );

    // cblas
    let mut a_ref = a0.clone();
    cblas_sger_colmajor(
        m as i32, n as i32, alpha,
        x.as_ptr(), incx as i32,
        y.as_ptr(), incy as i32,
        a_ref.as_mut_ptr(), lda as i32,
    );

    assert_allclose(&a_coral, &a_ref, 1e-4);
}


