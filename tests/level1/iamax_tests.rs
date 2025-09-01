use rusty_blas::level1::{
    isamax::isamax, 
    idamax::idamax, 
    icamax::icamax, 
    izamax::izamax,
};
use cblas_sys::{
    cblas_isamax, 
    cblas_idamax, 
    cblas_icamax, 
    cblas_izamax,
};


#[test]
fn isamax_matches_cblas_stride() {
    let n     = 1000usize;
    let mut x = vec![0.0f32; n];

    x[123] = -5.0;
    x[987] =  4.9999;

    let rusty1 = isamax(n, &x, 1);
    let cblas1 = unsafe { 
        cblas_isamax(
            n as i32, 
            x.as_ptr(), 
            1
        ) 
    } as usize;

    // non unit stride 
    let stride = 2isize;
    let mut x  = vec![0.0f32; n * 2];
    for i in 0..n {
        x[2*i] = (i as f32 * 0.01).sin().abs();
    }
    x[2*700] = 10.0;

    let rusty2 = isamax(n, &x, stride);
    let cblas2 = unsafe { 
        cblas_isamax(
            n as i32, 
            x.as_ptr(), 
            stride as i32
        )
    } as usize;

    assert_eq!(rusty1, cblas1 + 1);
    assert_eq!(rusty2, cblas2 + 1);
}

#[test]
fn idamax_matches_cblas_tie() {
    let n     = 8usize;
    let mut x = vec![1.0f64; n];

    // idx 2 should be returned
    x[2] = -3.0;
    x[5] =  3.0;

    let rusty1 = idamax(n, &x, 1);
    let cblas1 = unsafe { 
        cblas_idamax(
            n as i32, 
            x.as_ptr(), 
            1
        ) 
    } as usize;

    assert_eq!(rusty1, cblas1 + 1);
    assert_eq!(rusty1, 3);
}


#[test]
fn idamax_matches_cblas_stride() {
    let n   = 301usize;
    let inc = 4isize;
    let len = 1 + (n - 1) * (inc as usize);

    let mut x = vec![0.0f64; len];
    for i in 0..n {
        x[i * (inc as usize)] = (i as f64 * 0.1).sin().abs();
    }
    x[77 * (inc as usize)] = 9e9;

    let rusty1 = idamax(n, &x, inc);
    let cblas1 = unsafe { 
        cblas_idamax(
            n as i32, 
            x.as_ptr(), 
            inc as i32
        ) 
    } as usize;

    assert_eq!(rusty1, cblas1 + 1);
}

#[test]
fn icamax_matches_cblas_stride() {
    let n    = 240usize;
    let inc  = 3isize;
    let step = inc as usize;
    let len  = 1 + (n - 1) * step;

    let mut z = vec![0.0f32; 2 * len];
    for i in 0..n {
        let k = i * step;
        z[2*k]   = (i as f32 * 0.13).sin();
        z[2*k+1] = (i as f32 * 0.17).cos() * 0.5;
    }


    let i_big = 101 * step;
    z[2*i_big]     = 40.0; 
    z[2*i_big + 1] = 40.0; 

    let rusty1 = icamax(n, &z, inc);
    let cblas1 = unsafe { 
        cblas_icamax(
            n as i32, 
            z.as_ptr().cast::<[f32; 2]>(), 
            inc as i32
        )
    } as usize;

    assert_eq!(rusty1, cblas1 + 1);
}

#[test]
fn izamax_matches_cblas_stride() {
    let n    = 256usize;
    let inc  = 3isize;
    let step = inc as usize;
    let len  = 1 + (n - 1) * step;

    let mut z = vec![0.0f64; 2 * len];
    for i in 0..n {
        let k = i * step;
        z[2*k]     = (i as f64 * 0.19).sin() * 0.5;
        z[2*k + 1] = (i as f64 * 0.23).cos() * 0.5;
    }

    let i_big = 137 * step;
    z[2*i_big]     = -80.0;
    z[2*i_big + 1] =  60.0;

    let rusty1 = izamax(n, &z, inc);
    let cblas1 = unsafe { 
        cblas_izamax(
            n as i32, 
            z.as_ptr().cast::<[f64; 2]>(), 
            inc as i32
        ) 
    } as usize;

    assert_eq!(rusty1, cblas1 + 1);
}

