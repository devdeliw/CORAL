#[inline]
fn ensure_len(buf: &mut Vec<f32>, n: usize) {
    if buf.len() != n {
        buf.resize(n, 0.0);
    }
}

#[inline]
pub(crate) unsafe fn pack_and_scale_x_f32(
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: isize,        
    dst: &mut Vec<f32>, // unit-stride, scaled x
) { unsafe { 
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    if n == 0 { return; }
    ensure_len(dst, n);

    if alpha == 0.0 {
        core::ptr::write_bytes(dst.as_mut_ptr(), 0, n * core::mem::size_of::<f32>());
        return;
    }
    if incx == 1 && alpha == 1.0 {
        core::ptr::copy_nonoverlapping(x.as_ptr(), dst.as_mut_ptr(), n);
        return;
    }

    let pd = dst.as_mut_ptr();

    // fast path 
    if incx == 1 {
        let a = vdupq_n_f32(alpha);
        let mut i = 0usize;
        while i + 16 <= n {
            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p.add(0));
            let x1 = vld1q_f32(p.add(4));
            let x2 = vld1q_f32(p.add(8));
            let x3 = vld1q_f32(p.add(12));

            vst1q_f32(pd.add(i + 0),  vmulq_f32(x0, a));
            vst1q_f32(pd.add(i + 4),  vmulq_f32(x1, a));
            vst1q_f32(pd.add(i + 8),  vmulq_f32(x2, a));
            vst1q_f32(pd.add(i + 12), vmulq_f32(x3, a));

            i += 16;
        }
        while i + 8 <= n {

            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p.add(0));
            let x1 = vld1q_f32(p.add(4));

            vst1q_f32(pd.add(i + 0), vmulq_f32(x0, a));
            vst1q_f32(pd.add(i + 4), vmulq_f32(x1, a));

            i += 8;
        }
        while i + 4 <= n {
            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p);

            vst1q_f32(pd.add(i), vmulq_f32(x0, a));

            i += 4;
        }
        while i < n {
            *pd.add(i) = alpha * *x.as_ptr().add(i);
            i += 1;
        }
    } else {
        // non-unit stride 
        let step  = incx.unsigned_abs() as usize;
        let mut i = 0usize;

        let mut idx  = if incx > 0 { 0usize as isize } else { ((n - 1) * step) as isize };
        let delta    = if incx > 0 { step as isize } else { -(step as isize) };

        while i + 4 <= n {
            let p0 = x.as_ptr().offset(idx + 0 * delta);
            let p1 = x.as_ptr().offset(idx + 1 * delta);
            let p2 = x.as_ptr().offset(idx + 2 * delta);
            let p3 = x.as_ptr().offset(idx + 3 * delta);

            *pd.add(i + 0) = alpha * *p0;
            *pd.add(i + 1) = alpha * *p1;
            *pd.add(i + 2) = alpha * *p2;
            *pd.add(i + 3) = alpha * *p3;

            idx += 4 * delta;
            i   += 4;
        }
        while i < n {
            let p = x.as_ptr().offset(idx);
            *pd.add(i) = alpha * *p;

            idx += delta;
            i   += 1;
        }
    }
}}

#[inline]
pub(crate) unsafe fn pack_y_to_unit_f32(
    m: usize,
    y: &[f32],
    incy: isize,        
    ybuf: &mut Vec<f32>,
) { unsafe {
    if m == 0 { return; }
    ensure_len(ybuf, m);

    let pd = ybuf.as_mut_ptr();

    if incy == 1 {
        core::ptr::copy_nonoverlapping(y.as_ptr(), pd, m);
        return;
    }

    let step  = incy.unsigned_abs() as usize;
    let mut i = 0usize;

    let mut idx  = if incy > 0 { 0usize as isize } else { ((m - 1) * step) as isize };
    let delta    = if incy > 0 { step as isize } else { -(step as isize) };

    while i + 4 <= m {
        let p0 = y.as_ptr().offset(idx + 0 * delta);
        let p1 = y.as_ptr().offset(idx + 1 * delta);
        let p2 = y.as_ptr().offset(idx + 2 * delta);
        let p3 = y.as_ptr().offset(idx + 3 * delta);

        *pd.add(i + 0) = *p0;
        *pd.add(i + 1) = *p1;
        *pd.add(i + 2) = *p2;
        *pd.add(i + 3) = *p3;

        idx += 4 * delta;
        i   += 4;
    }

    while i < m {
        *pd.add(i) = *y.as_ptr().offset(idx);
        idx += delta;
        i   += 1;
    }
}}

#[inline(always)]
pub(crate) unsafe fn copy_back_y_from_unit_f32(
    m   : usize,
    ybuf: &[f32],
    y   : &mut [f32],
    incy: isize,
) { unsafe {
    if m == 0 { return; }

    if incy > 0 {
        let step   = incy as usize;
        let mut py = y.as_mut_ptr();
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.add(step);
        }
    } else {
        let step   = (-incy) as usize;
        let mut py = y.as_mut_ptr().add((m - 1) * step);
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.sub(step);
        }
    }
}}


#[inline(always)]
pub(crate) unsafe fn pack_and_scale_x_f64(
    n    : usize,
    alpha: f64,
    x    : &[f64],
    incx : isize,
    dst  : &mut Vec<f64>, // updated scaled x
) { unsafe {
    dst.clear();
    dst.reserve_exact(n);
    dst.set_len(n);

    if n == 0 { return; }

    if incx > 0 {
        let step = incx as usize;
        let mut px = x.as_ptr();
        for i in 0..n {
            *dst.get_unchecked_mut(i) = alpha * *px;
            px = px.add(step);
        }
    } else {
        let step = (-incx) as usize;
        let mut px = x.as_ptr().add((n - 1) * step);
        for i in 0..n {
            *dst.get_unchecked_mut(i) = alpha * *px;
            px = px.sub(step);
        }
    }
}}

#[inline(always)]
pub(crate) unsafe fn pack_y_to_unit_f64(
    m   : usize,
    y   : &[f64],
    incy: isize,
    ybuf: &mut Vec<f64>, // updated unit stride y buf
) { unsafe {
    ybuf.clear();
    ybuf.reserve_exact(m);
    ybuf.set_len(m);

    if m == 0 { return; }

    if incy > 0 {
        let step = incy as usize;
        let mut py = y.as_ptr();
        for i in 0..m {
            *ybuf.get_unchecked_mut(i) = *py;
            py = py.add(step);
        }
    } else {
        let step = (-incy) as usize;
        let mut py = y.as_ptr().add((m - 1) * step);
        for i in 0..m {
            *ybuf.get_unchecked_mut(i) = *py;
            py = py.sub(step);
        }
    }
}}

#[inline(always)]
pub(crate) unsafe fn copy_back_y_from_unit_f64(
    m   : usize,
    ybuf: &[f64],
    y   : &mut [f64],
    incy: isize,
) { unsafe {
    if m == 0 { return; }

    if incy > 0 {
        let step   = incy as usize;
        let mut py = y.as_mut_ptr();
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.add(step);
        }
    } else {
        let step   = (-incy) as usize;
        let mut py = y.as_mut_ptr().add((m - 1) * step);
        for i in 0..m {
            *py = *ybuf.get_unchecked(i);
            py  = py.sub(step);
        }
    }
}}

