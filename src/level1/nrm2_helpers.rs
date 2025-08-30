#[inline(always)] 
pub(crate) fn upd_f32(scale: &mut f32, ssq: &mut f32, cmax: f32, cssq: f32) {
    if *scale < cmax {

        let r  = *scale / cmax;
        *ssq   = *ssq * (r*r) + cssq;
        *scale = cmax;

    } else if *scale > 0.0 {

        let r = cmax / *scale;
        *ssq += cssq * (r*r);

    } else {

        *scale = cmax;
        *ssq   = cssq;

    }
}

#[inline(always)] 
pub(crate) fn upd_f64(scale: &mut f64, ssq: &mut f64, cmax: f64, cssq: f64) {
    if *scale < cmax {

        let r  = *scale / cmax;
        *ssq   = *ssq * (r*r) + cssq;
        *scale = cmax;

    } else if *scale > 0.0 {

        let r = cmax / *scale;
        *ssq += cssq * (r*r);

    } else {

        *scale = cmax;
        *ssq   = cssq;

    }
} 
