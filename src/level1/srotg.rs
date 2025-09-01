#[inline]
pub fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    let roe = if a.abs() > b.abs() { *a } else { *b };
    let scale = a.abs() + b.abs();

    // degenerate quick return  
    if scale == 0.0 {
        *c = 1.0;
        *s = 0.0;
        *a = 0.0; 
        *b = 0.0; 
        return;
    }

    let mut r = scale * ((*a / scale).powi(2) + (*b / scale).powi(2)).sqrt();
    if roe < 0.0 { r = -r; }

    *c = *a / r;
    *s = *b / r;

    let mut z = 1.0f32;
    if a.abs() > b.abs() {
        z = *s;
    }
    if b.abs() >= a.abs() && *c != 0.0 {
        z = 1.0 / *c;
    }

    *a = r; 
    *b = z; 
}

