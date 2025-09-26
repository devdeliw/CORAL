//! Constructs a Givens rotation ROTG for double precision scalars.
//!
//! This function implements the BLAS [`drotg`] routine, computing parameters
//! `c` and `s` that define a Givens plane rotation such that
//!
//! ```text
//! [ c  s ] [ a ] = [ r ]
//! [-s  c ] [ b ]   [ 0 ]
//! ```
//!
//! The values of `a` and `b` are overwritten with `r` and `z`, where `r`
//! is the nonzero result and `z` encodes information for reconstructing
//! the rotation.
//!
//! # Arguments
//! - `a` : Input scalar, overwritten with `r`, the rotated value.
//! - `b` : Input scalar, overwritten with `z`, a parameter related to the rotation.
//! - `c` : Output scalar cosine component of the rotation.
//! - `s` : Output scalar sine component of the rotation.
//!
//! # Returns
//! - Nothing. The results are written in place to `a`, `b`, `c`, and `s`.
//!
//! # Notes
//! - If both `a` and `b` are zero, the routine sets `c = 1.0`, `s = 0.0`,
//!   and overwrites `a = b = 0.0`.
//! - The value `z` stored in `b` allows reconstruction of the rotation without
//!   recomputing `c` and `s` in some BLAS contexts.
//!
//! # Author
//! Deval Deliwala


#[inline]
pub fn drotg(a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) {
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

    let mut z = 1.0f64;
    if a.abs() > b.abs() {
        z = *s;
    }
    if b.abs() >= a.abs() && *c != 0.0 {
        z = 1.0 / *c;
    }

    *a = r; 
    *b = z; 
}


