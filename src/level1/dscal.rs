//! Scales a double precision vector by a real scalar.
//!
//! This function implements the BLAS [`dscal`] routine, multiplying each element 
//! of the input vector `x` by the scalar `alpha` over `n` entries with a specified stride.
//!
//! # Arguments 
//! - `n`     : Number of elements to scale. 
//! - `alpha` : Scalar multiplier. 
//! - `x`     : Input/output slice containing vector elements. 
//! - `incx`  : Stride between consecutive elements of `x`. 
//!
//! # Returns 
//! - Nothing. The contents of `x` are updated in place as `x[i] = alpha * x[i]`. 
//!
//! # Notes 
//! - For `incx == 1`, [`dscal`] does not perform SIMD or unrolling; LLVM is enough.  
//! - For non-unit strides, LLVM does not vectorize and the function falls back to a scalar loop. 
//! - If `n == 0` or `incx <= 0`, the function returns immediately; no slice modification. 
//!
//! # Author 
//! Deval Deliwala


#[inline(always)]
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: isize) {
    // quick return
    if n == 0 || incx <= 0 { return; }

    // fast path
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p  = p.add(1);
                i += 1;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let mut p = x.as_mut_ptr();
            let step = incx as usize;
            let mut i = 0usize;
            while i < n {
                *p = *p * alpha;
                p = p.add(step);
                i += 1;
            }
        }
    }
}
