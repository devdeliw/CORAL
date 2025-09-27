#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f32, 
    vfmaq_f32, 
    vst1q_f32, 
    vdupq_n_f32,

    vld1q_f64, 
    vfmaq_f64, 
    vst1q_f64, 
    vdupq_n_f64,

    vfmsq_f32, 
    vld2q_f32, 
    vst2q_f32, 
    float32x4x2_t,

    vfmsq_f64, 
    vld2q_f64, 
    vst2q_f64, 
    float64x2x2_t
};

/// Adds a scaled column vector to a buffer in-place using NEON intrinsics.
///
/// Each element in `buffer` is updated as:
/// `buffer[i] += column[i] * scale`
///
/// # Arguments
/// - `buffer`  (*mut f32)  : Mutable pointer to the destination buffer.
/// - `column`  (*const f32): Pointer to the source column vector.
/// - `n_rows`  (usize)     : Number of elements to process.
/// - `scale`   (f32)       : Scalar multiplier applied to each element of `column`.
#[inline(always)]
#[cfg(target_arch = "aarch64")] 
pub(crate) fn single_add_and_scale_f32( 
    buffer  : *mut f32, 
    column  : *const f32, 
    n_rows  : usize, 
    scale   : f32, 
) { 
    unsafe {
        // quick return 
        if n_rows == 0 { return; } 
        
        let alpha = vdupq_n_f32(scale);

        let mut i = 0;
        while i + 16 <= n_rows { 
            let a0 = vld1q_f32(column.add(i)); 
            let y0 = vld1q_f32(buffer.add(i)); 
            let y0 = vfmaq_f32(y0, a0, alpha); 
            vst1q_f32(buffer.add(i), y0);

            let a1 = vld1q_f32(column.add(i + 4)); 
            let y1 = vld1q_f32(buffer.add(i + 4)); 
            let y1 = vfmaq_f32(y1, a1, alpha); 
            vst1q_f32(buffer.add(i + 4), y1);

            let a2 = vld1q_f32(column.add(i + 8)); 
            let y2 = vld1q_f32(buffer.add(i + 8)); 
            let y2 = vfmaq_f32(y2, a2, alpha); 
            vst1q_f32(buffer.add(i + 8), y2);

            let a3 = vld1q_f32(column.add(i + 12)); 
            let y3 = vld1q_f32(buffer.add(i + 12)); 
            let y3 = vfmaq_f32(y3, a3, alpha); 
            vst1q_f32(buffer.add(i + 12), y3);

            i += 16;
        }

        while i + 4 <= n_rows { 
            let a = vld1q_f32(column.add(i)); 
            let y = vld1q_f32(buffer.add(i)); 
            let y = vfmaq_f32(y, a, alpha); 
            vst1q_f32(buffer.add(i), y);

            i += 4; 
        }

        while i < n_rows { 
            *buffer.add(i) += *column.add(i) * scale; 

            i += 1
        }
    }
}

/// Adds a scaled column vector to a buffer in-place using NEON intrinsics.
///
/// Each element in `buffer` is updated as:
/// `buffer[i] += column[i] * scale`
///
/// # Arguments
/// - `buffer`  (*mut f64)  : Mutable pointer to the destination buffer.
/// - `column`  (*const f64): Pointer to the source column vector.
/// - `n_rows`  (usize)     : Number of elements to process.
/// - `scale`   (f64)       : Scalar multiplier applied to each element of `column`.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn single_add_and_scale_f64( 
    buffer  : *mut f64, 
    column  : *const f64, 
    n_rows  : usize, 
    scale   : f64, 
) { 
    unsafe {
        // quick return 
        if n_rows == 0 { return; } 
        
        let alpha = vdupq_n_f64(scale);

        let mut i = 0;
        while i + 8 <= n_rows { 
            let a0 = vld1q_f64(column.add(i)); 
            let y0 = vld1q_f64(buffer.add(i)); 
            let y0 = vfmaq_f64(y0, a0, alpha); 
            vst1q_f64(buffer.add(i), y0);

            let a1 = vld1q_f64(column.add(i + 2)); 
            let y1 = vld1q_f64(buffer.add(i + 2)); 
            let y1 = vfmaq_f64(y1, a1, alpha); 
            vst1q_f64(buffer.add(i + 2), y1);

            let a2 = vld1q_f64(column.add(i + 4)); 
            let y2 = vld1q_f64(buffer.add(i + 4)); 
            let y2 = vfmaq_f64(y2, a2, alpha); 
            vst1q_f64(buffer.add(i + 4), y2);

            let a3 = vld1q_f64(column.add(i + 6)); 
            let y3 = vld1q_f64(buffer.add(i + 6)); 
            let y3 = vfmaq_f64(y3, a3, alpha); 
            vst1q_f64(buffer.add(i + 6), y3);

            i += 8;
        }

        while i + 2 <= n_rows { 
            let a = vld1q_f64(column.add(i)); 
            let y = vld1q_f64(buffer.add(i)); 
            let y = vfmaq_f64(y, a, alpha); 
            vst1q_f64(buffer.add(i), y);

            i += 2; 
        }

        while i < n_rows { 
            *buffer.add(i) += *column.add(i) * scale; 

            i += 1
        }
    }
}

/// Adds a scaled complex column to a complex buffer in-place using NEON intrinsics.
///
/// For each complex element i:
/// `buffer[i] += column[i] * (scale_re + i * scale_im)`
///
/// # Arguments
/// - `buffer`  (*mut f32)  : Mutable pointer to the destination buffer.
/// - `column`  (*const f32): Pointer to the source column vector.
/// - `n_rows`  (usize)     : Number of elements to process; complex units. 
/// - `scale`   ([f32; 2])  : `[re, im`] complex scale
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn single_add_and_scale_c32(
    buffer   : *mut f32,   
    column   : *const f32, 
    n_rows   : usize,      
    scale    : [f32; 2]
) {

    let scale_re = scale[0]; 
    let scale_im = scale[1]; 

    unsafe {
        if n_rows == 0 { return; }

        let ar_v = vdupq_n_f32(scale_re);
        let ai_v = vdupq_n_f32(scale_im);

        let mut i = 0usize;

        while i + 8 <= n_rows {
            let p0 = 2 * i;

            let a01 = vld2q_f32(column.add(p0));
            let y01 = vld2q_f32(buffer.add(p0));

            let ar0     = a01.0;
            let ai0     = a01.1;
            let mut yr0 = y01.0;
            let mut yi0 = y01.1;

            yr0 = vfmaq_f32(yr0, ar_v, ar0);
            yr0 = vfmsq_f32(yr0, ai_v, ai0);
            yi0 = vfmaq_f32(yi0, ar_v, ai0);
            yi0 = vfmaq_f32(yi0, ai_v, ar0);

            vst2q_f32(buffer.add(p0), float32x4x2_t(yr0, yi0));

            let p1  = p0 + 8;
            let a23 = vld2q_f32(column.add(p1));
            let y23 = vld2q_f32(buffer.add(p1));

            let ar1     = a23.0;
            let ai1     = a23.1;
            let mut yr1 = y23.0;
            let mut yi1 = y23.1;

            yr1 = vfmaq_f32(yr1, ar_v, ar1);
            yr1 = vfmsq_f32(yr1, ai_v, ai1);
            yi1 = vfmaq_f32(yi1, ar_v, ai1);
            yi1 = vfmaq_f32(yi1, ai_v, ar1);

            vst2q_f32(buffer.add(p1), float32x4x2_t(yr1, yi1));

            i += 8;
        }

        while i + 4 <= n_rows {
            let p = 2 * i;

            let a01 = vld2q_f32(column.add(p));
            let y01 = vld2q_f32(buffer.add(p));

            let ar0     = a01.0;
            let ai0     = a01.1;
            let mut yr0 = y01.0;
            let mut yi0 = y01.1;

            yr0 = vfmaq_f32(yr0, ar_v, ar0);
            yr0 = vfmsq_f32(yr0, ai_v, ai0);
            yi0 = vfmaq_f32(yi0, ar_v, ai0);
            yi0 = vfmaq_f32(yi0, ai_v, ar0);

            vst2q_f32(buffer.add(p), float32x4x2_t(yr0, yi0));

            i += 4;
        }

        while i < n_rows {
            let p = 2 * i;

            let ar = *column.add(p);
            let ai = *column.add(p + 1);

            let yrp = buffer.add(p);
            let yip = buffer.add(p + 1);

            *yrp += ar * scale_re - ai * scale_im;
            *yip += ar * scale_im + ai * scale_re;

            i += 1;
        }
    }
}


/// Adds a scaled complex column to a complex buffer in-place using NEON intrinsics.
///
/// For each complex element i:
/// `buffer[i] += column[i] * (scale_re + i * scale_im)`
///
/// # Arguments
/// - `buffer`  (*mut f64)  : Mutable pointer to the destination buffer.
/// - `column`  (*const f64): Pointer to the source column vector.
/// - `n_rows`  (usize)     : Number of elements to process; complex units. 
/// - `scale`   ([f64; 2])  : `[re, im`] complex scale
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn single_add_and_scale_c64(
    buffer   : *mut f64,   
    column   : *const f64, 
    n_rows   : usize,      
    scale    : [f64; 2]
) {

    let scale_re = scale[0]; 
    let scale_im = scale[1]; 

    unsafe {
        if n_rows == 0 { return; }

        let ar_v = vdupq_n_f64(scale_re);
        let ai_v = vdupq_n_f64(scale_im);

        let mut i = 0usize;

        while i + 4 <= n_rows {
            let p0 = 2 * i;

            let a01 = vld2q_f64(column.add(p0));
            let y01 = vld2q_f64(buffer.add(p0));

            let ar0     = a01.0;
            let ai0     = a01.1;
            let mut yr0 = y01.0;
            let mut yi0 = y01.1;

            yr0 = vfmaq_f64(yr0, ar_v, ar0);
            yr0 = vfmsq_f64(yr0, ai_v, ai0);
            yi0 = vfmaq_f64(yi0, ar_v, ai0);
            yi0 = vfmaq_f64(yi0, ai_v, ar0);

            vst2q_f64(buffer.add(p0), float64x2x2_t(yr0, yi0));

            let p1  = p0 + 4;
            let a23 = vld2q_f64(column.add(p1));
            let y23 = vld2q_f64(buffer.add(p1));

            let ar1     = a23.0;
            let ai1     = a23.1;
            let mut yr1 = y23.0;
            let mut yi1 = y23.1;

            yr1 = vfmaq_f64(yr1, ar_v, ar1);
            yr1 = vfmsq_f64(yr1, ai_v, ai1);
            yi1 = vfmaq_f64(yi1, ar_v, ai1);
            yi1 = vfmaq_f64(yi1, ai_v, ar1);

            vst2q_f64(buffer.add(p1), float64x2x2_t(yr1, yi1));

            i += 4;
        }

        while i + 2 <= n_rows {
            let p = 2 * i;

            let a01 = vld2q_f64(column.add(p));
            let y01 = vld2q_f64(buffer.add(p));

            let ar0     = a01.0;
            let ai0     = a01.1;
            let mut yr0 = y01.0;
            let mut yi0 = y01.1;

            yr0 = vfmaq_f64(yr0, ar_v, ar0);
            yr0 = vfmsq_f64(yr0, ai_v, ai0);
            yi0 = vfmaq_f64(yi0, ar_v, ai0);
            yi0 = vfmaq_f64(yi0, ai_v, ar0);

            vst2q_f64(buffer.add(p), float64x2x2_t(yr0, yi0));

            i += 2;
        }

        while i < n_rows {
            let p = 2 * i;

            let ar = *column.add(p);
            let ai = *column.add(p + 1);

            let yrp = buffer.add(p);
            let yip = buffer.add(p + 1);

            *yrp += ar * scale_re - ai * scale_im;
            *yip += ar * scale_im + ai * scale_re;

            i += 1;
        }
    }
}


