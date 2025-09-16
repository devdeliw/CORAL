#[cfg(target_arch = "aarch64")] 
use core::arch::aarch64::{
    vld1q_f32, 
    vmlaq_f32, 
    vst1q_f32, 
    vdupq_n_f32, 
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
pub(crate) fn single_add_and_scale( 
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
            let y0 = vmlaq_f32(y0, a0, alpha); 
            vst1q_f32(buffer.add(i), y0);

            let a1 = vld1q_f32(column.add(i + 4)); 
            let y1 = vld1q_f32(buffer.add(i + 4)); 
            let y1 = vmlaq_f32(y1, a1, alpha); 
            vst1q_f32(buffer.add(i + 4), y1);

            let a2 = vld1q_f32(column.add(i + 8)); 
            let y2 = vld1q_f32(buffer.add(i + 8)); 
            let y2 = vmlaq_f32(y2, a2, alpha); 
            vst1q_f32(buffer.add(i + 8), y2);

            let a3 = vld1q_f32(column.add(i + 12)); 
            let y3 = vld1q_f32(buffer.add(i + 12)); 
            let y3 = vmlaq_f32(y3, a3, alpha); 
            vst1q_f32(buffer.add(i + 12), y3);

            i += 16;
        }

        while i + 4 <= n_rows { 
            let a = vld1q_f32(column.add(i)); 
            let y = vld1q_f32(buffer.add(i)); 
            let y = vmlaq_f32(y, a, alpha); 
            vst1q_f32(buffer.add(i), y);

            i += 4; 
        }

        while i < n_rows { 
            *buffer.add(i) += *column.add(i) * scale; 

            i += 1
        }
    }
}
