use core::slice; 
use crate::level2::enums::{CoralTranspose, CoralDiagonal}; 

// fused level1 
use crate::level1_special::{saxpyf::saxpyf, sdotf::sdotf}; 

// assert length helpers 
use crate::level1::assert_length_helpers::required_len_ok; 
use crate::level2::assert_length_helpers::required_len_ok_matrix; 


// for no transpose matrix
#[inline(always)] 
fn backward_substitution( 
    nb          : usize, 
    unit_diag   : bool, 
    mat_block   : *const f32, 
    lda         : usize, 
    x_block     : *mut f32,
) {
    if nb == 0 { return; } 

    // nb x nb 
    // contiguous backwards substitution 
    unsafe { 
        for i in (0..nb).rev() { 
            let mut sum = 0.0;  
            
            // start of row i 
            let row_ptr = mat_block.add(i); 
            
            // columns right of diagonal 
            for k in (i + 1)..nb { 
                // pointer to A[i, k] 
                let matrix_ik = *row_ptr.add(k * lda);

                // already solved 
                let x_k = *x_block.add(k); 

                sum += matrix_ik * x_k; 
            }

            let mut xi = *x_block.add(i) - sum; 

            if !unit_diag { 
                // divide by diagonal elements 
                let matrix_ii = *mat_block.add(i + i * lda); 
                xi /= matrix_ii 
            } 

            *x_block.add(i) = xi; 
        }
    }
}   
