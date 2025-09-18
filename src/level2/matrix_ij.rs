/// returns immutable pointer to A[i, j] 
/// used by TRMV routines 
#[inline(always)] 
pub(crate) fn a_ij_immutable_f32(
    matrix  : *const f32, 
    i       : usize, 
    j       : usize, 
    inc_row : usize, 
    inc_col : usize, 
) -> *const f32 { 
    unsafe { 
        matrix.add(i * inc_row + j * inc_col) 
    }
}
