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

/// returns mutable pointer to A[i, j]
/// used by GER routines 
#[inline(always)] 
pub(crate) fn a_ij_mutable_f32( 
    matrix  : *mut f32, 
    i       : usize, 
    j       : usize, 
    inc_row : usize, 
    inc_col : usize, 
) -> *mut f32 { 
    unsafe { 
        matrix.add(i * inc_row + j * inc_col) 
    }
}
