pub mod enums; 
pub(crate) mod panel_packing; 
pub(crate) mod vector_packing;
pub(crate) mod matrix_ij;
pub(crate) mod assert_length_helpers;
pub(crate) mod trmv_kernels;

pub mod sgemv; 
pub(crate) mod sgemv_notranspose;
pub(crate) mod sgemv_transpose;
pub mod dgemv;
pub(crate) mod dgemv_notranspose;
pub(crate) mod dgemv_transpose; 

pub mod ssymv;
pub mod dsymv;

pub mod strmv; 
pub(crate) mod strumv; 
pub(crate) mod strlmv; 
pub mod dtrmv; 
pub(crate) mod dtrumv; 
pub(crate) mod dtrlmv; 

pub mod strsv; 
pub(crate) mod strusv; 
pub(crate) mod strlsv; 
pub mod dtrsv; 
pub(crate) mod dtrusv; 
pub(crate) mod dtrlsv; 

pub mod sger;
pub mod ssyr; 
pub mod ssyr2;

pub mod dger; 
pub mod dsyr;  
pub mod dsyr2;







