pub mod enums; 
pub(crate) mod panel_packing; 
pub(crate) mod vector_packing;
pub(crate) mod matrix_ij;
pub(crate) mod assert_length_helpers;

pub mod sgemv; 
pub(crate) mod sgemv_notranspose;
pub(crate) mod sgemv_transpose;

pub mod dgemv;
pub(crate) mod dgemv_notranspose;
pub(crate) mod dgemv_transpose; 

pub mod ssymv; 

pub mod strmv; 
pub(crate) mod trmv_kernels;
pub(crate) mod strumv; 
pub(crate) mod strlmv; 

pub mod strsv; 
pub(crate) mod strusv; 
pub(crate) mod strlsv; 

pub mod sger;
pub mod ssyr; 
pub mod ssyr2; 







