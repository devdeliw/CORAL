pub(crate) mod panel_packing; 
pub(crate) mod vector_packing;
pub(crate) mod matrix_ij;
pub(crate) mod assert_length_helpers;
pub(crate) mod trmv_kernels;

pub(crate) mod sgemv_notranspose;
pub(crate) mod sgemv_transpose;
pub(crate) mod dgemv_notranspose;
pub(crate) mod dgemv_transpose;
pub(crate) mod cgemv_notranspose; 
pub(crate) mod cgemv_transpose; 
pub(crate) mod cgemv_conjtranspose;
pub(crate) mod zgemv_notranspose; 
pub(crate) mod zgemv_transpose; 
pub(crate) mod zgemv_conjtranspose;
pub mod sgemv; 
pub mod dgemv;
pub mod cgemv; 
pub mod zgemv; 

pub(crate) mod strumv; 
pub(crate) mod strlmv; 
pub(crate) mod dtrumv; 
pub(crate) mod dtrlmv; 
pub(crate) mod ctrumv; 
pub(crate) mod ctrlmv; 
pub(crate) mod ztrumv; 
pub(crate) mod ztrlmv; 
pub mod strmv; 
pub mod dtrmv; 
pub mod ctrmv; 
pub mod ztrmv;

pub(crate) mod strusv; 
pub(crate) mod strlsv; 
pub(crate) mod dtrusv; 
pub(crate) mod dtrlsv;
pub(crate) mod ctrusv; 
pub(crate) mod ctrlsv; 
pub(crate) mod ztrusv; 
pub(crate) mod ztrlsv; 
pub mod strsv; 
pub mod dtrsv; 
pub mod ctrsv; 
pub mod ztrsv; 

pub mod ssymv;
pub mod dsymv;
pub mod chemv; 
pub mod zhemv; 

pub mod sger;
pub mod dger; 
pub mod cgeru; 
pub mod cgerc; 
pub mod zgeru; 
pub mod zgerc; 

pub mod ssyr;
pub mod dsyr; 
pub mod cher; 
pub mod zher; 

pub mod ssyr2;
pub mod dsyr2;
pub mod cher2; 
pub mod zher2; 







