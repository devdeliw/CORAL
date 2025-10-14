pub(crate) mod f32_packers; 
pub(crate) mod f64_packers;
pub(crate) mod c32_packers; 
pub(crate) mod c64_packers;

pub(crate) mod f32_macro_kernel; 
pub(crate) mod f64_macro_kernel;
pub(crate) mod c32_macro_kernel; 
pub(crate) mod c64_macro_kernel; 

pub(crate) mod microkernel; 

pub mod sgemm; 
pub(crate) mod sgemm_nn;
pub(crate) mod sgemm_nt; 
pub(crate) mod sgemm_tn; 
pub(crate) mod sgemm_tt;

pub mod dgemm;
pub(crate) mod dgemm_nn;
pub(crate) mod dgemm_nt; 
pub(crate) mod dgemm_tn; 
pub(crate) mod dgemm_tt;

pub mod cgemm; 
pub(crate) mod cgemm_nn;
pub(crate) mod cgemm_nt; 
pub(crate) mod cgemm_tn; 
pub(crate) mod cgemm_tt;
pub(crate) mod cgemm_nc;
pub(crate) mod cgemm_tc;
pub(crate) mod cgemm_cn;
pub(crate) mod cgemm_ct;
pub(crate) mod cgemm_cc;

pub mod zgemm; 

