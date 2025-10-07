// level 1 
#[path = "level1/axpy_tests.rs"]
mod axpy_tests; 
#[path = "level1/copy_tests.rs"]
mod copy_tests;
#[path = "level1/scal_tests.rs"] 
mod scal_tests;
#[path = "level1/swap_tests.rs"] 
mod swap_tests;
#[path = "level1/asum_tests.rs"] 
mod asum_tests;
#[path = "level1/nrm2_tests.rs"] 
mod nrm2_tests; 
#[path = "level1/dot_tests.rs" ] 
mod dot_tests; 
#[path = "level1/rot_tests.rs" ] 
mod rot_tests;
#[path = "level1/iamax_tests.rs"] 
mod iamax_tests;


// level 2 
//
// single precision 
//
#[path = "level2/sgemv_tests.rs"]
mod sgemv_tests;
#[path = "level2/strmv_tests.rs"] 
mod strmv_tests;
#[path = "level2/strsv_tests.rs"] 
mod strsv_tests; 
#[path = "level2/sger_tests.rs" ] 
mod sger_tests; 
#[path = "level2/ssymv_tests.rs"] 
mod ssymv_tests;
#[path = "level2/ssyr_tests.rs" ] 
mod ssyr_tests; 
#[path = "level2/ssyr2_tests.rs"] 
mod ssyr2_tests;
//
// double precision
//
#[path = "level2/dgemv_tests.rs"] 
mod dgemv_tests; 
#[path = "level2/dtrmv_tests.rs"] 
mod dtrmv_tests;
#[path = "level2/dtrsv_tests.rs"] 
mod dtrsv_tests; 
#[path = "level2/dger_tests.rs" ] 
mod dger_tests; 
#[path = "level2/dsymv_tests.rs"] 
mod dsymv_tests;
#[path = "level2/dsyr_tests.rs" ] 
mod dsyr_tests; 
#[path = "level2/dsyr2_tests.rs"] 
mod dsyr2_tests;
// complex single precision 
#[path = "level2/cgemv_tests.rs"]
mod cgemv_tests;
#[path = "level2/cger_tests.rs" ] 
mod cger_tests;
#[path = "level2/chemv_tests.rs"] 
mod chemv_tests; 
#[path = "level2/cher_tests.rs" ] 
mod cher_tests;
#[path = "level2/cher2_tests.rs"] 
mod cher2_tests;
#[path = "level2/ctrmv_tests.rs"] 
mod ctrmv_tests;
#[path = "level2/ctrsv_tests.rs"] 
mod ctrsv_tests;
//
// complex double precision 
// 
#[path = "level2/zgemv_tests.rs"] 
mod zgemv_tests; 
#[path = "level2/zger_tests.rs" ] 
mod zger_tests;
#[path = "level2/zhemv_tests.rs"] 
mod zhemv_tests;
#[path = "level2/zher_tests.rs" ] 
mod zher_tests; 
#[path = "level2/zher2_tests.rs"] 
mod zher2_tests; 
#[path = "level2/ztrmv_tests.rs"] 
mod ztrmv_tests;
#[path = "level2/ztrsv_tests.rs"] 
mod ztrsv_tests; 


// level 3 
//
// double precision 
//
#[path = "level3/dgemm_tests.rs"] 
mod dgemm_tests;
