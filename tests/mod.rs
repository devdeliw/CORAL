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
#[path = "level1/dot_tests.rs"] 
mod dot_tests; 
#[path = "level1/rot_tests.rs"] 
mod rot_tests;


// level 2 
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
#[path = "level2/ssyr_tests.rs"] 
mod ssyr_tests; 
#[path = "level2/ssyr2_tests.rs"] 
mod ssyr2_tests;



