# CORAL

**COre** **R**ust **A**rchitecture for **L**inear-algebra — *pure Rust BLAS*

This is a project aiming to be comparable to BLAS performance while remaining 100% Rust.

Core Level-1 and Level-2 routines are implemented. Some outperform
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and 
[Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas/)
on Apple Silicon. 

Level-3 routines are being written. 

> **Architecture:** Optimized currently for **AArch64 only**.

---

## Preliminary benchmarks

Early microbenchmarks (contiguous) suggest competitive performance up to 
$n \simeq 2.5 \times 10^3$.  Below are example `GEMM` benchmarks from 
[benches/plots/](benches/plots/). 

The following benchmarks were performed single-threaded and on Apple Silicon. 

### DGEMM 
![DGEMM NN](benches/plots/DGEMM_NOTRANSPOSE_x_NOTRANSPOSE.png)

### SGEMM 
![SGEMM NN](benches/plots/SGEMM_NOTRANSPOSE_x_NOTRANSPOSE.png)


The transpose GEMM variants are similar. Hence, for `?GEMM` my implementations 
on AArch64 are well-comparable. Additionally, for many critical Level 2 routines, 
notably `SGEMV/STRMV` (transpose)  and `STRSV` (no transpose), 
my implementations do perform better than OpenBLAS, but are *extremely* slower than 
Apple Accelerate. These benchmarks, along with the transpose variants of GEMM are 
also in [benches/plots](benches/plots/). 


*These results are preliminary and subject to change as kernels and packing strategies evolve.*

