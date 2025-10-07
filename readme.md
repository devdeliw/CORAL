# CORAL

**COre** **R**ust **A**rchitecture for **L**inear-algebra — *pure Rust BLAS*

This is a work-in-progress project aiming to reach *~70–80%* of BLAS performance while remaining 100% Rust.

Core Level-1 and Level-2 routines are implemented. Some outperform
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas/)
on Apple Silicon. 

Level-3 routines (including GEMM) are being written. 

> **Architecture:** Optimized currently for **AArch64**.

---

## Preliminary benchmarks

Early microbenchmarks (contiguous, unit-stride) suggest competitive performance on AArch64 up to $n \simeq 2.5 \times 10^3$. 
Below are two example plots from [benches/plots/](benches/plots/). 

### DGEMM (NO TRANSPOSE X NO TRANSPOSE) 
![DGEMM NN](benches/plots/DGEMM%20NOTRANSPOSE%20x%20NOTRANSPOSE.png)

### SGEMV (TRANSPOSE)
![SGEMV TRANSPOSE](benches/plots/SGEMV%20TRANSPOSE.png)

Apple Accelerate uses Apple-specific magic (AMX/matrix units) beyond
NEON intrinsics. It's not included in the above plots because it's an 
order-of-magnitude faster, reaching ~400GFLOP/s and masks any comparison between
CORAL and OpenBLAS. 

### STRSV (UPPER, NOTRANSPOSE)
![STRSV UPPER NOTRANSPOSE](benches/plots/STRSV%20UPPER%20NOTRANSPOSE.png)

However, for triangular solves my implementations exceed both Accelerate and OpenBLAS
(for $n \leq 2.5 \times 10^3$).

*These results are preliminary and subject to change as kernels and packing strategies evolve.*

