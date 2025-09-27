# CORAL

**COre** **R**ust **A**rchitecture for **L**inear-algebra — *pure Rust BLAS*

This is a work-in-progress project aiming to reach *~70–80%* of BLAS performance while remaining 100% Rust.

Core Level-1 and Level-2 routines are implemented. Some outperform
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas/)
for *contiguous, unit-stride layouts* thanks to AArch64-specialized kernels.

Level-3 routines (including GEMM) are not implemented yet.

> **Architecture:** Optimized currently for **AArch64**.

---

## Preliminary benchmarks

Early microbenchmarks (contiguous, unit-stride) suggest competitive performance on AArch64 up to $n \simeq$ 2500. 
Below are two example plots from [benches/plots/](benches/plots/). 

### SGEMV (TRANSPOSE)
![SGEMV TRANSPOSE](benches/plots/SGEMV%20TRANSPOSE.png)

### STRSV (UPPER, NOTRANSPOSE)
![STRSV UPPER NOTRANSPOSE](benches/plots/STRSV%20UPPER%20NOTRANSPOSE.png)

Apple Accelerate is outrageously fast for GEMV routines. It is not included in
the above SGEMV plot because it is an order-of-magnitude faster, reaching ~400 GFLOP/s and masks my improvement over OpenBLAS. 
However, for triangular solves my implementations exceed both Accelerate and OpenBLAS (for $n \leq$ 2500).

*These results are preliminary and subject to change as kernels and packing strategies evolve.*

