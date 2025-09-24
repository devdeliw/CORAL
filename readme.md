# CORAL

**COre** **R**ust **A**lgebra **Library** is a pure-Rust implementation of BLAS routines. 

I am not knowledgable enough *yet* to make it as fast as traditional BLAS libraries, but I hope
to reach about **70–80% of their performance**, while making it written entirely in Rust.

So far core Level 1 and Level 2 routines are written. Most outperform
[openblas](https://github.com/OpenMathLib/OpenBLAS) *for contiguous, unit-stride memory
layouts* because coral is specialized for aarch64. 

Level 3 routines, including GEMM, have not been written yet. 

Currently, it is optimized only for **AArch64**.

