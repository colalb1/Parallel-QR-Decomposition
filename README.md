# Parallel-QR-Decomposition

BASIC INTRODUCTION GOES HERE

## Precursor

### Why I am doing this project:

I want to learn more about parallelizing programs for high-performance numerical computations on CPUs.

### What this project is:

Writing parallel QR decomposition algorithms. Part of this will be implementing the GPU-limited algorithms from [this repo](https://github.com/HybridScale/CholeskyQR2-IM) and continuing to other algorithms (like TLQR or Householder (want stable parallel option)) after I get my bearings.

Start by reading [this paper](https://arxiv.org/abs/2405.04237) for background. Now you may continue reading.

### Why you should care:

Algorithms like this significantly speed up least-squares problems and eigenvalue computations for [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), among other relevant applications. Basically, data scientists will waste less time waiting for models to finish computing and can iterate/improve solutions faster.

What this means for business people who don't care about all that academic stuff is that your engineers can iterate on solutions faster and make you more money.

## Files

## Background

### QR Factorization

### Cholesky Decomposition

### Gram-Schmidt Orthogonalization

### Tall-and-Skinny Matrices

### Ill-Conditioned Matrices

### Condition Numbers

### Floating-Point Arthimetic

### Distributed Memory Architecture

### Weak and Strong Scaling Performance Analysis

### Fine and Course-Grained Parallelization



## Algorithms

### Cholesky QR (CQR)
- Performs QR decomposition using Cholesky factorization.

### Parallel Cholesky QR
- Parallel implementation of the Cholesky QR algorithm.

### Cholesky QR 2 (CQR2)
- Performs QR decomposition using two iterations of Cholesky QR for improved numerical accuracy.

### Shifted Cholesky QR (sCQR)
- Performs QR decomposition with a stability shift applied to the diagonal of the Gram matrix.

### Parallel Shifted Cholesky QR
- Parallel implementation of the Shifted Cholesky QR algorithm.

### Shifted Cholesky QR 3 (sCQR3)
- Performs QR decomposition using a combination of shifted Cholesky QR and Cholesky QR 2.

### Cholesky QR with Gram-Schmidt (CQRGS)
- Combines Cholesky QR with Gram-Schmidt orthogonalization for block-wise processing.

### Cholesky QR2 with Gram-Schmidt (CQR2GS)
- Performs two iterations of Cholesky QR with Gram-Schmidt orthogonalization.

### Distributed Cholesky QR with Gram-Schmidt (dCQRGS)
- Distributed implementation of Cholesky QR with Gram-Schmidt orthogonalization.

### Modified Cholesky QR2 with Gram-Schmidt (mCQR2GS)
- Modified version of Cholesky QR2 with Gram-Schmidt, incorporating reorthogonalization and parallel processing.

## Conclusion

Much more important work came up, and I accomplished the minimal acceptable output for this project.

## TODO:

1. ~~Figure out how to run C++ programs on PC without breaking~~
2. ~~Implement simple parallel applications (for loops, other basics)~~
3. ~~Implement iterative Cholesky (used for speed comparison)~~
4. ~~Implement parallel Cholesky (used for speed comparison)~~
5. ~~Implement CholeskyQR2~~
6. ~~Implement sCQR3~~
7. ~~Implement CholeskyQR2 with Gram-Schmidt (CQRGS, CQR2GS)~~
8. ~~Implement Distributed Cholesky QR with blocked GS (dCQRbGS)~~
9. ~~Implement Modified Cholesky QRwGS~~
10. ~~Implement mCQR2GS (test THEN potentiall revert indexing, parallelized panels if computation slower)~~
11. ~~Accuracy test: CholeskyQR2, sCQR, sCQR3, CQRGS, CQR2GS, dCQRGS, mCQR2GS~~
12. ~~Fix CQRGS, dCQRGS, mCQR2GS~~
13. ~~Speed test: CQR2GS, dCQRbGS, mCQR2GS (run the tests)~~
14. Speed refactor

    a. ~~Goal is to make these significantly faster than CQR while preserving orthogonal stability gains~~

    b. ~~Flame graph to find overhead~~

    c. ~~Write out algo in ONE function to find computation reductions~~

    d. Code speed optimization
    
        i. Own functions (see flame graph)
            
            1. Cholesky QR2 with Gram Schmidt

                a. Use `const`, `constexpr`, and proper C++ objects for clarity and speed
    
                b. Mathematical manipulations/simplifications

            2. Modified Cholesky QR2 with Gram Schmidt
                
                a. Use `const`, `constexpr`, and proper C++ objects for clarity and speed
    
                b. Mathematical manipulations/simplifications

            3. Parallel CQR
                
                a. Use `const`, `constexpr`, and proper C++ objects for clarity and speed
    
                b. Mathematical manipulations/simplifications

        ii. Comparison functions

            1. LAPACK
            
            2. Intel MKL
            
            3. Eigen
            
            4. Armadillo

    e. After editing in helper, insert updated functions back into original file(s).

16. GENERAL CODE CLEANUP
17. Write description


I am not optimizing `distributed_cholesky_QR_w_gram_schmidt` because it was meant to run on a CPU/GPU mix, and I am only running on a CPU for this project.
