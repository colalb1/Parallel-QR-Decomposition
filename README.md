# Parallel-QR-Decomposition

Course-grained parallel thin-QR decomposition algorithms for tall-and-skinny matrices on a CPU in C++ using OpenMP.

## Precursor

### Why I did this project:

I wanted to learn more about parallelizing programs for high-performance numerical computations on CPUs. The [authors' Github](https://github.com/HybridScale/CholeskyQR2-IM) that these algorithms are originally based on contains [heterogeneous](https://www.intel.com/content/www/us/en/developer/articles/technical/efficient-heterogenous-parallel-programming-openmp.html) versions that are relatively difficult to understand without first seeing the pseudocode. Thus, this provided an opportunity to mesh my math and HPC interests to learn parallelization in practice, OpenMP, and do a QR decomposition math-refresh while providing some user-friendly(er) code.

The concepts and insights of this project are not novel, but I wanted to implement numerical algorithms from literature as a "warm-up" to a very interesting project that I begin working on very soon (and to show my C++ competence). [This is a hint](https://en.wikipedia.org/wiki/Asian_option) at said project's topic.

### What this project is:

C++ implementation of novel parallel QR decomposition algorithms from [this paper](https://arxiv.org/abs/2405.04237). I will implement the GPU-limited algorithms from [its repository](https://github.com/HybridScale/CholeskyQR2-IM) (**EDIT AFTER IMPLEMENTATION**: the GPU-limited algorithms were *VERY* slow as they were meant for GPUs).

Start by reading [this paper](https://arxiv.org/abs/2405.04237) for background. You may continue reading now.

### Why you should care:

Parallel algorithms like these significantly speed up least-squares regression and eigenvalue computations for [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), among [other relevant applications](https://people.duke.edu/~hpgavin/SystemID/References/Tam-QR-history-2010.pdf). Basically, data scientists will waste less time waiting for models to finish training and can iterate/improve solutions faster.

What this means for business people who don't care about any of that high-performance-numerical-computing-math stuff: the computer is faster and your engineers can make you more money.

## File Navigation

[exploration](https://github.com/colalb1/Parallel-QR-Decomposition/tree/main/exploration): I tinker with OpenMP to make sure it works on my PC.

[utils](https://github.com/colalb1/Parallel-QR-Decomposition/tree/main/utils): Contains the master helper file with all algorithm implementations (some algorithms are helper functions; this is convenient for testing).

[tests](https://github.com/colalb1/Parallel-QR-Decomposition/tree/main/tests): Speed and orthogonality error tests. The raw `.csv` for the speed tests (in seconds) is in the `/data` path.* 

The remaining folders are named after their algorithm and contain a `.cpp` file with the respective implementation.

*I know it is bad practice to put the `.csv` in a Github repo, but its size is negligible and the raw data provides relevant insights.

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

### Fine vs Course-Grained Parallelization



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
