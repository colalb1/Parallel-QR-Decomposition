# Parallel-QR-Decomposition

## Why I am doing this project:

I want to learn more about parallelizing programs for high-performance numerical computations on CPUs.

## What this project is:

Writing parallel QR decomposition algorithms. Part of this will be implementing the GPU-limited algorithms from [this repo](https://github.com/HybridScale/CholeskyQR2-IM) and continuing to other algorithms (like TLQR or Householder (want stable parallel option)) after I get my bearings.

Start by reading [this paper](https://arxiv.org/abs/2405.04237) for background. Now you may continue reading.

## Why you should care:

Algorithms like this significantly speed up least-squares problems and eigenvalue computations for [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), among other relevant applications. Basically, data scientists will waste less time waiting for models to finish computing and can iterate/improve solutions faster.

What this means for business people who don't care about all that academic stuff is that your engineers can iterate on solutions faster and make you more money.

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
12. Fix ~~CQRGS~~, dCQRGS, mCQR2GS
13. Speed test: CQR2GS, dCQRbGS, mCQR2GS
14. Speed refactor
    a. Goal is to make these significantly faster than CQR while preserving orthogonal stability gains
    b. Faster than LAPACK, Intel MKL, Eigen, Armadillo
    c. Flame graph to find overhead
    d. Write out algo in ONE function to find computation reductions
    e. Code cleanup for speed and RAM optimization (remove unnecessary temp variables, use setting with complex indexing instead of summing/aggs)
    f. Use `const`, `constexpr`, and proper C++ objects for clarity and speed
    g. Mathematical manipulations/simplifications
16. ~~Delete dead code/comments~~
17. Official speed and accuracy tests (ones where you can make graphs)
18. Write description
