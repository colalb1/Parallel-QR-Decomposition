# Parallel-QR-Decomposition

## Why I am doing this project:

I want to learn more about parallelizing programs for high-performance numerical computations on CPUs.

## What this project is:

Writing parallel QR decomposition algorithms. Part of this will be implementing the GPU-limited algorithms from [this repo](https://github.com/HybridScale/CholeskyQR2-IM) and continuing to other algorithms (like TLQR or Householder (want stable parallel option)) after I get my bearings.
I will probably rewrite the algorithm they have written for a CPU for sanity purposes.

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
10. Implement mCQR2GS
11. Accuracy test: CholeskyQR2, sCQR, sCQR3, CQRGS, CQR2GS, dCQRbGS, mCQR2GS
12. Speed test: CholeskyQR2, sCQR, sCQR3, CQRGS, CQR2GS, dCQRbGS, mCQR2GS
13. Implement TSQR
14. Implement Householder (used for speed comparison)
15. Implement parallel Householder
16. Delete dead code/comments
17. Code cleanup for speed and RAM optimization (remove unnecessary temp variables, use setting with complex indexing instead of summing/aggs)
18. Use `const` and proper C++ objects for clarity and speed

Average time for `cholesky_QR_decomposition`: 7.65933 seconds.

Average time for `parallel_cholesky_QR_decomposition`: 0.909875 seconds.

Gee, that's quick!!
