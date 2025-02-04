# Parallel-QR-Decomposition

*Note to the recruiter/senior engineer who is considering hiring me and is reading this:* I write more buttoned-up (dry) documentation in a professional setting. This is **MY** project so the information will be conveyed in a tone true to myself. This certainly does **NOT** imply I will sacrifice any technical quality; there is no point in doing anything if you don't *try* to make it the best it can be. If the persona quirks of this document are a dealbreaker for you, so be it. *I* think this work is interesting, so I encourage you to read it (the technical parts, anyway) regardless.

## Why I am doing this project:

I want to learn more about parallelizing programs for high-performance numerical computations on CPUs.

## What this project is:

Writing parallel QR decomposition algorithms. Part of this will be implementing the GPU-limited algorithms from [this repo](https://github.com/HybridScale/CholeskyQR2-IM) and continuing to other algorithms (like TLQR or Householder (want stable parallel option)) after I get my bearings.
I will probably rewrite the algorithm they have written for a CPU for sanity purposes (reading other people's code is a hellish experience).

Start by reading [this paper](https://arxiv.org/abs/2405.04237) for background. Now you may continue reading.

## Why you should care:

Realistically, you probably shouldn't. 

No, but for real, algorithms like this significantly speed up least-squares problems and eigenvalue computations for [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), among other relevant applications. Basically, data scientists will waste less time ~~scrolling on TikTok~~ waiting for models to finish computing, and can iterate/improve solutions faster.

What this means for business people who don't care about all that academic stuff is that your engineers can iterate on solutions faster and hence make you more money.

## TODO:

1. Figure out how to run C++ programs on PC without breaking
2. Implement simple parallel applications (for loops, other basics)
3. Implement iterative Cholesky (used for speed comparison)
4. Implement parallel Cholesky (used for speed comparison)
5. Implement CholeskyQR2
6. Implement CholeskyQR2 with Gram-Schmidt (CQR2GS)
7. Implement sCQR3
8. Implement TSQR
9. Implement Householder (used for speed comparison)
10. Implement parallel Householder
