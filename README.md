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

Skip if you have an undergraduate-level understanding of numerical computing and parallelism.

### Tall-and-Skinny Matrices

$A\in\mathbb{R}^{m, n}$ is tall and skinny when $m \gg n$. 

The most common occurrence of these is design matrices for machine learning where the number of data points is much larger than the number of features. Other common applications of tall-and-skinny matrices are Fourier transforms in sensors, finite element methods, and the Jacobian matrices for iterative optimization algorithms.

$A$ matrix is "short-and-wide" when $m \ll n$. Short-and-wide problems are analogous to tall-and-skinny problems under the transpose operation.

### Machine Epsilon or Machine Precision

Upper bound on approximation error for floating point computations. **Unit roundoff** is a synonym. It is denoted by $\epsilon_{\text{machine}}$.

Given $b$-based arithmetic and $t$-digit precision, $\epsilon_{\text{machine}} = b^{1 - t}$. Common standards are IEEE 754 single-precision where $b = 2$ and $t = 24$ ($\epsilon_{\text{machine}} \approx 1.1920929\times10^{-7}$) and double-precision where $b = 2$ and $t = 52$ ($\epsilon_{\text{machine}} \approx 2.22044605\times10^{-16}$).

Informally, this is the smallest difference between one floating point number and another. 

### Condition Number

Measures the sensitivity of the solution to pertubations in the input. It is typically denoted by $\kappa$. Condition number with respect to a matrix is denoted by $\kappa(A)$.

Suppose $f: X \to Y$ is an exact function and $\hat{f}$ is an algorithm. $\delta f(x) = f(x) - \hat{f}(x)$ where $x\in X$.

Absolute condition number: $\lim_{\epsilon_{\text{machine}}\to 0^+} \left[\sup_{||\delta x||\leq \epsilon_{\text{machine}}} \Large\frac{||\delta f(x)||}{||\delta x||}\right]$.

Relative condition number: $\lim_{\epsilon_{\text{machine}}\to 0^+} \left[\sup_{||\delta x||\leq \epsilon_{\text{machine}}} \Large\frac{||\delta f(x)|| / ||f(x)||}{||\delta x|| / ||x||}\right]$.

When using the $L^2$ norm, $\kappa(A) = \Large\frac{\sigma_{\text{max}}(A)}{\sigma_{\text{min}}(A)}$ where $\sigma_{\text{max}}$ and $\sigma_{\text{min}}$ are the maximum and minimum [singular values](https://en.wikipedia.org/wiki/Singular_value) of $A$, respectively.

If a condition number is near $1$, it is called **well-conditioned**. If the condition number is very large, it is **ill-conditioned**. Examples of ill-conditioned matrices are [Hilbert](https://en.wikipedia.org/wiki/Hilbert_matrix) matrices, [Vandermonde](https://en.wikipedia.org/wiki/Vandermonde_matrix) matrices with dense nodes, and matrices with nearly-independent rows/columns.

### QR Decomposition

A matrix decomposition method known primarily for solving linear least squares via back-substitution ($A = QR$ and $Ax=b \implies QRx=b \implies Rx = Q^Tb$ because $Q$ is orthonormal).

$A\in\mathbb{R}^{m, n}\implies Q\in\mathbb{R}^{m, n}$ and $R\in\mathbb{R}^{n, n}$.

$Q$ is [orthonormal](https://en.wikipedia.org/wiki/Orthogonal_matrix) and $R$ is upper-triangular.

A QR decomposition is "thin" when $m > n$. A thin decomposition follows as such: $A = QR = \begin{bmatrix} Q_1 & Q_2\end{bmatrix}\begin{bmatrix}R_1 \\ 0\end{bmatrix} = Q_1R_1$.

QR decomposition is preferred to the [normal equations](https://mathworld.wolfram.com/NormalEquation.html) for solving linear systems since the normal equations square the [condition number](https://en.wikipedia.org/wiki/Condition_number) and may lead to significant rounding errors when $A$ is [singular](https://www.geeksforgeeks.org/singular-matrix/). The condition number of using the normal equations is $\kappa(A)^2$ while its QR decomposition counterpart's is $\mathcal{O}(\kappa(A)\epsilon)$.

There are various other special properties regarding the matrices in QR decomposition and variants of the algorithm that improve the condition number + computation speed. I implore you to [discover them for yourself](https://en.wikipedia.org/wiki/QR_decomposition); this documentation is a very basic crash course.

### Gram-Schmidt Orthogonalization

The Gram-Schmidt algorithm finds an orthonormal basis for a set of linearly independent vectors.

**Algorithm: Gram-Schmidt Orthogonalization**

**Input:** Linearly independent vectors $\{v_0, \dots, v_n\}$ in an [inner product space](https://en.wikipedia.org/wiki/Inner_product_space).

**Output:** Orthonormal basis $\{e_0, \dots, e_n\}$.

1. **Initialize**:
   - Set $u_0 = v_0$
   - Normalize: $e_0 = \frac{u_0}{\|u_0\|}$

2. **For** $i = 1\dots n + 1$:
   - Set $u_i = v_i$
   - **For** $j = 0\dots i$:
     - Compute projection: $\text{proj}_{u_j}(v_i) = \frac{\langle v_i, u_j \rangle}{\langle u_j, u_j \rangle} u_j$
     - Subtract projection: $u_i = u_i - \text{proj}_{u_j}(v_j)$
   - Normalize: $e_i = \frac{u_i}{\| u_i \|}$

3. **Return** $\{e_0, \dots, e_n\}$

This orthogonalization method is relevant because $Q = \begin{bmatrix}e_0 \cdots e_n\end{bmatrix}$ (the orthonormal vectors) and $R_{i, j} = \langle v_j, e_i\rangle, \text{  } 0 \leq i \leq j \leq n$ (the projection coefficients). One may deduce that $R_{i, i} = \|u_i\|$.

### Fine vs Course-Grained Parallelization

This will remain relatively high-level for brevity.

#### Fine-Grained Parallelization

Decomposes a large computation into trivial tasks at an individual (or small block) level.

If I do a matrix multiplication, task size $T$ for each processor may be $c_{i, j} = a_{i, k}b_{k, j}$ (relatively VERY small).

Communication cost $C_{\text{fine}}$ grows at rate $\mathcal{O}(Pc)$ where $P$ is the number of processors and $c$ is the cost per communication. This communication rate is relatively high, meaning it is advantageous to have fast communication in the form of [shared memory architecture](https://en.wikipedia.org/wiki/Shared_memory). 

Read more about how NVIDIA is taking advantage of shared memory [here](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/).

The work-to-communication ratio is very small, implying each processor performs few computations.

#### Course-Grained Parallelization

Decomposes a large computation into medium to large-sized tasks.

Following the matrix multiplication example, $|T|$ may be $\vec{a}_i\vec{b_j}$ or $A_{I, K}B_{K, J}$ where $A_{I, K}$ and $B_{K, J}$ are matrix slices.

Communication cost $C_{\text{coarse}}$ is $\mathcal{O}(\sqrt{P}c)$, much lower than that of fine-grained parallelization.

Work-to-communication ratio is relatively large, implying each processor performs MANY computations.

Coarse-grained parallelization is better suited for problems limited by synchronization and communication latency such as in distributed databases where data is partitioned across nodes or in graph algorithms whose workload per edge/vertex [varies greatly](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm).

### Weak and Strong Scaling Speedup

**Strong scaling** measures how execution time decreases as the number of processors $P$ increases. The **strong speedup** is the ratio of time taken to complete a task with one processor $t(1)$ and the same task with $P$ processors, denoted by $t(P)$.

$$\text{\bf{speedup}}_{\text{strong}} = \frac{t(1)}{t(P)}$$

See [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) for more details. Hiring more people to paint a fence speeds it up, but adding too many does not since the work cannot be divided infinitely.

**Weak scaling** measures how execution time changes as $P$ increases while task-size per processor is constant. The **weak speedup** is...

$$\text{\bf{speedup}}_{\text{weak}} = s + p * P$$

where $s$ is the proportion of execution time spent computing serial tasks and $p$ is that of the parallel tasks.

Informally, weak scaling and [Gustafson's law](https://en.wikipedia.org/wiki/Gustafson%27s_law) explain that increasing the problem size and the number of processors results in near-linear speedups. Instead of painting a fence faster, paint a longer fence in the same amount of time by hiring more people.

## Algorithms

See [this link](https://arxiv.org/html/2405.04237v1) for the full pseudocode; it is not rewritten here for brevity. 

Suppose $p=$ number of processors.

### Cholesky QR (CQR)

**Algorithm: Cholesky QR**

**Input:** Matrix $A \in \mathbb{R}^{m \times n}$.

**Output:** Matrices $Q \in \mathbb{R}^{m \times n}$ and $R \in \mathbb{R}^{n \times n}$.

1. **Construct Gram Matrix**:
   - $W = A^T A$

2. **Cholesky Factorization**:
   - $W = R^T R$

3. **Compute \( Q \)**:
   - $Q = AR^{-1}$

4. **Return** $(Q, R)$

### Parallel Cholesky QR

**Algorithm: ParallelCholeskyQR**

**Input:** Matrix $A \in \mathbb{R}^{m \times n}$.

**Output:** Matrices $Q \in \mathbb{R}^{m \times n}$ and $R \in \mathbb{R}^{n \times n}$.

1. **Initialize Variables**  
   - $\text{rows} \gets \text{rows}(A)$  
   - $\text{cols} \gets \text{cols}(A)$  
   - $\text{threads} \gets \text{max\_threads()} $  
   - $W$ as a zero matrix of size $n \times n$  
   - $\text{local\_W}$ as an array of zero matrices $n \times n$, one for each thread  

2. **Compute Gram Matrix in Parallel**  
   - **Parallel for each** $\text{thread\_id} \in \{0, ..., \text{threads} - 1\}$:  
     1. $ \text{chunk\_size} \gets \big\lfloor\frac{\text{rows}}{\text{threads}}\big\rfloor $  
     2. $ \text{start} \gets \text{thread\_id} \times \text{chunk\_size} $
     3. $ \text{end} \gets  
        \begin{cases}  
        \text{rows}, & \text{if thread\_id} = \text{threads} - 1 \\  
        \text{start} + \text{chunk\_size}, & \text{otherwise}  
        \end{cases} $ 
     4. $ A_i \gets A[\text{start}:\text{end}] $ 
     5. $\text{local\_W}[\text{thread\_id}] \gets A_i^T A_i$  
     6. **Critical Section:** $W \gets W + \text{local\_W}[\text{thread\_id}]$  
     7. $Q[\text{start}:\text{end}] \gets A_i$  

3. **Perform Cholesky Factorization**  
   - $W = R^T R$

4. **Compute $Q$ in Parallel**  
   - **Parallel for each** $\text{thread\_id} \in \{0, ..., \text{threads} - 1\}$:  
     1. $Q[\text{start}:\text{end}] \gets Q[\text{start}:\text{end}] \times R^{-1}$  

5. **Return** $(Q, R)$

The Gram Matrix is computed in parallel by slicing $A$ into chunks, reducing complexity to $\mathcal{O}\big(\frac{mn^2}{p}\big)$ from $\mathcal{O}(mn^2)$. Synchronization overhead from updating $W$ is negligible. Computing the  final $Q$ in parallel scales similarly.

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
