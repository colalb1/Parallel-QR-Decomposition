# Parallel-QR-Decomposition

Course-grained parallel thin-QR decomposition algorithms for tall-and-skinny matrices on a CPU in C++ using OpenMP.

## Precursor

### Why I did this project:

I wanted to learn more about parallelizing programs for high-performance numerical computations on CPUs. The [authors' Github](https://github.com/HybridScale/CholeskyQR2-IM) that these algorithms are originally based on contains [heterogeneous](https://www.intel.com/content/www/us/en/developer/articles/technical/efficient-heterogenous-parallel-programming-openmp.html) versions that are relatively difficult to understand without first seeing the pseudocode. Thus, this provided an opportunity to mesh my math and HPC interests to learn parallelization in practice, OpenMP, and do a QR decomposition math-refresh while providing some user-friendly(er) code.

The concepts and insights of this project are not novel, but I wanted to implement numerical algorithms from literature as a "warm-up" to a very interesting project that I begin working on very soon (and to show my C++ competence). [This is a hint](https://en.wikipedia.org/wiki/Asian_option) at said project's topic.

### What this project is:

C++ implementation of novel parallel QR decomposition algorithms from [this paper](https://arxiv.org/abs/2405.04237). I will implement the GPU-limited algorithms from [its repository](https://github.com/HybridScale/CholeskyQR2-IM) (**EDIT AFTER IMPLEMENTATION**: the GPU-limited algorithms were *VERY* slow as they rely on fine-grained parallelization).

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

$A$ matrix is "short-and-wide" when $m \ll n$. Short-and-wide problems are analogous to tall-and-skinny problems under a transpose operation.

### Machine Epsilon or Machine Precision

Upper bound on approximation error for floating point computations. **Unit roundoff** is a synonym. It is denoted by $\epsilon_{\text{machine}}$.

Given $b$-based arithmetic and $t$-digit precision, $\epsilon_{\text{machine}} = b^{1 - t}$. Common standards are IEEE 754 single-precision where $b = 2$ and $t = 24$ ($\epsilon_{\text{machine}} \approx 1.1920929\times10^{-7}$) and double-precision where $b = 2$ and $t = 52$ ($\epsilon_{\text{machine}} \approx 2.22044605\times10^{-16}$).

Informally, this is the smallest difference between one floating point number and another. 

### Condition Number

Measures the sensitivity of the solution to pertubations in the input. It is typically denoted by $\kappa$. Condition number with respect to a matrix is denoted by $\kappa(A)$.

Suppose $f: X \to Y$ is an exact function and $\hat{f}$ is an algorithm. $\delta f(x) = f(x) - \hat{f}(x)$ where $x\in X$.

Absolute condition number: $\lim_{\epsilon_{\text{machine}}\to 0^+} \left[\sup_{||\delta x||\leq \epsilon_{\text{machine}}} \Large\frac{||\delta f(x)||}{||\delta x||}\right]$.

Relative condition number: $\lim_{\epsilon_{\text{machine}}\to 0^+} \left[\sup_{||\delta x||\leq \epsilon_{\text{machine}}} \Large\frac{||\delta f(x)|| / ||f(x)||}{||\delta x|| / ||x||}\right]$.

Under the $L^2$ norm, $\kappa(A) = \Large\frac{\sigma_{\text{max}}(A)}{\sigma_{\text{min}}(A)}$ where $\sigma_{\text{max}}$ and $\sigma_{\text{min}}$ are the maximum and minimum [singular values](https://en.wikipedia.org/wiki/Singular_value) of $A$, respectively.

If a condition number is near $1$, it is called **well-conditioned**. If the condition number is very large, it is **ill-conditioned**. Examples of ill-conditioned matrices are [Hilbert](https://en.wikipedia.org/wiki/Hilbert_matrix) matrices, [Vandermonde](https://en.wikipedia.org/wiki/Vandermonde_matrix) matrices with dense nodes, and matrices with nearly-independent rows/columns.

### QR Decomposition

A matrix decomposition method known primarily for solving linear least squares via back-substitution ($A = QR$ and $Ax=b \implies QRx=b \implies Rx = Q^Tb$ because $Q$ is orthonormal).

$A\in\mathbb{R}^{m, n}\implies Q\in\mathbb{R}^{m, n}$ and $R\in\mathbb{R}^{n, n}$.

$Q$ is [orthonormal](https://en.wikipedia.org/wiki/Orthogonal_matrix) and $R$ is upper-triangular.

A QR decomposition is "thin" when $m > n$. A thin decomposition follows as such: 

$$A = QR = \begin{bmatrix} Q_1 & Q_2\end{bmatrix}\begin{bmatrix}R_1 , 0\end{bmatrix} = Q_1R_1$$

QR decomposition is preferred to the [normal equations](https://mathworld.wolfram.com/NormalEquation.html) when solving linear systems since the normal equations square the [condition number](https://en.wikipedia.org/wiki/Condition_number) and may lead to significant rounding errors when $A$ is [singular](https://www.geeksforgeeks.org/singular-matrix/). The condition number for the normal equations is $\kappa(A)^2$; it is $\mathcal{O}(\kappa(A)\epsilon)$ for QR decomposition.

There are several special properties of matrices in QR decomposition, along with algorithm variants that enhance the condition number and/or speed up computations. I implore you to [discover them for yourself](https://en.wikipedia.org/wiki/QR_decomposition); this documentation is a very basic crash course.

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

#### Fine-Grained Parallelization

Decomposes a large computation into trivial tasks at an individual (or small block) level.

For a matrix multiplication, task size $T$ for each processor may be $c_{i, j} = a_{i, k}b_{k, j}$ (VERY small).

Communication cost $C_{\text{fine}}$ grows at rate $\mathcal{O}(Pc)$ where $P$ is the number of processors and $c$ is the cost per communication. This communication rate is relatively high, meaning it is advantageous to have fast communication in the form of [shared memory architecture](https://en.wikipedia.org/wiki/Shared_memory). 

Read more about how NVIDIA takes advantage of shared memory [here](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/).

The work-to-communication ratio is very small, implying each processor performs few computations.

#### Course-Grained Parallelization

Decomposes a large computation into medium to large-sized tasks.

Following the matrix multiplication example, $T$ may be of magnitude $\vec{a}_i\vec{b_j}$ or $A_{I, K}B_{K, J}$ where $A_{I, K}$ and $B_{K, J}$ are matrix slices.

Communication cost $C_{\text{coarse}}$ grows at an order of $\mathcal{O}(\sqrt{P}c)$, much lower than that of fine-grained parallelization.

Work-to-communication ratio is relatively large, implying each processor performs MANY computations.

Coarse-grained parallelization is better suited for problems limited by synchronization and communication latency such as in distributed databases where data is partitioned across nodes or in graph algorithms whose workload per edge/vertex [varies greatly](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm).

### Weak and Strong Scaling Speedup

**Strong scaling** measures how execution time decreases as the number of processors $P$ increases. **Strong speedup** is the ratio of time taken to complete a task with one processor $t(1)$ and the same task with $P$ processors, denoted by $t(P)$.

$$\text{\bf{speedup}}_{\text{strong}} = \frac{t(1)}{t(P)}$$

See [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) for more details. Hiring more people to paint a fence speeds it up, but adding too many does not since the work cannot be divided infinitely.

**Weak scaling** measures how execution time changes as $P$ increases while task size per processor remains constant. The **weak speedup** is...

$$\text{\bf{speedup}}_{\text{weak}} = s + p * P$$

where $s$ is the proportion of execution time spent computing serial tasks and $p$ is that of the parallel tasks. For simplicity, assume $s + p \leq 1$ and $s + p \approx 1$. I could not find the "approximately less than" TeX symbol.

Informally, weak scaling and [Gustafson's law](https://en.wikipedia.org/wiki/Gustafson%27s_law) explain that increasing the problem size and the number of processors results in near-linear speedups. Instead of painting a fence faster, paint a longer fence in the same amount of time by hiring more people.

## Algorithms

See [this link](https://arxiv.org/html/2405.04237v1) for the full pseudocode; it is not ALL rewritten here for brevity. 

Suppose $p=$ processor count.

### Cholesky QR (CQR)

**Algorithm: CholeskyQR**

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
   - $\text{threads} \gets \text{max-threads()} $  
   - $W$ as a zero matrix of size $n \times n$  
   - $\text{local W}$ as an array of zero matrices $n \times n$, one for each thread  

2. **Compute Gram Matrix in Parallel**  
   - **Parallel for each** $\text{thread id} \in [0, ..., \text{threads} - 1]$:  
     1. $\text{chunk size} \gets \big\lfloor\frac{\text{rows}}{\text{threads}}\big\rfloor$  
     2. $\text{start} \gets \text{thread id} \times \text{chunk size}$
     3. $\text{end} \gets \begin{cases} \text{rows}, & \text{if thread id} = \text{threads} - 1 \\ \text{start} +\text{chunk size}, & \text{otherwise} \end{cases}$ 
     4. $A_i \gets A[\text{start}:\text{end}]$ 
     5. $\text{local W}[\text{thread id}] \gets A_i^T A_i$  
     6. **Critical Section:** $W \gets W + \text{local W}[\text{thread id}]$  
     7. $Q[\text{start}:\text{end}] \gets A_i$  

3. **Perform Cholesky Factorization**  
   - $W = R^T R$

4. **Compute $Q$ in Parallel**  
   - **Parallel for each** $\text{thread id} \in [0, ..., \text{threads} - 1]$:  
     1. $Q[\text{start}:\text{end}] \gets Q[\text{start}:\text{end}] \times R^{-1}$  

5. **Return** $(Q, R)$

The Gram Matrix is computed in parallel by slicing $A$ into chunks, reducing complexity to $\mathcal{O}\big(\frac{mn^2}{p}\big)$ from $\mathcal{O}(mn^2)$. Synchronization overhead from updating $W$ is negligible. Computing the  final $Q$ in parallel scales similarly.

### Cholesky QR 2 (CQR2)

**Algorithm: CQR2**  

**Input:** Matrix $A \in \mathbb{R}^{m \times n}$.

**Output:** Matrices $Q \in \mathbb{R}^{m \times n}$ and $R \in \mathbb{R}^{n \times n}$. 

1. **First QR Decomposition**  
   - $[Q_1, R_1] \gets \text{ParallelCholeskyQR}(A)$

2. **Second QR Decomposition for Accuracy**  
   - $[Q, R_2] \gets \text{ParallelCholeskyQR}(Q_1)$

3. **Compute Final $R$**  
   - $R \gets R_2 R_1$

4. **Return** $(Q, R)$

**CQR** can produce non-orthogonal vectors, becoming unstable as the condition number increases. Repeating orthogonalization improves stability, as detailed [here](https://link.springer.com/article/10.1007/s00211-005-0615-4). Orthogonality error scales as such: $\mathcal{O}(\kappa(A)^2\bold{u})$.

### Shifted Cholesky QR (sCQR)

From here I will ONLY be giving a brief explanation of each algorithm's improvements. See [the paper](https://arxiv.org/html/2405.04237v1) for pseudocode; it is redundant to write here.

A shift $s = \sqrt{m}\bold{u}||A||_F^2$ is applied to the diagonal of the Gram matrix to force it to be positive definite. The rest of the steps follow **CQR**.

### Shifted Cholesky QR 3 (sCQR3)

This is essentially **CQR2** but instead of applying **CQR** twice, it applies **sCQR** as a preconditioner to **CQR2**, which achieves further orthogonalization.

### Cholesky QR with Gram-Schmidt (CQRGS)

Similar to **CQR** but with block processing and panel update/reorthogonalization before computing the final $R$.

### Cholesky QR2 with Gram-Schmidt (CQR2GS)

**CQR2** with **CQRGS** instead of **CQR**, leveraging parallel block processing and Gram-Schmidt reorthogonalization. It improves stability, efficiency, and accuracy while optimizing computational cost.

### Modified Cholesky QR2 with Gram-Schmidt (mCQR2GS)

**mCQR2GS** restructures **CQRGS** to reduce the number of panels while maintaining computational and communication efficiency. It adaptively selects the paneling strategy based on matrix conditioning, ensuring stability with fewer operations. Compared to **CQR2GS**, **mCQR2GS** requires fewer floating-point operations by avoiding explicit factor construction and achieves better orthogonality with fewer panels for high-condition-number matrices.

## Conclusion

This project served as a hands-on exploration of parallel QR decomposition using OpenMP, blending high-performance computing with numerical linear algebra. 

While the concepts are not novel, implementing them deepened my understanding of parallelization and provided a practical refresh on QR decomposition. 
The algorithms showcased here accelerate least-squares regression and eigenvalue computations, making large-scale data analysis more efficient.

## Side Note

I used the [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) flame graph for performance analysis and stack trace inspection.

I burnt out while writing the documentation and realized I want to spend more time writing code instead of words that people (likely) won't read. For this reason, I will start contributing to some kind of relevant open-source project related to numerical/parallel computing moving forward (after the OTHER project ;)), because writing documentation is time-intensive and (usually) not thrilling. Do NOT mistake this for disinterest in the mathematical concepts or in communicating my ideas. Building "the thing" is more fun than explaining "the thing," and these projects are extracurricular for me, so I'm going to lean toward the "fun" aspect. 

I also want to spend more time watching "The Walking Dead" after work. I watched through [season 7](https://en.wikipedia.org/wiki/The_Walking_Dead_season_7) a few years ago but never saw the whole show. I also want to watch the spinoffs. Realistically, I'll start another project while watching it because I want to learn more things and advance my career, but I thought I would let you know where I'm at. 

A colleague (and close friend) (of whose advice I take more often than many others) exclaimed that I'm unsatisfied with some of my work because it is a compulsion to work on it for me, and that I should relax more to "produce excellence when it counts." I don't really consider my work "excellent" (although it is what I strive for), but I will be taking his advice and watching more "Walking Dead" in the near future.
