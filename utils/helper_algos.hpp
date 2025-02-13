#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>

// NOTE TO READER:
// This is a helper file to make implementation of algorithms that rely on
// simpler algorithms easier to implement.

typedef Eigen::MatrixXd Matrix;

// Compute unit roundoff for given floating-point type
template <typename T>
T compute_unit_roundoff()
{
    T u = 1.0;

    while (1.0 + u != 1.0)
    {
        u /= 2.0;
    }

    return u;
}

// Cholesky QR decomposition
std::pair<Matrix, Matrix> cholesky_QR(const Matrix &A)
{
    // Compute Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(A.transpose() * A);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}

// Parallel Cholesky QR decomposition
std::pair<Matrix, Matrix> parallel_cholesky_QR(const Matrix &A)
{
    int num_rows = A.rows();
    int num_cols = A.cols();
    int num_threads = omp_get_max_threads();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        Eigen::MatrixXd A_i = A.middleRows(start, end - start);

        // Compute A_i^T * A_i for the sliced submatrix
        local_W[thread_id].noalias() += A_i.transpose() * A_i;

        // Use a critical section to safely update the global Gram matrix W
#pragma omp critical
        {
            W += local_W[thread_id];
        }

        // Compute Q slice directly; I will multiply by R inverse later
        // This reduces storage by not storing A_i slices
        Q.middleRows(start, end - start) = A_i;
    }

    // Cholesky factorization of Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(W);
    Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}

// Cholesky QR decomposition 2
std::pair<Matrix, Matrix> cholesky_QR_2(const Matrix &A)
{
    // Initial Q and R extraction; this computation will be performed again
    // to increase numerical accuracy.
    auto [Q_1, R_1] = parallel_cholesky_QR(A);

    // Final Q and R extraction
    auto [Q, R_2] = parallel_cholesky_QR(Q_1);

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}

// Shifted Cholesky QR
std::pair<Matrix, Matrix> shifted_cholesky_QR(const Matrix &A)
{
    // Number of cols for shift application
    int const num_rows = A.rows();
    int const num_cols = A.cols();

    // Unit roundoff for double precision
    double const u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    double const norm_A = A.norm();

    // Stability shift
    double s = std::sqrt(num_rows) * u * norm_A;

    // Compute shifted Gram matrix
    Matrix G = A.transpose() * A;
    G.diagonal().array() += s; // Shift diagonal

    // Perform Cholesky factorization
    Eigen::LLT<Matrix> cholesky_factorization(G);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}

// Parallel Shifted Cholesky QR decomposition
std::pair<Matrix, Matrix> parallel_shifted_cholesky_QR(const Matrix &A)
{
    int num_rows = A.rows();
    int num_cols = A.cols();
    int num_threads = omp_get_max_threads();

    // Unit roundoff for double precision
    double const u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    double const norm_A = A.norm();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        Eigen::MatrixXd A_i = A.middleRows(start, end - start);

        // Compute A_i^T * A_i for the sliced submatrix
        local_W[thread_id].noalias() += A_i.transpose() * A_i;

        // Use a critical section to safely update the global Gram matrix W
#pragma omp critical
        {
            W += local_W[thread_id];
        }

        // Compute Q slice directly; I will multiply by R inverse later
        // This reduces storage by not storing A_i slices
        Q.middleRows(start, end - start) = A_i;
    }

    // Stability shift
    double s = std::sqrt(num_rows) * u * norm_A;

    // Apply shift to the diagonal of the Gram matrix
    W.diagonal().array() += s;

    // Cholesky factorization of Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(W);
    Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}

// Shifted Cholesky QR decomposition 3
std::pair<Matrix, Matrix> shifted_cholesky_QR_3(const Matrix &A)
{
    // Initial shifted extraction (shift for stability)
    auto [Q_1, R_1] = parallel_shifted_cholesky_QR(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_2(Q_1);

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}

// Cholesky QR decomposition with Gram-Schmidt
std::pair<Matrix, Matrix> cholesky_QR_w_gram_schmidt(Matrix &A) // A not const to allow in-place modification
{
    int const m = A.rows();
    int const n = A.cols();
    int const block_size = std::min(64, n); // Adjust block size dynamically

    Matrix Q(m, n);
    Matrix R(n, n);

    for (int j = 0; j < n; j += block_size)
    {
        // Adjusts block size for last iteration
        int const current_block_size = std::min(block_size, n - j);
        Matrix A_j = A.block(0, j, m, current_block_size);

        // Cholesky factorization on Gram matrix with no explicit inversion (W_j)
        Eigen::LLT<Matrix> cholesky_decomposition(A_j.transpose() * A_j);
        Matrix U = cholesky_decomposition.matrixU();

        // Compute orthogonal block Q_j
        Matrix Q_j = A_j * U.inverse();

        // Update Q and R
        Q.block(0, j, m, current_block_size) = Q_j;
        R.block(j, j, current_block_size, current_block_size) = U;

        // Update trailing panels
        if (j + current_block_size < n)
        {
            int const start_block_index = j + current_block_size;
            int const end_block_index = n - (j + current_block_size);

            Matrix A_next = A.block(0, start_block_index, m, end_block_index);
            Matrix Y = Q_j.transpose() * A_next;

            // A and R panel update
            A_next.noalias() -= Q_j * Y;
            A.block(0, start_block_index, m, end_block_index) = A_next;
            R.block(j, start_block_index, current_block_size, end_block_index) = Y;
        }
    }

    return {Q, R};
}

// CQR2GS
std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    // Initial extraction
    auto [Q_1, R_1] = cholesky_QR_w_gram_schmidt(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_w_gram_schmidt(Q_1);

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}