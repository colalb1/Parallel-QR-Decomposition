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
T compute_unit_roundoff() {
    T u = 1.0;

    while (1.0 + u != 1.0) {
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

        // Compute Q slice directly; I will multiply by R inverse later
        // This reduces storage by not storing A_i slices
        Q.middleRows(start, end - start) = A_i;
    }

    // Sum local Gram matrices
    // #pragma omp parallel for reduction(+ : W)
    for (int i = 0; i < num_threads; ++i)
    {
        W += local_W[i];
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