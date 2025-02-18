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

// CQR
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

// Parallel CQR
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

// CQR2
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

// sCQR
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

// Parallel sCQR
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

// sCQR3
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

// CQRGS
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
            int const end_block_index = n - start_block_index;

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

// dCQRGS
std::pair<Matrix, Matrix> distributed_cholesky_QR_w_gram_schmidt(Matrix &A)
{
    int const m = A.rows();
    int const n = A.cols();
    int const block_size = std::min(64, n);          // Adjust block size dynamically
    int const k = (n + block_size - 1) / block_size; // Number of panels
    int const num_threads = omp_get_max_threads();

    Matrix Q = Matrix::Zero(m, n);
    Matrix R = Matrix::Zero(n, n);

    for (int j = 0; j < k; ++j)
    {
        int const current_block_size = std::min(block_size, n - j * block_size);
        int const current_panel_col = j * block_size;

        // Compute global Gram matrix W_j = \sum_p (A_{p, j}^T A_{p, j})
        Matrix W_j = Matrix::Zero(current_block_size, current_block_size);
        std::vector<Matrix> local_W(num_threads, Matrix::Zero(current_block_size, current_block_size));

#pragma omp parallel
        {
            int const thread_id = omp_get_thread_num();
            int const chunk_size = m / num_threads;

            int const start = thread_id * chunk_size;
            int const end = (thread_id == num_threads - 1) ? m : start + chunk_size;

            // FIXME: Try to simplify computation by putting directly into W_j. Hard indexing.
            for (int row = start; row < end; ++row)
            {
                local_W[thread_id].noalias() += A.block(row, current_panel_col, 1, current_block_size).transpose() *
                                                A.block(row, current_panel_col, 1, current_block_size);
            }

#pragma omp critical
            W_j += local_W[thread_id];
        }

        // Cholesky factorization (thread redundant)
        Eigen::LLT<Matrix> cholesky_factorization(W_j);
        Matrix U = cholesky_factorization.matrixU();

// Compute Q_{p, j} = A_{p, j} * U^{-1}
#pragma omp parallel
        {
            int const thread_id = omp_get_thread_num();
            int const chunk_size = m / num_threads;

            int const start = thread_id * chunk_size;
            int const end = (thread_id == num_threads - 1) ? m : start + chunk_size;

            for (int row = start; row < end; ++row)
            {
                Q.block(row, current_panel_col, 1, current_block_size) = A.block(row, current_panel_col, 1, current_block_size) * U.inverse();
            }
        }

        R.block(current_panel_col, current_panel_col, current_block_size, current_block_size) = U;

        // Compute Y = \sum_p Q_{p, j}^T A_{p, j+1:k}
        if (j < k - 1)
        {
            int const next_panel_col = (j + 1) * block_size;
            int const trailing_cols = n - next_panel_col;

            Matrix Y = Matrix::Zero(current_block_size, trailing_cols);
            std::vector<Matrix> local_Y(num_threads, Matrix::Zero(current_block_size, trailing_cols));

#pragma omp parallel
            {
                int const thread_id = omp_get_thread_num();
                int const chunk_size = m / num_threads;

                int const start = thread_id * chunk_size;
                int const end = (thread_id == num_threads - 1) ? m : start + chunk_size;

                // FIXME: Try to directly insert into Y. Hard indexing.
                for (int row = start; row < end; ++row)
                {
                    local_Y[thread_id].noalias() += Q.block(row, current_panel_col, 1, current_block_size).transpose() *
                                                    A.block(row, next_panel_col, 1, trailing_cols);
                }

                //  Assign local_Y to the correct slice of Y
                Y.middleCols(0, trailing_cols) = local_Y[thread_id];
            }

// Update trailing panels
#pragma omp parallel
            {
                int const thread_id = omp_get_thread_num();
                int const chunk_size = m / num_threads;

                int const start = thread_id * chunk_size;
                int const end = (thread_id == num_threads - 1) ? m : start + chunk_size;

                for (int row = start; row < end; ++row)
                {
                    A.block(row, next_panel_col, 1, trailing_cols) -=
                        Q.block(row, current_panel_col, 1, current_block_size) * Y;
                }
            }

            R.block(current_panel_col, next_panel_col, current_block_size, trailing_cols) = Y;
        }
    }

    return {Q, R};
}