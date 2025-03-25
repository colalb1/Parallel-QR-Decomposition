#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>
#include <fstream>

// NOTE TO READER:
// This is a helper file to make implementation of algorithms that rely on
// simpler algorithms easier to implement.

using Matrix = Eigen::MatrixXd;

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
constexpr std::pair<Matrix, Matrix> cholesky_QR(Matrix &A)
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
constexpr std::pair<Matrix, Matrix> parallel_cholesky_QR(Matrix &A)
{
    const int num_rows = A.rows();
    const int num_cols = A.cols();
    const int num_threads = omp_get_max_threads();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        const Matrix A_i = A.middleRows(start, end - start);

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
    Matrix cholesky_factorization(W);
    const Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    const Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}

// CQR2
constexpr std::pair<Matrix, Matrix> cholesky_QR_2(Matrix &A)
{
    // Initial Q and R extraction; this computation will be performed again
    // to increase numerical accuracy.
    auto [Q_1, R_1] = parallel_cholesky_QR(A);

    // Final Q and R extraction
    auto [Q, R_2] = parallel_cholesky_QR(Q_1);

    // Calculate final R
    const Matrix R = R_2 * R_1;

    return {Q, R};
}

// sCQR
constexpr std::pair<Matrix, Matrix> shifted_cholesky_QR(Matrix &A)
{
    // Number of cols for shift application
    const int num_rows = A.rows();
    const int num_cols = A.cols();

    // Unit roundoff for double precision
    const double u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    const double norm_A = A.norm();

    // Stability shift
    // From this link: https://arxiv.org/abs/1809.11085
    const double s = 11 * num_cols * (num_rows + num_cols + 1) * std::sqrt(num_rows) * u * std::pow(norm_A, 2);

    // Compute shifted Gram matrix
    Matrix G = A.transpose() * A;
    G.diagonal().array() += s; // Shift diagonal

    // Perform Cholesky factorization
    Matrix cholesky_factorization(G);

    // Get upper triangular Cholesky factor R
    const Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    const Matrix Q = A * R.inverse();

    return {Q, R};
}

// Parallel sCQR
constexpr std::pair<Matrix, Matrix> parallel_shifted_cholesky_QR(Matrix &A)
{
    const int num_rows = A.rows();
    const int num_cols = A.cols();
    const int num_threads = omp_get_max_threads();

    // Unit roundoff for double precision
    const double u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    const double norm_A = A.norm();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        Matrix A_i = A.middleRows(start, end - start);

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
    const double s = 11 * num_cols * (num_rows + num_cols + 1) * std::sqrt(num_rows) * u * std::pow(norm_A, 2);

    // Apply shift to the diagonal of the Gram matrix
    W.diagonal().array() += s;

    // Cholesky factorization of Gram matrix
    Matrix cholesky_factorization(W);
    const Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}

// sCQR3
constexpr std::pair<Matrix, Matrix> shifted_cholesky_QR_3(Matrix &A)
{
    // Initial shifted extraction (shift for stability)
    auto [Q_1, R_1] = parallel_shifted_cholesky_QR(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_2(Q_1);

    // Calculate final R
    const Matrix R = R_2 * R_1;

    return {Q, R};
}

// CQRGS
constexpr std::pair<Matrix, Matrix> cholesky_QR_w_gram_schmidt(Matrix &A) // A not const to allow in-place modification
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
        Matrix gram_matrix = A_j.transpose() * A_j;

        // Cholesky factorization on Gram matrix with no explicit inversion (W_j)
        Eigen::LLT<Matrix> cholesky_decomposition(gram_matrix);
        Matrix U = cholesky_decomposition.matrixU();

        // Compute orthogonal block Q_j
        Matrix Q_j = A_j * U.inverse();

        // Update Q and R
        Q.block(0, j, m, current_block_size) = Q_j;
        R.block(j, j, current_block_size, current_block_size) = U;

        int const start_block_index = j + current_block_size;

        // Update trailing panels
        if (start_block_index < n)
        {
            int const end_block_size = n - start_block_index;

            Matrix A_next = A.block(0, start_block_index, m, end_block_size);
            Matrix Y = Q_j.transpose() * A_next;

            // A and R panel update
            A_next.noalias() -= Q_j * Y;
            A.block(0, start_block_index, m, end_block_size) = A_next;
            R.block(j, start_block_index, current_block_size, end_block_size) = Y;
        }
    }

    return {Q, R};
}

// CQR2GS
constexpr std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    // Initial extraction
    auto [Q_1, R_1] = cholesky_QR_w_gram_schmidt(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_w_gram_schmidt(Q_1);

    // Set threads
    omp_set_num_threads(omp_get_max_threads());

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}

// dCQRGS
constexpr std::pair<Matrix, Matrix> distributed_cholesky_QR_w_gram_schmidt(Matrix &A)
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
        int const current_panel_col = j * block_size;
        int const current_block_size = std::min(block_size, n - current_panel_col);

        // Compute global Gram matrix W_j = \sum_p (A_{p, j}^T A_{p, j})
        Matrix W_j = Matrix::Zero(current_block_size, current_block_size);
        std::vector<Matrix> local_W(num_threads, Matrix::Zero(current_block_size, current_block_size));

#pragma omp parallel
        {
            int const thread_id = omp_get_thread_num();
            int const chunk_size = m / num_threads;

            int const start = thread_id * chunk_size;
            int const end = (thread_id == num_threads - 1) ? m : start + chunk_size;

            // Inserting directly into W_j would be more space efficient.
            // This could cause a race condition or inconsistent state of W_j.
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

                // Again, would be more space efficient to insert directly into Y.
                // This could lead to race conditions or inconsistent state of Y.
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

// mCQR2GS
constexpr std::pair<Matrix, Matrix> modified_cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    int const m = A.rows();
    int const n = A.cols();
    int const block_size = std::min(64, n);          // Dynamic block size selection
    int const k = (n + block_size - 1) / block_size; // Number of panels
    int const num_threads = omp_get_max_threads();

    Matrix Q = Matrix::Zero(m, n);
    Matrix R = Matrix::Zero(n, n);

    // Orthogonalize first panel
    Matrix A_block = A.block(0, 0, m, block_size);
    auto [Q_1, R_11] = cholesky_QR_2(A_block);

    Q.block(0, 0, m, block_size) = Q_1;
    R.block(0, 0, block_size, block_size) = R_11;

    for (int j = 1; j < k; ++j)
    {
        int const previous_panel_col = (j - 1) * block_size;
        int const previous_block_size = std::min(block_size, n - previous_panel_col);

        int const current_panel_col = j * block_size;
        int const trailing_cols = n - current_panel_col;
        int const current_block_size = std::min(block_size, trailing_cols);

        Matrix Q_jm1 = Q.block(0, previous_panel_col, m, previous_block_size);

        // Projections of orthogonal panels
        Matrix Y = Matrix::Zero(previous_block_size, trailing_cols);

#pragma omp parallel for num_threads(num_threads)
        for (int col = 0; col < trailing_cols; ++col)
        {
            Y.col(col) = Q_jm1.transpose() * A.block(0, current_panel_col + col, m, 1);
        }

// Update trailing panels
#pragma omp parallel for num_threads(num_threads)
        for (int col = 0; col < trailing_cols; ++col)
        {
            A.block(0, current_panel_col + col, m, 1) -= Q_jm1 * Y.col(col);
        }
        R.block(previous_panel_col, current_panel_col, previous_block_size, trailing_cols) = Y;

        // Process current panel j (after trailing update)
        Matrix A_j = A.block(0, current_panel_col, m, current_block_size);
        auto [Q_j, R_jj] = parallel_cholesky_QR(A_j);

        // Reorthogonalize current panel with respect to Q_prev
        Matrix Q_previous = Q.block(0, 0, m, current_panel_col); // To current_panel_col because the 1:(j - 1) panels implies stopping at the index of the j^th panel
        Matrix proj = Matrix::Zero(current_panel_col, current_block_size);

#pragma omp parallel for num_threads(num_threads)
        for (int col = 0; col < current_block_size; ++col)
        {
            proj.col(col) = Q_previous.transpose() * A_j.col(col);
            A_j.col(col) -= Q_previous * proj.col(col);
        }
        A.block(0, current_panel_col, m, current_block_size) = A_j;

        // Fully orthogonalize current panel
        Q.block(0, current_panel_col, m, current_block_size) = Q_j;
        R.block(current_panel_col, current_panel_col, current_block_size, current_block_size) = R_jj;
    }

    return {Q, R};
}