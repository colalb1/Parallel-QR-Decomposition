#include <utils/helper_algos.hpp>

std::pair<Matrix, Matrix> distributed_cholesky_QR_w_gram_schmidt(Matrix &A, int block_size)
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
        int current_block_size = std::min(block_size, n - j * block_size);
        int current_panel_col = j * block_size;

        // Compute global Gram matrix W_j = \sum_p (A_{p, j}^T A_{p, j})
        Matrix W_j = Matrix::Zero(current_block_size, current_block_size);
        std::vector<Matrix> local_W(num_threads, Matrix::Zero(current_block_size, current_block_size));

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int chunk_size = m / num_threads;

            int start = thread_id * chunk_size;
            int end = (thread_id == num_threads - 1) ? m : start + chunk_size;

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
            int thread_id = omp_get_thread_num();
            int chunk_size = m / num_threads;

            int start = thread_id * chunk_size;
            int end = (thread_id == num_threads - 1) ? m : start + chunk_size;

            for (int row = start; row < end; ++row)
            {
                Q.block(row, current_panel_col, 1, current_block_size) = A.block(row, current_panel_col, 1, current_block_size) * U.inverse();
            }
        }

        // Compute Y = \sum_p Q_{p, j}^T A_{p, j+1:k}
        if (j < k - 1)
        {
            int next_panel_col = (j + 1) * block_size;
            int trailing_cols = n - next_panel_col;
            Matrix Y = Matrix::Zero(current_block_size, trailing_cols);
            std::vector<Matrix> local_Y(num_threads, Matrix::Zero(current_block_size, trailing_cols));

#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int chunk_size = m / num_threads;
                int start = thread_id * chunk_size;
                int end = (thread_id == num_threads - 1) ? m : start + chunk_size;

                for (int row = start; row < end; ++row)
                {
                    local_Y[thread_id].noalias() += Q.block(row, current_panel_col, 1, current_block_size).transpose() *
                                                    A.block(row, next_panel_col, 1, trailing_cols);
                }

#pragma omp critical
                Y += local_Y[thread_id];
            }

// Update trailing panels
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int chunk_size = m / num_threads;
                int start = thread_id * chunk_size;
                int end = (thread_id == num_threads - 1) ? m : start + chunk_size;

                for (int row = start; row < end; ++row)
                {
                    A.block(row, next_panel_col, 1, trailing_cols) -=
                        Q.block(row, current_panel_col, 1, current_block_size) * Y;
                }
            }

            R.block(current_panel_col, next_panel_col, current_block_size, trailing_cols) = Y;
        }

        // Store U in R
        R.block(current_panel_col, current_panel_col, current_block_size, current_block_size) = U;
    }

    return {Q, R};
}