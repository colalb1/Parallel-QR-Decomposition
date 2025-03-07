#include <utils/helper_algos.hpp>

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