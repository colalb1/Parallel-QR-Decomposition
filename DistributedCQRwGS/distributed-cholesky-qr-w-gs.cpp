#include <utils/helper_algos.hpp>

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

// int main()
// {
//     const double EPSILON = 1e-6;
//     // Define a tall and skinny matrix (m > n)
//     Matrix A(8, 3);
//     A << 3.0, 2.0, 1.0,
//         2.0, 3.0, 2.0,
//         1.0, 2.0, 3.0,
//         4.0, 1.0, 2.0,
//         3.0, 4.0, 1.0,
//         5.0, 3.0, 2.0,
//         2.0, 5.0, 3.0,
//         1.0, 2.0, 4.0;

//     // Compute Cholesky QR decomposition
//     auto [Q, R] = distributed_cholesky_QR_w_gram_schmidt(A);

//     // Verify that Q is orthonormal: Q^T * Q ≈ I
//     Matrix I = Matrix::Identity(Q.cols(), Q.cols());
//     Matrix QtQ = Q.transpose() * Q;
//     bool isOrthonormal = QtQ.isApprox(I, EPSILON);

//     // Verify that R is upper triangular
//     bool isUpperTriangular = R.isUpperTriangular(EPSILON);

//     // Verify that A ≈ Q * R
//     Matrix reconstructedA = Q * R;
//     bool isDecompositionCorrect = A.isApprox(reconstructedA, EPSILON);

//     // Print results
//     if (isOrthonormal && isUpperTriangular && isDecompositionCorrect)
//     {
//         std::cout << "Test Passed: Cholesky QR decomposition is correct." << std::endl;
//     }
//     else
//     {
//         std::cout << "Test Failed: Issues detected in decomposition." << std::endl;
//         if (!isOrthonormal)
//         {
//             std::cout << "Q is not orthonormal (Q^T * Q ≠ I).\n";
//         }
//         if (!isUpperTriangular)
//         {
//             std::cout << "R is not upper triangular.\n";
//         }
//         if (!isDecompositionCorrect)
//         {
//             std::cout << "A ≠ Q * R.\n";
//         }
//     }

//     return 0;
// }