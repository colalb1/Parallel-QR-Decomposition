#include <utils/helper_algos.hpp>

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
