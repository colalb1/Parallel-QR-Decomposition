#include "utils/helper_algos.hpp"

// mCQR2GS
constexpr std::pair<Matrix, Matrix> modified_cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    const int m = A.rows();
    const int n = A.cols();
    const int block_size = std::min(64, n);          // Dynamic block size selection
    const int k = (n + block_size - 1) / block_size; // Number of panels
    const int num_threads = omp_get_max_threads();

    Matrix Q = Matrix::Zero(m, n);
    Matrix R = Matrix::Zero(n, n);

    // Orthogonalize first panel
    Matrix A_block = A.block(0, 0, m, block_size);
    auto [Q_1, R_11] = cholesky_QR_2(A_block);

    Q.block(0, 0, m, block_size) = Q_1;
    R.block(0, 0, block_size, block_size) = R_11;

    for (int j = 1; j < k; ++j)
    {
        const int previous_panel_col = (j - 1) * block_size;
        const int previous_block_size = std::min(block_size, n - previous_panel_col);

        const int current_panel_col = j * block_size;
        const int trailing_cols = n - current_panel_col;
        const int current_block_size = std::min(block_size, trailing_cols);

        const Matrix Q_jm1 = Q.block(0, previous_panel_col, m, previous_block_size);

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