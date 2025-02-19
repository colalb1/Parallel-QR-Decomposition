#include <utils/helper_algos.hpp>

std::pair<Matrix, Matrix> modified_cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    int const m = A.rows();
    int const n = A.cols();
    int const block_size = std::min(64, n);          // Adjust block size dynamically
    int const k = (n + block_size - 1) / block_size; // Number of panels
    int const num_threads = omp_get_max_threads();

    Matrix Q = Matrix::Zero(m, n);
    Matrix R = Matrix::Zero(n, n);

    // Orthogonalize first panel
    auto [Q_1, R_11] = cholesky_QR_2(A.block(0, 0, m, block_size));

    Q.block(0, 0, m, block_size) = Q_1;
    R.block(0, 0, block_size, block_size) = R_11;

    // FIXME: Should j < k + 1? Are the loops in the pseudocode 1-indexed?
    for (int j = 1; j < k; ++j)
    {
        int const previous_panel_col = (j - 1) * block_size;
        int const previous_block_size = std::min(block_size, n - previous_panel_col);

        int const current_panel_col = j * block_size;
        int const trailing_cols = n - current_panel_col;
        int const current_block_size = std::min(block_size, trailing_cols);

        Matrix Q_jm1 = Q.block(0, previous_panel_col, m, previous_block_size);

        // Projections of orthogonal panels
        Matrix Y = Q_jm1.transpose() * A.block(0, current_panel_col, m, trailing_cols);

        // Update trailing panels
        A.block(0, current_panel_col, m, trailing_cols) -= Q_jm1 * Y;
        R.block(previous_panel_col, current_panel_col, previous_block_size, trailing_cols) = Y;

        // Process current panel j (after trailing update)
        Matrix A_j = A.block(0, current_panel_col, m, current_block_size);
        auto [Q_j, R_jj] = parallel_cholesky_QR(A_j);

        // Reorthogonalize current panel with respect to Q_prev
        Matrix Q_previous = Q.block(0, 0, m, current_panel_col); // To current_panel_col because the 1:(j - 1) panels implies stopping at the index of the j^th panel
        Matrix proj = Q_previous.transpose() * Q_previous;
        A_j -= proj * A_j;
        A.block(0, current_panel_col, m, current_block_size) = A_j;

        // Fully orthogonalize current panel
        Q.block(0, current_panel_col, m, current_block_size) = Q_j;
        R.block(current_panel_col, current_panel_col, current_block_size, current_block_size) = R_jj;
    }

    return {Q, R};
}