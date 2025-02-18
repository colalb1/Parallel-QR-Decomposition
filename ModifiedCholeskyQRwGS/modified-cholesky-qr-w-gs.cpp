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
        Matrix current_A = A.block(0, current_panel_col, m, current_block_size);
        // auto [Qj_temp, R_temp] = parallel_cholesky_QR(current_A);

        // // Reorthogonalize current panel with respect to Q_prev
        // Matrix proj = Q_prev.transpose() * Qj_temp;
        // Qj_temp -= Q_prev * proj;

        // // Perform CQR again on reorthogonalized panel
        // auto [Q_j, R_jj] = parallel_cholesky_QR(Qj_temp);

        // // Update Q and R
        // Q.block(0, current_panel_start, m, s) = Q_j;
        // R.block(current_panel_start, current_panel_start, s, s) = R_jj;
    }

    return {Q, R};
}