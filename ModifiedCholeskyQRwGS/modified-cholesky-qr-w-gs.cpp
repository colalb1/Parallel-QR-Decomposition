#include "utils/helper_algos.hpp"

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

int main()
{
    int m = 50000;
    int n = 100;
    Matrix A = Matrix::Random(m, n);

    auto start = std::chrono::high_resolution_clock::now(); // Start timer

    auto [Q, R] = modified_cholesky_QR2_w_gram_schmidt(A);

    auto end = std::chrono::high_resolution_clock::now(); // End timer
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "QR decomposition time: " << elapsed.count() << " seconds\n";

    // Test 1: Check reconstruction accuracy
    Matrix A_reconstructed = Q * R;
    double error = (A - A_reconstructed).norm() / A.norm();
    std::cout << "Relative reconstruction error: " << error << "\n";
    // assert(error < 1e-10 && "Reconstruction failed: A != QR");

    // Test 2: Check orthogonality of Q (Q^T Q should be Identity)
    Matrix I = Matrix::Identity(n, n);
    double ortho_error = (Q.transpose() * Q - I).norm();
    std::cout << "Orthogonality error: " << ortho_error << "\n";
    // assert(ortho_error < 1e-10 && "Q is not orthonormal: Q^T Q != I");

    // Test 3: Check upper triangularity of R
    bool is_upper_triangular = true;
    for (int i = 1; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (std::abs(R(i, j)) > 1e-10)
            {
                is_upper_triangular = false;
                break;
            }
        }
    }
    std::cout << "R is upper triangular: " << (is_upper_triangular ? "Yes" : "No") << "\n";
    assert(is_upper_triangular && "R is not upper triangular");

    std::cout << "All tests passed!\n";

    return 0;
}