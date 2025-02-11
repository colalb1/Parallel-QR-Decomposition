#include <utils/helper_algos.hpp>

// Shifted Cholesky QR decomposition 3
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

int main()
{
    // Tall matrix dimensions
    int num_rows = 100000;
    int num_cols = 100;

    // Timing
    double total_time = 0.0;

    // Generate a random tall matrix A
    Matrix A = Matrix::Random(num_rows, num_cols);

    auto start = std::chrono::high_resolution_clock::now();
    auto [Q, R] = shifted_cholesky_QR_3(A);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    total_time += elapsed.count();

    // Output results
    std::cout << "Time for shifted_cholesky_QR_3: " << total_time << " seconds\n";

    return 0;
}