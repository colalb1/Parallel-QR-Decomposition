#include <iostream>
#include <Eigen/Dense>
#include <chrono>

typedef Eigen::MatrixXd Matrix;

// Cholesky QR decomposition
std::pair<Matrix, Matrix> cholesky_QR_decomposition(const Matrix &A)
{
    // Compute Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(A.transpose() * A);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}

int main()
{
    // Tall matrix dimensions
    int num_rows = 100000;
    int num_cols = 100;
    int num_iterations = 200;

    // Timing for serial decomposition
    double total_time_serial = 0.0;
    for (int i = 0; i < num_iterations; ++i)
    {
        std::cout << "Iteration: " << i << " \n";

        // Generate a random tall matrix A
        Matrix A = Matrix::Random(num_rows, num_cols);

        auto start = std::chrono::high_resolution_clock::now();
        auto [Q, R] = cholesky_QR_decomposition(A);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        total_time_serial += elapsed.count();
    }
    double average_time_serial = total_time_serial / num_iterations;

    // Output results
    std::cout << "Average time for cholesky_QR_decomposition: " << average_time_serial << " seconds\n";
}
