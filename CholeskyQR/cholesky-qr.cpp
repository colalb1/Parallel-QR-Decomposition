#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using Matrix = Eigen::MatrixXd;

// CQR
constexpr std::pair<Matrix, Matrix> cholesky_QR(Matrix &A)
{
    // Compute Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(A.transpose() * A);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}