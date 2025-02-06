#include <iostream>
#include <Eigen/Dense>

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
    Matrix A(3, 3);
    A << 12, -51, 4,
        6, 167, -68,
        -4, 24, -41;

    auto [Q, R] = cholesky_QR_decomposition(A);

    std::cout << "Matrix Q:" << std::endl;
    std::cout << Q << std::endl;

    std::cout << "Matrix R:" << std::endl;
    std::cout << R << std::endl;

    return 0;
}
