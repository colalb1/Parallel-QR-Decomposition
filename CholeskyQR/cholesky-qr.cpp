#include <iostream>
#include <Eigen/Dense>

typedef Eigen::MatrixXd Matrix;

// Function to perform Cholesky QR decomposition
void choleskyQR(const Matrix &A, Matrix &Q, Matrix &R)
{
    Eigen::LLT<Matrix> llt(A.transpose() * A);
    R = llt.matrixU();
    Q = A * R.inverse().transpose();
}

int main()
{
    Matrix A(3, 3);
    A << 4, 12, -16,
        12, 37, -43,
        -16, -43, 98;

    Matrix Q, R;
    choleskyQR(A, Q, R);

    std::cout << "Matrix Q:" << std::endl;
    std::cout << Q << std::endl;

    std::cout << "Matrix R:" << std::endl;
    std::cout << R << std::endl;

    return 0;
}
