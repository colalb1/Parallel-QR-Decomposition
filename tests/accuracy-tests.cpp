#include "utils/helper_algos.hpp"

// Forward declarations of QR decomposition functions
std::pair<Matrix, Matrix> cholesky_QR(Matrix &A);
std::pair<Matrix, Matrix> parallel_cholesky_QR(Matrix &A);
std::pair<Matrix, Matrix> cholesky_QR_2(Matrix &A);
std::pair<Matrix, Matrix> shifted_cholesky_QR(Matrix &A);
std::pair<Matrix, Matrix> parallel_shifted_cholesky_QR(Matrix &A);
std::pair<Matrix, Matrix> shifted_cholesky_QR_3(Matrix &A);
std::pair<Matrix, Matrix> cholesky_QR_w_gram_schmidt(Matrix &A);
std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A);
std::pair<Matrix, Matrix> distributed_cholesky_QR_w_gram_schmidt(Matrix &A);
std::pair<Matrix, Matrix> modified_cholesky_QR2_w_gram_schmidt(Matrix &A);

// Test function for QR decomposition
void test_qr_decomposition(const std::function<std::pair<Matrix, Matrix>(Matrix &)> &qr_func, Matrix &A, const std::string &funcName)
{
    auto [Q, R] = qr_func(A);

    // Test 1: Check reconstruction accuracy
    Matrix A_reconstructed = Q * R;
    double error = (A - A_reconstructed).norm() / A.norm();
    std::cout << "[" << funcName << "] Relative reconstruction error: " << error << "\n";

    // Test 2: Check orthogonality of Q (Q^T Q should be Identity)
    Matrix I = Matrix::Identity(Q.cols(), Q.cols());
    double ortho_error = (Q.transpose() * Q - I).norm();
    std::cout << "[" << funcName << "] Orthogonality error: " << ortho_error << "\n";

    // Test 3: Check upper triangularity of R
    double lower_triangular_error = 0.0;
    for (int i = 1; i < R.rows(); ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            lower_triangular_error += std::abs(R(i, j)); // Sum the absolute values of the lower triangular elements
        }
    }
    std::cout << "[" << funcName << "] Lower triangular error of R: " << lower_triangular_error << "\n";
}

int main()
{
    int m = 50000;
    int n = 500;
    Matrix A = Matrix::Random(m, n); // Use Eigen::Matrix for dynamic-sized matrices

    // Define the type for the vector of QR functions
    using QRFuncPair = std::pair<std::function<std::pair<Matrix, Matrix>(Matrix &)>, std::string>;

    // Create an empty vector to hold the QR functions and their names
    std::vector<QRFuncPair> qr_functions;

    // Add each QR decomposition function to the vector
    qr_functions.push_back({cholesky_QR, "cholesky_QR"});
    qr_functions.push_back({parallel_cholesky_QR, "parallel_cholesky_QR"});
    qr_functions.push_back({cholesky_QR_2, "cholesky_QR_2"});
    qr_functions.push_back({shifted_cholesky_QR, "shifted_cholesky_QR"});
    qr_functions.push_back({parallel_shifted_cholesky_QR, "parallel_shifted_cholesky_QR"});
    qr_functions.push_back({shifted_cholesky_QR_3, "shifted_cholesky_QR_3"});
    qr_functions.push_back({cholesky_QR_w_gram_schmidt, "cholesky_QR_w_gram_schmidt"});
    qr_functions.push_back({cholesky_QR2_w_gram_schmidt, "cholesky_QR2_w_gram_schmidt"});
    qr_functions.push_back({distributed_cholesky_QR_w_gram_schmidt, "distributed_cholesky_QR_w_gram_schmidt"});
    qr_functions.push_back({modified_cholesky_QR2_w_gram_schmidt, "modified_cholesky_QR2_w_gram_schmidt"});

    // Test each QR decomposition function
    for (const auto &[qr_func, funcName] : qr_functions)
    {
        test_qr_decomposition(qr_func, A, funcName);
    }

    return 0;
}