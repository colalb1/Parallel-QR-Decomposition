#include <utils/helper_algos.hpp>

// Compute unit roundoff for given floating-point type
template <typename T>
T compute_unit_roundoff() {
    T u = 1.0;

    while (1.0 + u != 1.0) {
        u /= 2.0;
    }

    return u;
}

// Shifted Cholesky QR
std::pair<Matrix, Matrix> shifted_cholesky_QR(const Matrix &A)
{
    // Number of cols for shift application
    int const num_rows = A.rows();
    int const num_cols = A.cols();

    // Unit roundoff for double precision
    double const u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    double const norm_A = A.norm();

    // Stability shift
    double s = std::sqrt(num_rows) * u * norm_A;

    // Compute shifted Gram matrix
    Matrix gram_matrix = A.transpose() * A;
    gram_matrix.diagonal().array() += s; // Shift diagonal

    // Perform Cholesky factorization
    Eigen::LLT<Matrix> cholesky_factorization(gram_matrix);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}