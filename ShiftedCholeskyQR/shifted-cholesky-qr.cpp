#include <utils/helper_algos.hpp>

// Compute unit roundoff for given floating-point type
template <typename T>
T compute_unit_roundoff()
{
    T u = 1.0;

    while (1.0 + u != 1.0)
    {
        u /= 2.0;
    }

    return u;
}

// Shifted Cholesky QR
std::pair<Matrix, Matrix> shifted_cholesky_QR(Matrix &A)
{
    // Number of cols for shift application
    int const num_rows = A.rows();
    int const num_cols = A.cols();

    // Unit roundoff for double precision
    double const u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    double const norm_A = A.norm();

    // Stability shift
    // From this link: https://arxiv.org/abs/1809.11085
    double const s = 11 * num_cols * (num_rows + num_cols + 1) * std::sqrt(num_rows) * u * std::pow(norm_A, 2);

    // Compute shifted Gram matrix
    Matrix G = A.transpose() * A;
    G.diagonal().array() += s; // Shift diagonal

    // Perform Cholesky factorization
    Eigen::LLT<Matrix> cholesky_factorization(G);

    // Get upper triangular Cholesky factor R
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    Matrix Q = A * R.inverse();

    return {Q, R};
}