#include <utils/helper_algos.hpp>

// sCQR
constexpr std::pair<Matrix, Matrix> shifted_cholesky_QR(Matrix &A)
{
    // Number of cols for shift application
    const int num_rows = A.rows();
    const int num_cols = A.cols();

    // Unit roundoff for double precision
    const double u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    const double norm_A = A.norm();

    // Stability shift
    // From this link: https://arxiv.org/abs/1809.11085
    const double s = 11 * num_cols * (num_rows + num_cols + 1) * std::sqrt(num_rows) * u * std::pow(norm_A, 2);

    // Compute shifted Gram matrix
    Matrix G = A.transpose() * A;
    G.diagonal().array() += s; // Shift diagonal

    // Perform Cholesky factorization
    Matrix cholesky_factorization(G);

    // Get upper triangular Cholesky factor R
    const Matrix R = cholesky_factorization.matrixU();

    // Compute Q matrix
    const Matrix Q = A * R.inverse();

    return {Q, R};
}