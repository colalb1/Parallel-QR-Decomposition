#include <utils/helper_algos.hpp>

// Shifted Cholesky QR decomposition 3
std::pair<Matrix, Matrix> shifted_cholesky_QR_3(Matrix &A)
{
    // Initial shifted extraction (shift for stability)
    auto [Q_1, R_1] = parallel_shifted_cholesky_QR(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_2(Q_1);

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}
