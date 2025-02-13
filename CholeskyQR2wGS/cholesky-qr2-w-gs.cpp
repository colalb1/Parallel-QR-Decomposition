#include <utils/helper_algos.hpp>

// CQR2GS
std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    // Initial extraction
    auto [Q_1, R_1] = cholesky_QR_w_gram_schmidt(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_w_gram_schmidt(Q_1);

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}