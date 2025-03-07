#include <utils/helper_algos.hpp>

// CQR2GS
constexpr std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A)
{
    // Initial extraction
    auto [Q_1, R_1] = cholesky_QR_w_gram_schmidt(A);

    // Second Q and R extraction
    auto [Q, R_2] = cholesky_QR_w_gram_schmidt(Q_1);

    // Set threads
    omp_set_num_threads(omp_get_max_threads());

    // Calculate final R
    Matrix R = R_2 * R_1;

    return {Q, R};
}