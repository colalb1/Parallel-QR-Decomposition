#include <utils/helper_algos.hpp>

// CQR2
constexpr std::pair<Matrix, Matrix> cholesky_QR_2(Matrix &A)
{
    // Initial Q and R extraction; this computation will be performed again
    // to increase numerical accuracy.
    auto [Q_1, R_1] = parallel_cholesky_QR(A);

    // Final Q and R extraction
    auto [Q, R_2] = parallel_cholesky_QR(Q_1);

    // Calculate final R
    const Matrix R = R_2 * R_1;

    return {Q, R};
}
