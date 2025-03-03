#include <utils/helper_algos.hpp>

int main()
{
    const std::vector<std::pair<int, int>> matrix_sizes = {
        {3'000, 300},
        {12'000, 120},
        {12'000, 600},
        {12'000, 1'200},
        {4'000, 300},
        {8'000, 300},
        {12'000, 300},
    };

    // Running speed tests
    for (const auto &[m, n] : matrix_sizes)
    {
        Matrix A = Matrix::Random(m, n);

        cholesky_QR(A);
        parallel_cholesky_QR(A);
        cholesky_QR2_w_gram_schmidt(A);
        distributed_cholesky_QR_w_gram_schmidt(A);
        modified_cholesky_QR2_w_gram_schmidt(A);
    }

    return 0;
}