#include <utils/helper_algos.hpp>

std::pair<Matrix, Matrix> cholesky_QR_w_gram_schmidt(const Matrix &A)
{
    //     int n = A.rows();
    //     Matrix R = A.transpose() * A;
    //     Matrix Q = Matrix::Identity(n, n);

    //     // Cholesky decomposition
    //     for (int k = 0; k < n; ++k)
    //     {
    //         R(k, k) = std::sqrt(R(k, k));
    // #pragma omp parallel for
    //         for (int i = k + 1; i < n; ++i)
    //         {
    //             R(i, k) /= R(k, k);
    //         }
    // #pragma omp parallel for collapse(2)
    //         for (int j = k + 1; j < n; ++j)
    //         {
    //             for (int i = j; i < n; ++i)
    //             {
    //                 R(i, j) -= R(i, k) * R(j, k);
    //             }
    //         }
    //     }

    //     // Gram-Schmidt process
    //     for (int k = 0; k < n; ++k)
    //     {
    //         Q.col(k) = A.col(k);
    //         for (int j = 0; j < k; ++j)
    //         {
    //             double dot_product = Q.col(j).dot(A.col(k));
    //             Q.col(k) -= dot_product * Q.col(j);
    //         }
    //         Q.col(k).normalize();
    //     }

    //     return {Q, R};
}
