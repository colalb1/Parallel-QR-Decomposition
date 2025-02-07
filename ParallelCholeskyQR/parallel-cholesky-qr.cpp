#include <iostream>
#include <Eigen/Dense>
#include <omp.h>

typedef Eigen::MatrixXd Matrix;

// BELOW NEEDS TO BE CHECKED
std::pair<Matrix, Matrix> parallel_cholesky_QR_decomposition(const Matrix &A)
{
    int num_rows = A.rows();
    int num_cols = A.cols();

    int num_threads = omp_get_max_threads();

    // A is tall and skinny (rows >> cols)
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Compute Gram matrix using multiple cores
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        for (int i = start; i < end; ++i)
        {
            local_W[thread_id].noalias() += A.row(i).transpose() * A.row(i);
        }
    }

    // Sum local Gram matrices
    for (const auto &W_i : local_W)
    {
        W += W_i;
    }

    // Cholesky factorization of Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(W);
    Matrix R = cholesky_factorization.matrixU();

    // Compute Q locally for a tall and skinny matrix
    Matrix Q(num_rows, num_cols);
    Matrix R_inv = R.inverse();

#pragma omp parallel for
    for (int i = 0; i < num_cols; ++i)
    {
        Q.col(i) = A * R_inv.col(i);
    }

    return {Q, R};
}