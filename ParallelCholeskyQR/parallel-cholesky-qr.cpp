#include <iostream>
#include <Eigen/Dense>
#include <omp.h>

typedef Eigen::MatrixXd Matrix;

std::pair<Matrix, Matrix> parallel_cholesky_QR_decomposition(const Matrix &A)
{
    int num_rows = A.rows();
    int num_cols = A.cols();
    int num_threads = omp_get_max_threads();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        Eigen::MatrixXd A_i = A.middleRows(start, end - start);

        // Compute A_i^T * A_i for the sliced submatrix
        local_W[thread_id].noalias() += A_i.transpose() * A_i;

        // Compute Q slice directly; I will multiply by R inverse later
        // This reduces storage by not storing A_i slices
        Q.middleRows(start, end - start) = A_i;
    }

    // Sum local Gram matrices
    // #pragma omp parallel for reduction(+ : W)
    for (int i = 0; i < num_threads; ++i)
    {
        W += local_W[i];
    }

    // Cholesky factorization of Gram matrix
    Eigen::LLT<Matrix> cholesky_factorization(W);
    Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = num_rows / num_threads;

        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}

int main()
{
    // Tall matrix dimensions
    int num_rows = 100000;
    int num_cols = 100;

    // Generate tall matrix
    Matrix A = Matrix::Random(num_rows, num_cols);

    // QR decomposition
    auto [Q, R] = parallel_cholesky_QR_decomposition(A);

    // Verify Q orthogonality
    Matrix Q_transpose_Q = Q.transpose() * Q;
    Matrix identity = Matrix::Identity(num_cols, num_cols);
    double orthogonality_error = (Q_transpose_Q - identity).norm();

    std::cout << "Orthogonality error (Q^T * Q - I): " << orthogonality_error << "\n";

    // Verify Q * R reconstructs A
    Matrix A_reconstructed = Q * R;
    double reconstruction_error = (A_reconstructed - A).norm();

    std::cout << "Reconstruction error (Q * R - A): " << reconstruction_error << "\n";

    // Check if errors are within tolerance
    double tolerance = 1e-6;
    if (orthogonality_error < tolerance && reconstruction_error < tolerance)
    {
        std::cout << "Test passed! The QR decomposition is CORRECT!!\n";
    }
    else
    {
        std::cout << "Test failed! The QR decomposition is INCORRECT.\n";
    }

    return 0;
}