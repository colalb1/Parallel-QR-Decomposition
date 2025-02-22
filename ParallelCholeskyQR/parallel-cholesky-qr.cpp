#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>

typedef Eigen::MatrixXd Matrix;

// Parallel Cholesky QR decomposition
std::pair<Matrix, Matrix> parallel_cholesky_QR(const Matrix &A)
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

        // Use a critical section to safely update the global Gram matrix W
#pragma omp critical
        {
            W += local_W[thread_id];
        }

        // Compute Q slice directly; I will multiply by R inverse later
        // This reduces storage by not storing A_i slices
        Q.middleRows(start, end - start) = A_i;
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
    int num_iterations = 10;

    // Timing for parallel decomposition
    double total_time_parallel = 0.0;
    for (int i = 0; i < num_iterations; ++i)
    {
        std::cout << "Iteration: " << i << " \n";

        // Generate a random tall matrix A
        Matrix A = Matrix::Random(num_rows, num_cols);

        auto start = std::chrono::high_resolution_clock::now();
        auto [Q, R] = parallel_cholesky_QR(A);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        total_time_parallel += elapsed.count();
    }
    double average_time_parallel = total_time_parallel / num_iterations;

    // Output results
    std::cout << "Average time for parallel_cholesky_QR_decomposition: " << average_time_parallel << " seconds\n";

    // // Generate tall matrix
    // Matrix A = Matrix::Random(num_rows, num_cols);

    // // QR decomposition
    // auto [Q, R] = parallel_cholesky_QR_decomposition(A);

    // // Verify Q orthogonality
    // Matrix Q_transpose_Q = Q.transpose() * Q;
    // Matrix identity = Matrix::Identity(num_cols, num_cols);
    // double orthogonality_error = (Q_transpose_Q - identity).norm();

    // std::cout << "Orthogonality error (Q^T * Q - I): " << orthogonality_error << "\n";

    // // Verify Q * R reconstructs A
    // Matrix A_reconstructed = Q * R;
    // double reconstruction_error = (A_reconstructed - A).norm();

    // std::cout << "Reconstruction error (Q * R - A): " << reconstruction_error << "\n";

    // // Check if errors are within tolerance
    // double tolerance = 1e-6;
    // if (orthogonality_error < tolerance && reconstruction_error < tolerance)
    // {
    //     std::cout << "Test passed! The QR decomposition is CORRECT!!\n";
    // }
    // else
    // {
    //     std::cout << "Test failed! The QR decomposition is INCORRECT.\n";
    // }

    return 0;
}