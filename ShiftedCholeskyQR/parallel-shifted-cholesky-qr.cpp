#include <utils/helper_algos.hpp>

// Parallel sCQR
constexpr std::pair<Matrix, Matrix> parallel_shifted_cholesky_QR(Matrix &A)
{
    const int num_rows = A.rows();
    const int num_cols = A.cols();
    const int num_threads = omp_get_max_threads();

    // Unit roundoff for double precision
    const double u = compute_unit_roundoff<double>();

    // Frobenius norm of A
    const double norm_A = A.norm();

    // Gram matrix
    Matrix W = Matrix::Zero(num_cols, num_cols);

    // Local Gram matrices
    std::vector<Matrix> local_W(num_threads, Matrix::Zero(num_cols, num_cols));

    // Q initialization
    Matrix Q(num_rows, num_cols);

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Slice A into a submatrix for this thread
        Matrix A_i = A.middleRows(start, end - start);

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

    // Stability shift
    const double s = 11 * num_cols * (num_rows + num_cols + 1) * std::sqrt(num_rows) * u * std::pow(norm_A, 2);

    // Apply shift to the diagonal of the Gram matrix
    W.diagonal().array() += s;

    // Cholesky factorization of Gram matrix
    Matrix cholesky_factorization(W);
    const Matrix R = cholesky_factorization.matrixU();

    // Compute R inverse
    Matrix R_inv = R.inverse();

    // Compute Q in parallel using pre-sliced A
#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int chunk_size = num_rows / num_threads;

        const int start = thread_id * chunk_size;
        const int end = (thread_id == num_threads - 1) ? num_rows : start + chunk_size;

        // Calculate Q slices
        Q.middleRows(start, end - start) *= R_inv;
    }

    return {Q, R};
}