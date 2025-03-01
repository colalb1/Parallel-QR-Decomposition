#include <utils/helper_algos.hpp>

// Forward declarations of QR decomposition functions
std::pair<Matrix, Matrix> cholesky_QR(Matrix &A);
std::pair<Matrix, Matrix> parallel_cholesky_QR(Matrix &A);

std::pair<Matrix, Matrix> cholesky_QR2_w_gram_schmidt(Matrix &A);
std::pair<Matrix, Matrix> distributed_cholesky_QR_w_gram_schmidt(Matrix &A);
std::pair<Matrix, Matrix> modified_cholesky_QR2_w_gram_schmidt(Matrix &A);

double measure_computation_time(std::function<std::pair<Matrix, Matrix>(Matrix &)> &func, Matrix &A)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(A);
    auto end = std::chrono::high_resolution_clock::now();

    // Duration in seconds
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

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
        {48'000, 300}};

    const int number_of_tests_per_size = 10;

    // Open CSV file for writing
    std::ofstream csv_file("data/pre_optimization_speed_test_results.csv");
    csv_file << "ALGORITHM,NUMBER_ROWS,NUMBER_COLUMNS,TIME_S\n";

    // Running speed tests
    for (const auto &[m, n] : matrix_sizes)
    {
        Matrix A = Matrix::Random(m, n);

        std::vector<std::pair<std::function<std::pair<Matrix, Matrix>(Matrix &)>, std::string>> qr_functions = {
            // {cholesky_QR, "cholesky_QR"},
            {parallel_cholesky_QR, "parallel_cholesky_QR"},
            {cholesky_QR2_w_gram_schmidt, "cholesky_QR2_w_gram_schmidt"},
            {distributed_cholesky_QR_w_gram_schmidt, "distributed_cholesky_QR_w_gram_schmidt"},
            {modified_cholesky_QR2_w_gram_schmidt, "modified_cholesky_QR2_w_gram_schmidt"}};

        for (auto &[qr_func, qr_func_name] : qr_functions)
        {
            for (int i = 0; i < number_of_tests_per_size; ++i)
            {
                double time = measure_computation_time(qr_func, A);

                csv_file << qr_func_name << "," << m << "," << n << "," << time << "\n";

                std::cout << qr_func_name << " " << m << "x" << n << " " << time << "s\n";
            }
        }
    }

    csv_file.close();

    return 0;
}