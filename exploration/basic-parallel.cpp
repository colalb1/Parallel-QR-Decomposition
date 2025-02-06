#include <iostream>
#include <vector>
#include <omp.h>

int main()
{
    std::vector<int> vec(10);

#pragma omp parallel for
    for (int i = 0; i < vec.size(); ++i)
    {
        vec[i] = i;
    }

    for (const auto &val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}