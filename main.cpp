#include <iostream>

#include <LinearAlgebra.hpp>

BEGIN_PROGRAM

    LinearAlgebra::Matrix<float, 4, 5> matrix1;
    LinearAlgebra::Matrix<float, 4, 5> matrix2;

    matrix1.fill(
        LAMBDA(const int i, const int j) {
            return static_cast<float>(i + j);
        }
    );

    matrix2.fill(
        LAMBDA(const int i, const int j) {
            return static_cast<float>(i);
        }
    );

    std::cout << matrix1.toString() << std::endl;
    std::cout << matrix2.toString() << std::endl;

    std::cout << (100.0f - matrix1 + 100.0f - matrix1).toString() << std::endl;
    std::cout << (100.0f - matrix1).toString() << std::endl;
    std::cout << (matrix1 * 2.0f - 2.0f * matrix1).toString() << std::endl;
    std::cout << (matrix1 / 2.0f - 2.0f / matrix1).toString() << std::endl;

    std::cout << (matrix1 + matrix2).toString() << std::endl;
    std::cout << (matrix1 - matrix2).toString() << std::endl;

END_PROGRAM
