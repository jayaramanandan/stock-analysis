#include <iostream>

#include <LinearAlgebra.hpp>

BEGIN_PROGRAM

    LinearAlgebra::Matrix<2, float> matrix1(4, 3);
    LinearAlgebra::Matrix<2, float> matrix2(4, 3);

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

    //std::cout << (matrix1 + matrix2).toString() << std::endl;
    //std::cout << (matrix1 - matrix2).toString() << std::endl;

END_PROGRAM
