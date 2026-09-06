#include <iostream>

#include <LinearAlgebra.hpp>
#include <Optimise.hpp>

BEGIN_PROGRAM
    const auto mat2 = LinearAlgebra::linspace<5.0f, 10.0f, 5>();
    const auto mat1 = LinearAlgebra::applyFunction<LinearAlgebra::log>(LinearAlgebra::Matrix<float, 5, 5>() + 1.0f);

    std::cout << LinearAlgebra::matrixToString(mat1) << std::endl;

    std::cout << LinearAlgebra::matrixToString(mat2) << std::endl;

    std::cout << LinearAlgebra::matrixToString(mat1 * mat2) << std::endl;
END_PROGRAM
