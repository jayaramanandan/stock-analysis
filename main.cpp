#include <iostream>

#include <Kokkos_Core.hpp>

#include <LinearAlgebra.hpp>

FUNCTOR(float, fillVector, (const int i)) {
    return static_cast<float>(i);
}
END_FUNCTOR

FUNCTOR(float, fillMatrix, (const int i, const int j)) {
    return static_cast<float>(i + j);
}
END_FUNCTOR


BEGIN_PROGRAM

    LinearAlgebra::Matrix<2, float> matrix1(4, 5);

    matrix1.fill(
        KOKKOS_LAMBDA(const int i, const int j) {
            return static_cast<float>(i + j);
        }
    );

    std::cout << matrix1.toString() << std::endl;

END_PROGRAM
