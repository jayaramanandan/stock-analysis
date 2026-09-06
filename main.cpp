#include <iostream>

#include <LinearAlgebra.hpp>
#include <Optimise.hpp>

template <typename T>
static void printType() {
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

BEGIN_PROGRAM
    constexpr std::size_t ParametersLength = 2;

    using expression = Optimise::Operation<
        float,
        LinearAlgebra::ValidMathOperation::Add,
        Optimise::Operation<
            float,
            LinearAlgebra::ValidMathOperation::Multiply,
            Optimise::Power<float, 3, Optimise::Variable<float, 0, ParametersLength>, ParametersLength>,
            Optimise::Power<float, 2, Optimise::Variable<float, 1, ParametersLength>, ParametersLength>,
            ParametersLength
        >,
        Optimise::Power<float, 5, Optimise::Variable<float, 1, ParametersLength>, ParametersLength>,
        ParametersLength
    >;

    using deriv = Optimise::Derivative<float, 0, expression, ParametersLength>;

    const auto params = LinearAlgebra::Vector<float, 2>();

    Kokkos::parallel_for(
        "EvaluateDerivative",
        1,
        KOKKOS_LAMBDA(const int) {
            params.view(0, 0) = 1;
            params.view(1, 0) = 100;

            const float result = deriv{}(params.view);

            Kokkos::printf("result = %f\n", result);
        }
    );

    Kokkos::fence();
END_PROGRAM
