#include <iostream>

#include <LinearAlgebra.hpp>
#include <Optimise.hpp>

BEGIN_PROGRAM
    constexpr std::size_t parametersLength = 2;

    using deriv = Optimise::Derivative<
        float,
        0,
        Optimise::Power<float, 3, Optimise::Variable<float, 0, parametersLength>, parametersLength>,
        parametersLength
    >::derivative;

    constexpr auto derivative = deriv{};

    const LinearAlgebra::Matrix x = std::array{0.0f, 0.0f};
    std::cout << derivative(x) << std::endl;

END_PROGRAM
