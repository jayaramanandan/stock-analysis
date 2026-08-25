#include <iostream>

#include <LinearAlgebra.hpp>
#include <Optimise.hpp>

BEGIN_PROGRAM

    using deriv = Optimise::Derivative<
        float,
        0,
        Optimise::Power<float, 3, Optimise::Variable<float, 0>>
    >::derivative;

    constexpr auto derivative = deriv{};
    std::cout << derivative(2.0f) << std::endl;

END_PROGRAM
