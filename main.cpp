#include <iostream>

#include <LinearAlgebra.hpp>

BEGIN_PROGRAM

    const auto linspaceVals = LinearAlgebra::linspace<-5.0f, 5.0f, 10>();

    std::cout << linspaceVals.toString() << std::endl;

    std::cout << LinearAlgebra::exp(linspaceVals).toString() << std::endl;

END_PROGRAM
