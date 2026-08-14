#include <iostream>

#include <Kokkos_Core.hpp>

#include <LinearAlgebra.hpp>

FUNCTORIFY(
    fill,
    float,
    (const int i) const {
        return static_cast<float>(i);
    }
);

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        std::cout << "Execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';

        LinearAlgebra::Vector<float> myVec1(5);
        LinearAlgebra::Vector<float> myVec2(5);

        myVec1.fill<fill>();
        myVec2.fill<fill>();

        std::cout << (myVec1 + myVec2).to_string() << std::endl;
        std::cout << (myVec1 - myVec2).to_string() << std::endl;
        std::cout << (myVec1 * myVec2).to_string() << std::endl;
        std::cout << (myVec1 / myVec2).to_string() << std::endl;

        std::cout << (myVec1 + 2.0f).to_string() << std::endl;
        std::cout << (myVec1 - 2.0f).to_string() << std::endl;
        std::cout << (myVec1 * 2.0f).to_string() << std::endl;
        std::cout << (myVec1 / 2.0f).to_string() << std::endl;

        std::cout << (2.0f + myVec1).to_string() << std::endl;

        std::cout << (1.0f / myVec1).to_string() << std::endl;
    }

    Kokkos::finalize();

    return 0;
}
