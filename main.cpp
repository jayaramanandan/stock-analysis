#include <iostream>

#include <Kokkos_Core.hpp>

#include <LinearAlgebra/Vector.hpp>
#include <LinearAlgebra/Macros.hpp>

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

    /*
    Kokkos::initialize(argc, argv);

    {
        const int N = 10;

        // Allocate arrays
        Kokkos::View<double*> x("x", N);
        Kokkos::View<double*> y("y", N);

        // Fill x
        Kokkos::parallel_for(
            "Fill X",
            N,
            KOKKOS_LAMBDA(const int i)
            {
                x(i) = static_cast<double>(i);
            }
        );

        // Calculate y = x²
        Kokkos::parallel_for(
            "Square",
            N,
            KOKKOS_LAMBDA(const int i)
            {
                y(i) = x(i) * x(i);
            }
        );

        // Copy data back to the CPU
        auto x_host = Kokkos::create_mirror_view(x);
        auto y_host = Kokkos::create_mirror_view(y);

        Kokkos::deep_copy(x_host, x);
        Kokkos::deep_copy(y_host, y);

        // Print results
        for (int i = 0; i < N; ++i)
        {
            std::cout << x_host(i)
                      << "^2 = "
                      << y_host(i)
                      << '\n';
        }
    }

    // Shut down Kokkos
    Kokkos::finalize();
    */

    return 0;
}
