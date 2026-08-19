#ifndef STOCK_ANALYSIS_MACROS_HPP
#define STOCK_ANALYSIS_MACROS_HPP

// Main file macros

#define BEGIN_PROGRAM \
int main(int argc, char* argv[]) { \
Kokkos::initialize(argc, argv); \
{ \
std::cout << "Execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';

#define END_PROGRAM \
} \
Kokkos::finalize(); \
return 0; \
}

// Functor macros

namespace LinearAlgebra {
    struct Functor {};
}

#define FUNCTOR(return_type, name, args) \
struct name : LinearAlgebra::Functor { \
    KOKKOS_FUNCTION \
    return_type operator() args const \

#define END_FUNCTOR };

#endif
