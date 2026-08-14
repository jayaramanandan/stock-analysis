#ifndef STOCK_ANALYSIS_MACROS_HPP
#define STOCK_ANALYSIS_MACROS_HPP

namespace LinearAlgebra {
    struct Functor {};
}

#define FUNCTORIFY(name, return_type, body) \
struct name : LinearAlgebra::Functor { \
    KOKKOS_FUNCTION \
    return_type operator() body \
}

#endif
