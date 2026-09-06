#ifndef STOCK_ANALYSIS_HELPERFUNCTIONS_HPP
#define STOCK_ANALYSIS_HELPERFUNCTIONS_HPP

#include <cmath>

#include <Kokkos_MathematicalFunctions.hpp>

#include "../Matrix/Matrix.hpp"

#define DEFINE_FUNCTION(name) \
    struct name { \
        template <typename MatrixType> \
        KOKKOS_FUNCTION \
        auto operator()(MatrixType x) const {\
            return Kokkos::name(x); \
        } \
    };

namespace LinearAlgebra {
    template <typename T, typename... Ts>
    concept SameType = (std::is_same_v<T, Ts>&& ...);

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2, typename KokkosMathsFunction>
    Matrix<MatrixType, Dimension1, Dimension2> indexFill(KokkosMathsFunction function);

    template<typename MathsFunction, typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> applyFunction(Matrix<MatrixType, Dimension1, Dimension2>&& matrix);

    template <auto Start, auto Stop, auto Step>
    constexpr std::size_t calculateLength();

    template <auto Start, auto Stop, std::size_t Length>
    Matrix<decltype(Start), Length> linspace();

    template <auto Start, auto Stop, auto Step>
    Matrix<decltype(Start), calculateLength<Start, Stop, Step>()> arange();

    DEFINE_FUNCTION(exp)
    DEFINE_FUNCTION(log)
    DEFINE_FUNCTION(sqrt)
    DEFINE_FUNCTION(sin)
    DEFINE_FUNCTION(cos)
}

#include "HelperFunctions.tpp"

#endif
