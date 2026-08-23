#ifndef STOCK_ANALYSIS_HELPERFUNCTIONS_HPP
#define STOCK_ANALYSIS_HELPERFUNCTIONS_HPP

#include <cmath>

#include "LinearAlgebra/Matrix/Matrix.hpp"

namespace LinearAlgebra {
    template <typename T, typename... Ts>
    concept SameType = (std::is_same_v<T, Ts>&& ...);

    template <auto Start, auto Stop, auto Step>
    constexpr std::size_t calculateLength() {
        return static_cast<std::size_t>(
            std::ceil((Stop - Start) / Step)
        );
    }

    template <auto Start, auto Stop, std::size_t Length>
    Matrix<decltype(Start), Length> linspace();

    template <auto Start, auto Stop, auto Step>
    Matrix<decltype(Start), calculateLength<Start, Stop, Step>()> arange();

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2, typename KokkosMathsFunction>
    Matrix<MatrixType, Dimension1, Dimension2> applyFunction(const Matrix<MatrixType, Dimension1, Dimension2>& matrix, KokkosMathsFunction function);

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> exp(const Matrix<MatrixType, Dimension1, Dimension2>& matrix);

}

#include "HelperFunctions.tpp"

#endif
