#ifndef STOCK_ANALYSIS_MATRIX_TPP
#define STOCK_ANALYSIS_MATRIX_TPP

#include <array>
#include <utility>
#include "Matrix.hpp"

namespace LinearAlgebra {
    template<std::size_t... Indices, typename Function>
    constexpr auto spreadFuncToArray(const std::index_sequence<Indices...>, const Function func) {
        return std::array{
            func(static_cast<int>(Indices))...
        };
    }

    template <std::size_t Length, typename Function>
    constexpr auto spreadFuncToArray(const Function func) {
        return spreadFuncToArray(std::make_index_sequence<Length>{}, func);
    }

    /*
    template<std::size_t... Indices, typename Function, std::size_t Length, typename ArrayType>
    constexpr auto spreadArrayToFunc(const Function func, std::array<ArrayType, Length>& array, std::index_sequence<Indices...>) {
        return func(array[Indices]...);
    }

    template<typename Function, std::size_t Length, typename ArrayType>
    constexpr auto spreadArrayToFunc(const Function func, std::array<ArrayType, Length>& array) {
        return spreadArrayToFunc(func, array, std::make_index_sequence<Length>{});
    }
    */

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType>::Matrix(Kokkos::View<MatrixTypePointer> matrixView):
    m(matrixView),
    shape(spreadFuncToArray(Dimensions, matrixView.extent))
    {}

    template<std::size_t Dimensions, typename MatrixType>
    template<typename... Shape>
    Matrix<Dimensions, MatrixType>::Matrix(Shape... shape):
    m("LinearAlgebra::Matrix", shape...),
    shape(std::array<std::size_t, Dimensions>{static_cast<std::size_t>(shape)...}) {
        static_assert(
            sizeof...(Shape) == Dimensions,
            "Number of dimensions must match Dimensions"
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Kokkos::View<typename Matrix<Dimensions, MatrixType>::MatrixTypePointer> Matrix<Dimensions, MatrixType>::getM() const {
        return this->m;
    }
}

#endif
