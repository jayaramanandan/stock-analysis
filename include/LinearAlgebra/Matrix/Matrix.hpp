#ifndef STOCK_ANALYSIS_MATRIXOPERATIONS_HPP
#define STOCK_ANALYSIS_MATRIXOPERATIONS_HPP

#include <string>

#include <Kokkos_Core.hpp>

#include "LinearAlgebra/Settings.hpp"

namespace LinearAlgebra {
    template <typename T, std::size_t N>
    struct AddPointers {
        using type = AddPointers<T, N - 1>::type*;
    };

    template <typename T>
    struct AddPointers<T, 0> {
        using type = T;
    };

    template <typename T>
    using KokkosView = Kokkos::View<typename AddPointers<T, MAX_DIMENSIONS>::type, Kokkos::DefaultExecutionSpace>;

    // Keep matrix host only
    template <typename MatrixType, std::size_t Dimension1, std::size_t Dimension2 = 1>
    struct Matrix {
        KokkosView<MatrixType> view;

        Matrix();
    };

    template <typename MatrixType, std::size_t Length>
    using Vector = Matrix<MatrixType, Length>;

    template <typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    std::string matrixToString(const Matrix<MatrixType, Dimension1, Dimension2>& matrix);
}

#include "Matrix.tpp"

#endif
