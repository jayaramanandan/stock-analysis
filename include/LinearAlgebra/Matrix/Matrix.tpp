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

    template<std::size_t... Indices, typename Function, std::size_t Length, typename ArrayType>
    constexpr auto spreadArrayToFunc(const Function func, std::array<ArrayType, Length>& array, std::index_sequence<Indices...>) {
        return func(array[Indices]...);
    }

    template<typename Function, std::size_t Length, typename ArrayType>
    constexpr auto spreadArrayToFunc(const Function func, std::array<ArrayType, Length>& array) {
        return spreadArrayToFunc(func, array, std::make_index_sequence<Length>{});
    }

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

    template<std::size_t Dimensions, typename MatrixType>
    std::string Matrix<Dimensions, MatrixType>::toString() const {
        static_assert(
            Dimensions == 1 || Dimensions == 2,
            "Dimensions must be 1 or 2."
        );

        auto hostM = Kokkos::create_mirror_view(this->m);
        Kokkos::deep_copy(hostM, this->m);

        std::string outputString = "[\n";

        for (std::size_t i = 0; i < this->shape[0]; i++) {
            outputString += '\t';

            if constexpr (Dimensions == 1) {
                outputString += std::to_string(hostM(i)) + ", ";
            } else {
                for (std::size_t j = 0; j < this->shape[1]; j++) {
                    outputString += std::to_string(hostM(i, j)) + ", ";
                }
            }

            outputString += '\n';
        }

        outputString += "\n]";

        return outputString;
    }

    template<std::size_t Dimensions, typename MatrixType>
    template <typename KokkosFunction>
    void Matrix<Dimensions, MatrixType>::iterateElements(const KokkosFunction kokkosCallback) const {
        const auto shapeCopy = this->shape;
        Kokkos::parallel_for(
            "LinearAlgebra::Matrix::iterateElements",
            Kokkos::MDRangePolicy(
                std::array<std::size_t, Dimensions>{},
                shape
            ),
            kokkosCallback
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    template<typename KokkosFunction>
    void Matrix<Dimensions, MatrixType>::fill(const KokkosFunction fillFunction) {
        this->iterateElements(
            KOKKOS_LAMBDA(const auto... indices) {
                mCopy(indices...) = fillFunction(indices...);
            }
        );
    }
}

#endif
