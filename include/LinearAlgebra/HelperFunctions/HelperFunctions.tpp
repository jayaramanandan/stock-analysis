#ifndef STOCK_ANALYSIS_HELPERFUNCTIONS_TPP
#define STOCK_ANALYSIS_HELPERFUNCTIONS_TPP

#include "HelperFunctions.hpp"

namespace LinearAlgebra {
    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2, typename KokkosMathsFunction>
    Matrix<MatrixType, Dimension1, Dimension2> indexFill(const KokkosMathsFunction function) {
        Matrix<MatrixType, Dimension1, Dimension2> result;

        Kokkos::parallel_for(
            "LinearAlgebra::indexFill::parallel_for",
            Kokkos::MDRangePolicy(
                {0, 0},
                {Dimension1, Dimension2}
            ),
            KOKKOS_LAMBDA(const int i, const int j) {
                result.view(i, j) = function(i, j);
            }
        );

        return result;
    }

    template<typename MathsFunction, typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> applyFunction(Matrix<MatrixType, Dimension1, Dimension2>&& matrix) {
        const MathsFunction mathsFunction{};
        Matrix<MatrixType, Dimension1, Dimension2> result;

        Kokkos::parallel_for(
            "LinearAlgebra::applyFunction::parallel_for",
            Kokkos::MDRangePolicy(
                {0, 0},
                {Dimension1, Dimension2}
            ),
            KOKKOS_LAMBDA(const int i, const int j) {
                result.view(i, j) = mathsFunction(matrix.view(i, j));
            }
        );

        return result;
    }

    template <auto Start, auto Stop, auto Step>
    constexpr std::size_t calculateLength() {
        return static_cast<std::size_t>(
            std::ceil((Stop - Start) / Step)
        );
    }

    template <auto Start, auto Stop, std::size_t Length>
    Matrix<decltype(Start), Length> linspace() {
        static_assert(Length > 1, "Length must be greater than 1");
        static_assert(SameType<decltype(Start), decltype(Stop)>, "Start and Stop must be same type");

        using MatrixType = decltype(Start);

        constexpr MatrixType spacing = (Stop - Start) / (Length - 1);

        return indexFill<MatrixType, Length, 1>(
            KOKKOS_LAMBDA(const int i, int) {
                return Start + static_cast<MatrixType>(i) * spacing;
            }
        );
    }

    template <auto Start, auto Stop, auto Step>
    Matrix<decltype(Start), calculateLength<Start, Stop, Step>()> arange() {
        static_assert(Start + Step <= Stop, "Step is too big and surpasses Stop");
        static_assert(SameType<decltype(Start), decltype(Stop), decltype(Step)>, "Start, Stop and Step must be same type");

        using MatrixType = decltype(Start);
        static constexpr auto length = calculateLength<Start, Stop, Step>();

        return indexFill<MatrixType, length, 1>(
            KOKKOS_LAMBDA(const int i, int) {
                return Start + i * Step;
            }
        );
    }
}

#endif
