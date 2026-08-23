#ifndef STOCK_ANALYSIS_HELPERFUNCTIONS_TPP
#define STOCK_ANALYSIS_HELPERFUNCTIONS_TPP

#include "HelperFunctions.hpp"

#include <Kokkos_MathematicalFunctions.hpp>

#include "LinearAlgebra/Macros.hpp"

namespace LinearAlgebra {
    template <auto Start, auto Stop, std::size_t Length>
    Matrix<decltype(Start), Length> linspace() {
        static_assert(Length > 1, "Length must be greater than 1");
        static_assert(SameType<decltype(Start), decltype(Stop)>, "Start and Stop must be same type");

        using MatrixType = decltype(Start);

        Matrix<MatrixType, Length> result;

        constexpr MatrixType spacing = (Stop - Start) / (Length - 1);

        result.fill(
            LAMBDA(const int i, int) {
                return Start + static_cast<MatrixType>(i) * spacing;
            }
        );

        return result;
    }

    template <auto Start, auto Stop, auto Step>
    Matrix<decltype(Start), calculateLength<Start, Stop, Step>()> arange() {
        static_assert(Start + Step <= Stop, "Step is too big and surpasses Stop");
        static_assert(SameType<decltype(Start), decltype(Stop), decltype(Step)>, "Start, Stop and Step must be same type");

        using MatrixType = decltype(Start);

        Matrix<MatrixType, calculateLength<Start, Stop, Step>()> result;

        result.fill(
            LAMBDA(const int i, int) {
                return Start + i * Step;
            }
        );

        return result;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2, typename KokkosMathsFunction>
    Matrix<MatrixType, Dimension1, Dimension2> applyFunction(const Matrix<MatrixType, Dimension1, Dimension2>& matrix, const KokkosMathsFunction function) {
        Matrix<MatrixType, Dimension1, Dimension2> result;
        auto mCopy = matrix.getM();

        result.fill(
            LAMBDA(const int i, const int j) {
                return function(mCopy(i, j));
            }
        );

        return result;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> exp(const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        return applyFunction(
            matrix,
            LAMBDA(const MatrixType value) {
                return Kokkos::exp(value);
            }
        );
    }

    // add more functions

}

#endif
