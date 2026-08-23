#ifndef STOCK_ANALYSIS_MATRIX_TPP
#define STOCK_ANALYSIS_MATRIX_TPP

#include "Matrix.hpp"

#include <type_traits>

#include "../Macros.hpp"
#include "MathOperation.hpp"

namespace LinearAlgebra {
    template <
        typename MatrixType,
        ValidMathOperation Operation,
        typename OperandType1,
        typename OperandType2
    >
    KokkosView<MatrixType> applyMathsOperation(OperandType1 op1, OperandType2 op2) {
        OperationApplier<MatrixType, Operation, OperandType1, OperandType2> operationApplier;
        KokkosView<MatrixType> matrix;

        if constexpr (std::is_same_v<std::remove_cvref_t<OperandType1>, KokkosView<MatrixType>>) {
            matrix = op1;
        } else {
            matrix = op2;
        }

        KokkosView<MatrixType> result("LinearAlgebra::Matrix::applyMathsOperation::result", matrix.extent(0), matrix.extent(1));

        Kokkos::parallel_for(
            "LinearAlgebra::Matrix::applyMathsOperation",
            Kokkos::MDRangePolicy<Kokkos::Rank<MAX_DIMENSIONS>>(
                {0, 0},
                {matrix.extent(0), matrix.extent(1)}
            ),
            LAMBDA(const int i, const int j) {
                result(i, j) = operationApplier(op1, op2, i, j);
            }
        );

        return result;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2>::Matrix(KokkosView<MatrixType> matrixView): m(matrixView) {}

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2>::Matrix(): m("LinearAlgebra::Matrix::m", Dimension1, Dimension2) {}

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    KokkosView<MatrixType> Matrix<MatrixType, Dimension1, Dimension2>::getM() const {
        return this->m;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    std::string Matrix<MatrixType, Dimension1, Dimension2>::toString() const {
        auto hostM = Kokkos::create_mirror_view(this->m);
        Kokkos::deep_copy(hostM, this->m);

        std::string outputString = "[\n";

        for (std::size_t i = 0; i < Dimension1; i++) {
            outputString += '\t';

            for (std::size_t j = 0; j < Dimension2; j++) {
                outputString += std::to_string(hostM(i, j));

                if (i != Dimension1 - 1 || j != Dimension2 - 1) {
                    outputString += ", ";
                }
            }

            outputString += '\n';
        }

        outputString += "\n]";

        return outputString;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    template <typename KokkosFunction>
    void Matrix<MatrixType, Dimension1, Dimension2>::iterateElements(const KokkosFunction kokkosCallback) const {
        auto mCopy = this->m;

        Kokkos::parallel_for(
            "LinearAlgebra::Matrix::iterateElements",
            Kokkos::MDRangePolicy(
                {0, 0},
                {Dimension1, Dimension2}
            ),
            kokkosCallback
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    template<typename KokkosFunction>
    void Matrix<MatrixType, Dimension1, Dimension2>::fill(const KokkosFunction fillFunction) {
        auto mCopy = this->m;

        this->iterateElements(
            LAMBDA(const int i, const int j) {
                mCopy(i, j) = fillFunction(i, j);
            }
        );
    }


    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator+(const Matrix& otherMatrix) const {
        return Matrix(
            applyMathsOperation<MatrixType, Add>(this->m, otherMatrix.getM())
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator+(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<MatrixType, Add>(this->m, scalar)
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator-(const Matrix& otherMatrix) const {
        return Matrix(
            applyMathsOperation<MatrixType, Subtract>(this->m, otherMatrix.getM())
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator-(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<MatrixType, Subtract>(this->m, scalar)
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    template<std::size_t OtherDimension2>
    Matrix<MatrixType, Dimension1, OtherDimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator*(const Matrix<MatrixType, Dimension2, OtherDimension2>& otherMatrix) const {
        auto mCopy = this->m;
        auto otherM = otherMatrix.getM();

        KokkosView<MatrixType> result("LinearAlgebra::Matrix::operator*::result", Dimension1, OtherDimension2);

        Kokkos::parallel_for(
            "LinearAlgebra::Matrix::operator*::parallel_for",
            Kokkos::TeamPolicy(Dimension1 * OtherDimension2, Kokkos::AUTO),
            LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const std::size_t index = team.league_rank();

                const std::size_t rowIndex = index / OtherDimension2;
                const std::size_t colIndex = index % OtherDimension2;

                MatrixType sum = 0;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, Dimension2),
                    [&](const std::size_t i, MatrixType& localSum) {
                        localSum += mCopy(rowIndex, i) * otherM(i, colIndex);
                    },
                    sum
                );

                Kokkos::single(
                    Kokkos::PerTeam(team),
                    [&] {
                        result(rowIndex, colIndex) = sum;
                    }
                );
            }
        );

        return Matrix<MatrixType, Dimension1, OtherDimension2>(result);
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator*(MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<MatrixType, Multiply>(this->m, scalar)
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator/(const Matrix<MatrixType, Dimension1>& otherMatrix) const {
        return Matrix(
            applyMathsOperation<MatrixType, Divide>(this->m, otherMatrix.getM())
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> Matrix<MatrixType, Dimension1, Dimension2>::operator/(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<MatrixType, Divide>(this->m, scalar)
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator+(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        return matrix + scalar;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator-(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        return Matrix<MatrixType, Dimension1, Dimension2>(
            applyMathsOperation<MatrixType, Subtract>(scalar, matrix.getM())
        );
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator*(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        return matrix * scalar;
    }

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator/(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        return Matrix<MatrixType, Dimension1, Dimension2>(
            applyMathsOperation<MatrixType, Divide>(scalar, matrix.getM())
        );
    }
}

#endif
