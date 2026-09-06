#ifndef STOCK_ANALYSIS_ELEMENTARYOPERATIONS_TPP
#define STOCK_ANALYSIS_ELEMENTARYOPERATIONS_TPP

#include "ElementaryOperations.hpp"

namespace LinearAlgebra {
    template <
        typename MatrixType,
        ValidMathOperation Operation,
        std::size_t Dimension1,
        std::size_t Dimension2,
        typename OperandType1,
        typename OperandType2
    >
    Matrix<MatrixType, Dimension1, Dimension2> applyOperation(const OperandType1& m1, const OperandType2& m2) {
        const OperationApplier<MatrixType, Operation, Dimension1, Dimension2, OperandType1, OperandType2> operationApplier{};
        Matrix<MatrixType, Dimension1, Dimension2> result;

        Kokkos::parallel_for(
            "LinearAlgebra::Matrix::addMatrices::parallel_for",
            Kokkos::MDRangePolicy(
                {0, 0},
                {Dimension1, Dimension2}
            ),
            KOKKOS_LAMBDA(const int i, const int j) {
                result.view(i, j) = operationApplier(m1, m2, i, j);
            }
        );

        return result;
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Add, Dim1, Dim2>(m1, m2);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar) {
        return applyOperation<MatrixType, ValidMathOperation::Add, Dim1, Dim2>(m1, scalar);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Add, Dim1, Dim2>(scalar, m2);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Subtract, Dim1, Dim2>(m1, m2);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar) {
        return applyOperation<MatrixType, ValidMathOperation::Subtract, Dim1, Dim2>(m1, scalar);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Subtract, Dim1, Dim2>(scalar, m2);
    }

    template <typename MatrixType, std::size_t M1Dim1, std::size_t M1Dim2, std::size_t M2Dim1, std::size_t M2Dim2>
    Matrix<MatrixType, M1Dim1, M2Dim2> operator*(const Matrix<MatrixType, M1Dim1, M1Dim2>& m1, const Matrix<MatrixType, M2Dim1, M2Dim2>& m2) {
        Matrix<MatrixType, M1Dim1, M2Dim2> result;

        Kokkos::parallel_for(
        "LinearAlgebra::Matrix::operator*::parallel_for",
            Kokkos::TeamPolicy(M1Dim1 * M2Dim2, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const std::size_t index = team.league_rank();

                const std::size_t rowIndex = index / M2Dim2;
                const std::size_t colIndex = index % M2Dim2;

                MatrixType sum = 0;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, M1Dim2),
                    [&](const std::size_t i, MatrixType& localSum) {
                        localSum += m1.view(rowIndex, i) * m2.view(i, colIndex);
                    },
                    sum
                );

                Kokkos::single(
                    Kokkos::PerTeam(team),
                    [&] {
                        result.view(rowIndex, colIndex) = sum;
                    }
                );
            }
        );

        return result;
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator*(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar) {
        return applyOperation<MatrixType, ValidMathOperation::Multiply, Dim1, Dim2>(m1, scalar);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator*(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Multiply, Dim1, Dim2>(scalar, m2);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Divide, Dim1, Dim2>(m1, m2);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar) {
        return applyOperation<MatrixType, ValidMathOperation::Divide, Dim1, Dim2>(m1, scalar);
    }

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2) {
        return applyOperation<MatrixType, ValidMathOperation::Divide, Dim1, Dim2>(scalar, m2);
    }
}

#endif