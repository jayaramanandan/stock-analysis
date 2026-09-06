#ifndef STOCK_ANALYSIS_ELEMENTARYOPERATIONS_HPP
#define STOCK_ANALYSIS_ELEMENTARYOPERATIONS_HPP

#include "MathOperation.hpp"
#include "../Matrix/Matrix.hpp"

namespace LinearAlgebra {
    template <
        typename MatrixType,
        ValidMathOperation Operation,
        std::size_t Dimension1,
        std::size_t Dimension2,
        typename OperandType1,
        typename OperandType2
    >
    Matrix<MatrixType, Dimension1, Dimension2> applyOperation(const OperandType1& m1, const OperandType2& m2);

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator+(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2);

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator-(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2);

    template <typename MatrixType, std::size_t M1Dim1, std::size_t M1Dim2, std::size_t M2Dim1, std::size_t M2Dim2>
    Matrix<MatrixType, M1Dim1, M2Dim2> operator*(const Matrix<MatrixType, M1Dim1, M1Dim2>& m1, const Matrix<MatrixType, M2Dim1, M2Dim2>& m2);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator*(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator*(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2);

    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(const Matrix<MatrixType, Dim1, Dim2>& m1, const Matrix<MatrixType, Dim1, Dim2>& m2);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(const Matrix<MatrixType, Dim1, Dim2>& m1, MatrixType scalar);
    template <typename MatrixType, std::size_t Dim1, std::size_t Dim2>
    Matrix<MatrixType, Dim1, Dim2> operator/(MatrixType scalar, const Matrix<MatrixType, Dim1, Dim2>& m2);
}

#include "ElementaryOperations.tpp"

#endif
