#ifndef STOCK_ANALYSIS_MATRIX_HPP
#define STOCK_ANALYSIS_MATRIX_HPP

#include <array>
#include <string>

#include <Kokkos_Core.hpp>

#include "MathOperation.hpp"
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

    template <typename T, std::size_t N>
    using AddPointers_t = AddPointers<T, N>::type;

    template <typename MatrixType>
    using KokkosView = Kokkos::View<AddPointers_t<MatrixType, MAX_DIMENSIONS>>;

    template <
        typename MatrixType,
        ValidMathOperation Operation,
        typename OperandType1,
        typename OperandType2
    >
    KokkosView<MatrixType> applyMathsOperation(OperandType1 op1, OperandType2 op2);

    template <typename MatrixType, std::size_t Dimension1, std::size_t Dimension2 = 1>
    class Matrix {
        using MatrixTypePointer = AddPointers_t<MatrixType, MAX_DIMENSIONS>;
        
        KokkosView<MatrixType> m;

        public:
        explicit Matrix(KokkosView<MatrixType> matrixView);

        explicit Matrix();

        template <typename KokkosFunction>
        explicit Matrix(KokkosFunction fillFunction);

        Matrix(std::array<MatrixType, Dimension1> matrixArray) requires (Dimension2 == 1);

        Matrix(std::array<std::array<MatrixType, Dimension2>, Dimension1> matrixArray) requires (Dimension2 != 1);

        KokkosView<MatrixType> getM() const;

        [[nodiscard]] std::string toString() const;

        template <typename KokkosFunction>
        void iterateElements(KokkosFunction kokkosCallback) const;

        template <typename KokkosFunction>
        void fill(KokkosFunction fillFunction);

        KOKKOS_FUNCTION
        MatrixType operator()(std::size_t i) const requires (Dimension2 == 1);

        KOKKOS_FUNCTION
        MatrixType operator()(std::size_t i, std::size_t j) const requires (Dimension2 != 1);

        Matrix operator+(const Matrix& otherMatrix) const;

        Matrix operator+(MatrixType scalar) const;

        Matrix operator-(const Matrix& otherMatrix) const;

        Matrix operator-(MatrixType scalar) const;

        template <std::size_t OtherDimension2>
        Matrix<MatrixType, Dimension1, OtherDimension2> operator*(const Matrix<MatrixType, Dimension2, OtherDimension2>& otherMatrix) const;

        Matrix operator*(MatrixType scalar) const;

        Matrix operator/(const Matrix<MatrixType, Dimension1>& otherMatrix) const;

        Matrix operator/(MatrixType scalar) const;
    };

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator+(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix);

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator-(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix);

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator*(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix);

    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2> operator/(MatrixType scalar, const Matrix<MatrixType, Dimension1, Dimension2>& matrix);

}

#include "Matrix.tpp"

#endif
