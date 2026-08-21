#ifndef STOCK_ANALYSIS_MATRIX_HPP
#define STOCK_ANALYSIS_MATRIX_HPP

#include <array>
#include <string>

#include <Kokkos_Core.hpp>

#include "LinearAlgebra/MathOperation.hpp"

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

    template <std::size_t Dimensions, typename MatrixType>
    using KokkosView = Kokkos::View<AddPointers_t<MatrixType, Dimensions>>;

    template <
        std::size_t Dimensions,
        typename MatrixType,
        ValidMathOperation Operation,
        typename OperandType1,
        typename OperandType2
    >
    KokkosView<Dimensions, MatrixType> applyMathsOperation(OperandType1 op1, OperandType2 op2);

    template <std::size_t Dimensions, typename MatrixType>
    class Matrix {
        static_assert(
            Dimensions == 1 || Dimensions == 2,
            "Dimensions must be 1 or 2."
        );

        using MatrixTypePointer = AddPointers_t<MatrixType, Dimensions>;
        
        KokkosView<Dimensions, MatrixType> m;
        std::array<std::size_t, Dimensions> shape;

        public:
        explicit Matrix(KokkosView<Dimensions, MatrixType> matrixView);

        template<typename... Shape>
        explicit Matrix(Shape... shape);

        KokkosView<Dimensions, MatrixType> getM() const;

        [[nodiscard]] std::string toString() const;

        template <typename KokkosFunction>
        void iterateElements(KokkosFunction kokkosCallback) const;

        template <typename KokkosFunction>
        void fill(KokkosFunction fillFunction);

        Matrix operator+(const Matrix& otherMatrix) const;

        Matrix operator+(MatrixType scalar) const;

        Matrix operator-(const Matrix& otherMatrix) const;

        Matrix operator-(MatrixType scalar) const;

        //Matrix operator*(const Matrix& otherMatrix) const;

        Matrix operator*(MatrixType scalar) const;

        Matrix operator/(const Matrix<1, MatrixType>& otherMatrix) const;

        Matrix operator/(MatrixType scalar) const;
    };

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator+(MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix);

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator-(MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix);

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator*(MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix);

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator/(MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix);
}

#include "Matrix.tpp"

#endif
