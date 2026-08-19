#ifndef STOCK_ANALYSIS_MATRIX_HPP
#define STOCK_ANALYSIS_MATRIX_HPP

#include <array>
#include <string>

#include <Kokkos_Core.hpp>

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
    class Matrix {
        using MatrixTypePointer = AddPointers_t<MatrixType, Dimensions>;
        
        Kokkos::View<MatrixTypePointer> m;
        std::array<std::size_t, Dimensions> shape;

        template <typename KokkosFunction>
        void iterateElements(KokkosFunction kokkosCallback) const;

        public:
        explicit Matrix(Kokkos::View<MatrixTypePointer> matrixView);

        template<typename... Shape>
        explicit Matrix(Shape... shape);

        Kokkos::View<MatrixTypePointer> getM() const;

        [[nodiscard]] std::string toString() const;

        template <typename KokkosFunction>
        void fill(KokkosFunction fillFunction);

        /*
        [[nodiscard]] std::string to_string() const;

        template <typename Operation>
        Kokkos::View<MatrixTypePointer> elementWiseOperation(Kokkos::View<MatrixTypePointer> m1, Kokkos::View<MatrixTypePointer> m2, MathOperation<MatrixType, Operation> mathOperation) const;

        template <typename Operation>
        Kokkos::View<MatrixTypePointer> elementWiseOperation(Kokkos::View<MatrixTypePointer> m1, MatrixType scalar, MathOperation<MatrixType, Operation> mathOperation) const;

        template <typename Operation>
        Kokkos::View<MatrixTypePointer> elementWiseOperation(MatrixType scalar, Kokkos::View<MatrixTypePointer> m2, MathOperation<MatrixType, Operation> mathOperation) const;

        void fill(MatrixType value);

        template<typename FillFunctor>
        void fill();

        MatrixType operator()(std::size_t row, std::size_t col) const;

        Matrix operator+(const Matrix& otherMatrix) const;

        Matrix operator+(const MatrixType& scalar) const;

        Matrix operator-(const Matrix& otherMatrix) const;

        Matrix operator-(const MatrixType& scalar) const;

        Matrix operator*(const Matrix& otherMatrix) const;

        Matrix operator*(const MatrixType& scalar) const;

        Matrix operator/(const MatrixType& scalar) const;
        */
    };
}

#include "Matrix.tpp"

#endif
