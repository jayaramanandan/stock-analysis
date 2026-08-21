#ifndef STOCK_ANALYSIS_MATRIX_TPP
#define STOCK_ANALYSIS_MATRIX_TPP

#include "Matrix.hpp"

#include <array>
#include <type_traits>

#include "../Macros.hpp"
#include "LinearAlgebra/MathOperation.hpp"

namespace LinearAlgebra {
    template <
        std::size_t Dimensions,
        typename MatrixType,
        ValidMathOperation Operation,
        typename OperandType1,
        typename OperandType2
    >
    KokkosView<Dimensions, MatrixType> applyMathsOperation(OperandType1 op1, OperandType2 op2) {
        OperationApplier<Dimensions, MatrixType, Operation, OperandType1, OperandType2> operationApplier;
        KokkosView<Dimensions, MatrixType> matrix;

        if constexpr (std::is_same_v<std::remove_cvref_t<OperandType1>, KokkosView<Dimensions, MatrixType>>) {
            matrix = op1;
        } else {
            matrix = op2;
        }

        if constexpr (Dimensions == 1) {
            KokkosView<Dimensions, MatrixType> result("LinearAlgebra::Matrix::applyMathsOperation::result", matrix.extent(0));

            Kokkos::parallel_for(
                "LinearAlgebra::Matrix::iterateElements",
                matrix.extent(0),
                LAMBDA(const int i) {
                    result(i) = operationApplier(op1, op2, i);
                }
            );

            return result;
        } else {
            KokkosView<Dimensions, MatrixType> result("LinearAlgebra::Matrix::applyMathsOperation::result", matrix.extent(0), matrix.extent(1));

            Kokkos::parallel_for(
                "LinearAlgebra::Matrix::iterateElements",
                Kokkos::MDRangePolicy<Kokkos::Rank<Dimensions>>(
                    {0, 0},
                    {matrix.extent(0), matrix.extent(1)}
                ),
                LAMBDA(const int i, const int j) {
                    result(i, j) = operationApplier(op1, op2, i, j);
                }
            );

            return result;
        }
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType>::Matrix(KokkosView<Dimensions, MatrixType> matrixView): m(matrixView) {
        if constexpr (Dimensions == 1) {
            this->shape[0] = matrixView.extent(0);
        } else {
            this->shape[0] = matrixView.extent(0);
            this->shape[1] = matrixView.extent(1);
        }
    }

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
        auto hostM = Kokkos::create_mirror_view(this->m);
        Kokkos::deep_copy(hostM, this->m);

        std::string outputString = "[\n";

        for (std::size_t i = 0; i < this->shape[0]; i++) {
            outputString += '\t';

            if constexpr (Dimensions == 1) {
                outputString += std::to_string(hostM(i));

                if (i != this->shape[0] - 1) {
                    outputString += ", ";
                }
            } else {
                for (std::size_t j = 0; j < this->shape[1]; j++) {
                    outputString += std::to_string(hostM(i, j));

                    if (i != this->shape[0] - 1 || j != this->shape[1] - 1) {
                        outputString += ", ";
                    }
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
        auto mCopy = this->m;

        if constexpr (Dimensions == 1) {
            Kokkos::parallel_for(
                "LinearAlgebra::Matrix::iterateElements",
                shapeCopy[0],
                kokkosCallback
            );
        } else {
            Kokkos::parallel_for(
                "LinearAlgebra::Matrix::iterateElements",
                Kokkos::MDRangePolicy<Kokkos::Rank<Dimensions>>(
                    {0, 0},
                    {shapeCopy[0], shapeCopy[1]}
                ),
                kokkosCallback
            );
        }
    }

    template<std::size_t Dimensions, typename MatrixType>
    template<typename KokkosFunction>
    void Matrix<Dimensions, MatrixType>::fill(const KokkosFunction fillFunction) {
        auto mCopy = this->m;

        if constexpr (Dimensions == 1) {
            this->iterateElements(
                LAMBDA(const int i) {
                    mCopy(i) = fillFunction(i);
                }
            );
        } else {
            this->iterateElements(
                LAMBDA(const int i, const int j) {
                    mCopy(i, j) = fillFunction(i, j);
                }
            );
        }
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator+(const Matrix& otherMatrix) const {
        if (this->shape != otherMatrix.shape) {
            throw std::runtime_error("The shape of both matrices should be the same.");
        }

        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Add>(this->m, otherMatrix.getM())
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator+(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Add>(this->m, scalar)
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator-(const Matrix& otherMatrix) const {
        if (this->shape != otherMatrix.shape) {
            throw std::runtime_error("The shape of both matrices should be the same.");
        }

        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Subtract>(this->m, otherMatrix.getM())
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator-(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Subtract>(this->m, scalar)
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator*(MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Multiply>(this->m, scalar)
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    template<std::size_t OtherDimensions>
    auto Matrix<Dimensions, MatrixType>::operator*(const Matrix<OtherDimensions, MatrixType>& otherMatrix) const {}

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator/(const Matrix<1, MatrixType>& otherMatrix) const {
        static_assert(Dimensions == 1, "The dimensions of the matrix should be 1");

        if (this->shape != otherMatrix.shape) {
            throw std::runtime_error("The shape of both matrices should be the same.");
        }

        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Divide>(this->m, otherMatrix.getM())
        );
    }

    template<std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> Matrix<Dimensions, MatrixType>::operator/(const MatrixType scalar) const {
        return Matrix(
            applyMathsOperation<Dimensions, MatrixType, Divide>(this->m, scalar)
        );
    }

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator+(const MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix) {
        return matrix + scalar;
    }

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator-(const MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix) {
        return Matrix<Dimensions, MatrixType>(
            applyMathsOperation<Dimensions, MatrixType, Subtract>(scalar, matrix.getM())
        );
    }

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator*(const MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix) {
        return matrix * scalar;
    }

    template <std::size_t Dimensions, typename MatrixType>
    Matrix<Dimensions, MatrixType> operator/(const MatrixType scalar, const Matrix<Dimensions, MatrixType>& matrix) {
        return Matrix<Dimensions, MatrixType>(
            applyMathsOperation<Dimensions, MatrixType, Divide>(scalar, matrix.getM())
        );
    }
}

#endif
