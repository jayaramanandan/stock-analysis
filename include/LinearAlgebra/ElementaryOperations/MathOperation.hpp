#ifndef STOCK_ANALYSIS_MATHOPERATION_HPP
#define STOCK_ANALYSIS_MATHOPERATION_HPP

namespace LinearAlgebra {
    enum class ValidMathOperation {
        Add,
        Subtract,
        Multiply,
        Divide
    };

    template<typename MatrixType, ValidMathOperation Operation>
    struct MathOperation {
        KOKKOS_FUNCTION
        MatrixType operator()(const MatrixType a, const MatrixType b) const {
            if constexpr (Operation == ValidMathOperation::Add) {
                return a + b;
            } else if constexpr (Operation == ValidMathOperation::Subtract) {
                return a - b;
            } else if constexpr (Operation == ValidMathOperation::Multiply) {
                return a * b;
            } else if constexpr (Operation == ValidMathOperation::Divide) {
                return a / b;
            }

            Kokkos::abort("Not a valid math operation.");
        }
    };

    template<
        typename MatrixType,
        ValidMathOperation Operation,
        std::size_t Dimension1,
        std::size_t Dimension2,
        typename OperandType1,
        typename OperandType2
    >
    struct OperationApplier {
        using CleanOperandType1 = std::remove_cvref_t<OperandType1>;
        using CleanOperandType2 = std::remove_cvref_t<OperandType2>;

        const MathOperation<MatrixType, Operation> operation;

        KOKKOS_FUNCTION
        MatrixType operator()(const OperandType1& m1, const OperandType2& m2, const int i, const int j) const requires (MAX_DIMENSIONS == 2){
            if constexpr (std::is_same_v<CleanOperandType1, Matrix<MatrixType, Dimension1, Dimension2>> && std::is_same_v<CleanOperandType2, Matrix<MatrixType, Dimension1, Dimension2>>) {
                return operation(m1.view(i, j), m2.view(i, j));
            } else if constexpr (std::is_same_v<CleanOperandType1, Matrix<MatrixType, Dimension1, Dimension2>>) {
                return operation(m1.view(i, j), m2);
            } else if constexpr (std::is_same_v<CleanOperandType2, Matrix<MatrixType, Dimension1, Dimension2>>) {
                return operation(m1, m2.view(i, j));
            }
        }
    };
}

#endif
