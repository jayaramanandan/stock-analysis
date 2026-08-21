#ifndef STOCK_ANALYSIS_MATHOPERATION_HPP
#define STOCK_ANALYSIS_MATHOPERATION_HPP

#include <type_traits>

namespace LinearAlgebra {
    struct Add {};
    struct Subtract {};
    struct Multiply {};
    struct Divide {};

    template<typename Operation>
    concept ValidMathOperation =
    std::is_same_v<Operation, Add> ||
    std::is_same_v<Operation, Subtract> ||
    std::is_same_v<Operation, Multiply> ||
    std::is_same_v<Operation, Divide>;

    template<typename MatrixType, ValidMathOperation Operation>
    struct MathOperation {
        KOKKOS_FUNCTION
        MatrixType operator()(MatrixType a, MatrixType b) const {
            if constexpr (std::is_same_v<Operation, Add>) {
                return a + b;
            } else if constexpr (std::is_same_v<Operation, Subtract>) {
                return a - b;
            } else if constexpr (std::is_same_v<Operation, Multiply>) {
                return a * b;
            } else if constexpr (std::is_same_v<Operation, Divide>) {
                return a / b;
            }
        }
    };

    template<
        std::size_t Dimensions,
        typename MatrixType,
        ValidMathOperation Operation,
        typename OperandType1,
        typename OperandType2
    >
    struct OperationApplier {
        using CleanOperandType1 = std::remove_cvref_t<OperandType1>;
        using CleanOperandType2 = std::remove_cvref_t<OperandType2>;

        //static_assert(!std::is_same_v<CleanOperandType1, MatrixType> && !std::is_same_v<CleanOperandType2, MatrixType>, "Both operands cannot be scalars.");

        KOKKOS_FUNCTION
        MatrixType operator()(OperandType1 op1, OperandType2 op2, const int i) const requires (Dimensions == 1) {
            MathOperation<MatrixType, Operation> operation;

            if constexpr (std::is_same_v<CleanOperandType1, MatrixType>) {
                return operation(op1, op2(i));
            } else if constexpr (std::is_same_v<CleanOperandType2, MatrixType>) {
                return operation(op1(i), op2);
            } else {
                return operation(op1(i), op2(i));
            }
        }

        KOKKOS_FUNCTION
        MatrixType operator()(OperandType1 op1, OperandType2 op2, const int i, const int j) const requires (Dimensions == 2) {
            MathOperation<MatrixType, Operation> operation;

            if constexpr (std::is_same_v<CleanOperandType1, MatrixType>) {
                return operation(op1, op2(i, j));
            } else if constexpr (std::is_same_v<CleanOperandType2, MatrixType>) {
                return operation(op1(i, j), op2);
            } else {
                return operation(op1(i, j), op2(i, j));
            }
        }
    };
}

#endif
