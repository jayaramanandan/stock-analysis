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

    template<typename VectorType, ValidMathOperation Operation>
    struct MathOperation {
        KOKKOS_FUNCTION
        VectorType operator()(VectorType a, VectorType b) const {
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
}

#endif
