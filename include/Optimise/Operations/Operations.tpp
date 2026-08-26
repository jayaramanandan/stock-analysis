#ifndef STOCK_ANALYSIS_OPERATIONS_TPP
#define STOCK_ANALYSIS_OPERATIONS_TPP

#include "Operations.hpp"

namespace Optimise {
    template<typename ValueType, auto V>
    KOKKOS_FUNCTION
    constexpr ValueType Constant<ValueType, V>::operator()(ValueType) const {
        return value;
    }

    template<typename ValueType, std::size_t Index>
    KOKKOS_FUNCTION
    constexpr ValueType Variable<ValueType, Index>::operator()(const ValueType x) const {
        return x;
    }

    template<typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2>
    KOKKOS_FUNCTION
    ValueType Operation<ValueType, Op, Num1, Num2>::operator()(const ValueType x) const {
        const Num1 x1{};
        const Num2 x2{};

        return LinearAlgebra::MathOperation<ValueType, Op>{}(x1(x), x2(x));
    }

    // Functions

    template<typename ValueType, std::size_t P, typename Num>
    KOKKOS_FUNCTION
    ValueType Power<ValueType, P, Num>::operator()(const ValueType x) const {
        return Kokkos::pow(Num{}(x), p);
    }

    template<typename ValueType, std::size_t VariableIndex, typename X>
    KOKKOS_FUNCTION
    auto Derivative<ValueType, VariableIndex, X>::getDerivative() {
        if constexpr (requires { typename X::variable; }) {
            if constexpr (X::index == VariableIndex) {
                return Constant<ValueType, static_cast<ValueType>(1)>{};
            } else {
                return Constant<ValueType, static_cast<ValueType>(0)>{};
            }
        } else if constexpr (requires { typename X::constant; }) {
            return Constant<ValueType, static_cast<ValueType>(0)>{};
        } else if constexpr (requires { typename X::operation; }) {
            if constexpr (std::is_same_v<typename X::operation, LinearAlgebra::Add> || std::is_same_v<typename X::operation, LinearAlgebra::Subtract>) {
                return Operation<
                    ValueType,
                    typename X::operation,
                    typename Derivative<ValueType, VariableIndex, typename X::num1>::derivative,
                    typename Derivative<ValueType, VariableIndex, typename X::num2>::derivative
                >{};
            } else if constexpr (std::is_same_v<typename X::operation, LinearAlgebra::Multiply>) {
                return Operation<
                    ValueType,
                    LinearAlgebra::Add,
                    Operation<
                        ValueType,
                        LinearAlgebra::Multiply,
                        typename X::num1,
                        typename Derivative<ValueType, VariableIndex, typename X::num2>::derivative
                    >,
                    Operation<
                        ValueType,
                        LinearAlgebra::Multiply,
                        typename X::num2,
                        typename Derivative<ValueType, VariableIndex, typename X::num1>::derivative
                    >
                >{};
            } else if constexpr (std::is_same_v<typename X::operation, LinearAlgebra::Divide>) {
                return Operation<
                    ValueType,
                    LinearAlgebra::Divide,
                    Operation<
                        ValueType,
                        LinearAlgebra::Subtract,
                        Operation<
                            ValueType,
                            LinearAlgebra::Multiply,
                            typename X::num2,
                            typename Derivative<ValueType, VariableIndex, typename X::num1>::derivative
                        >,
                        Operation<
                            ValueType,
                            LinearAlgebra::Multiply,
                            typename X::num1,
                            typename Derivative<ValueType, VariableIndex, typename X::num2>::derivative
                        >
                    >,
                    Power<ValueType, 2, typename X::num2>
                >{};
            }
        } else if constexpr (requires { typename X::power; }) {
            return Operation<
                ValueType,
                LinearAlgebra::Multiply,
                Constant<ValueType, X::p>,
                Operation<
                    ValueType,
                    LinearAlgebra::Multiply,
                    typename Derivative<ValueType, VariableIndex, typename X::num>::derivative,
                    Power<ValueType, X::p - 1, typename X::num>
                >
            >{};
        } else {
            throw std::invalid_argument("Invalid Function in Derivative");
        }
    }

    template<typename ValueType, std::size_t VariableIndex, typename X>
    KOKKOS_FUNCTION
    ValueType Derivative<ValueType, VariableIndex, X>::operator()(const ValueType x) const {
        return derivative{}(x);
    }

    // Elementary symbolic operations

    template <LinearAlgebra::ValidMathOperation Op, typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto generalisedElementaryOperation(const Num1Type x, const Num1Type y) {
        if constexpr (std::is_arithmetic_v<Num1Type>) {
            return Operation<Num1Type, Op, Constant<Num1Type, x>, Num2Type>{}(x, y);
        } else if constexpr (std::is_arithmetic_v<Num2Type>) {
            return Operation<Num1Type, Op, Num1Type, Constant<Num2Type, x>>{}(x, y);
        } else {
            return Operation<Num1Type, Op, Num1Type, Num2Type>{}(x, y);
        }
    }

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator+(const Num1Type x, const Num2Type y) {
        return generalisedElementaryOperation<LinearAlgebra::Add>(x, y);
    }

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator-(const Num1Type x, const Num2Type y) {
        return generalisedElementaryOperation<LinearAlgebra::Subtract>(x, y);
    }

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator*(const Num1Type x, const Num2Type y) {
        return generalisedElementaryOperation<LinearAlgebra::Multiply>(x, y);
    }

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator/(const Num1Type x, const Num2Type y) {
        return generalisedElementaryOperation<LinearAlgebra::Divide>(x, y);
    }

    template <std::size_t P, typename Num1Type>
    KOKKOS_FUNCTION
    constexpr auto power(const Num1Type x) {
        if (std::is_arithmetic_v<Num1Type>) {
            return Power<Num1Type, P, Constant<Num1Type, x>>{}(x);
        }
    }
}

#endif
