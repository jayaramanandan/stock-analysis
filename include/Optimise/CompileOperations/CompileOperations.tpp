#ifndef STOCK_ANALYSIS_COMPILEOPERATIONS_TPP
#define STOCK_ANALYSIS_COMPILEOPERATIONS_TPP

#include "CompileOperations.hpp"

namespace Optimise {
    template<typename ValueType, auto V, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    constexpr ValueType Constant<ValueType, V, ParametersLength>::operator()(const LinearAlgebra::KokkosView<ValueType>) const {
        return value;
    }

    template<typename ValueType, std::size_t Index, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    constexpr ValueType Variable<ValueType, Index, ParametersLength>::operator()(const LinearAlgebra::KokkosView<ValueType> x) const {
        return x(Index, 0);
    }

    template<typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    ValueType Operation<ValueType, Op, Num1, Num2, ParametersLength>::operator()(const LinearAlgebra::KokkosView<ValueType> x) const {
        const Num1 x1{};
        const Num2 x2{};

        return LinearAlgebra::MathOperation<ValueType, Op>{}(x1(x), x2(x));
    }

    // Functions

    template<typename ValueType, std::size_t P, typename Num, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    ValueType Power<ValueType, P, Num, ParametersLength>::operator()(const LinearAlgebra::KokkosView<ValueType> x) const {
        return Kokkos::pow(Num{}(x), p);
    }

    template<typename ValueType, std::size_t VariableIndex, typename X, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    constexpr auto Derivative<ValueType, VariableIndex, X, ParametersLength>::getDerivative() {
        if constexpr (requires { typename X::variable; }) {
            if constexpr (X::index == VariableIndex) {
                return Constant<ValueType, static_cast<ValueType>(1), ParametersLength>{};
            } else {
                return Constant<ValueType, static_cast<ValueType>(0), ParametersLength>{};
            }
        } else if constexpr (requires { typename X::constant; }) {
            return Constant<ValueType, static_cast<ValueType>(0), ParametersLength>{};
        } else if constexpr (requires { typename X::operation; }) {
            if constexpr (X::op == LinearAlgebra::ValidMathOperation::Add || X::op == LinearAlgebra::ValidMathOperation::Subtract) {
                return Operation<
                    ValueType,
                    X::op,
                    typename Derivative<ValueType, VariableIndex, typename X::num1, ParametersLength>::derivative,
                    typename Derivative<ValueType, VariableIndex, typename X::num2, ParametersLength>::derivative,
                    ParametersLength
                >{};
            } else if constexpr (X::op == LinearAlgebra::ValidMathOperation::Multiply) {
                return Operation<
                    ValueType,
                    LinearAlgebra::ValidMathOperation::Add,
                    Operation<
                        ValueType,
                        LinearAlgebra::ValidMathOperation::Multiply,
                        typename X::num1,
                        typename Derivative<ValueType, VariableIndex, typename X::num2, ParametersLength>::derivative,
                        ParametersLength
                    >,
                    Operation<
                        ValueType,
                        LinearAlgebra::ValidMathOperation::Multiply,
                        typename X::num2,
                        typename Derivative<ValueType, VariableIndex, typename X::num1, ParametersLength>::derivative,
                        ParametersLength
                    >,
                    ParametersLength
                >{};
            } else if constexpr (X::op == LinearAlgebra::ValidMathOperation::Divide) {
                return Operation<
                    ValueType,
                    LinearAlgebra::ValidMathOperation::Divide,
                    Operation<
                        ValueType,
                        LinearAlgebra::ValidMathOperation::Subtract,
                        Operation<
                            ValueType,
                            LinearAlgebra::ValidMathOperation::Multiply,
                            typename X::num2,
                            typename Derivative<ValueType, VariableIndex, typename X::num1, ParametersLength>::derivative,
                            ParametersLength
                        >,
                        Operation<
                            ValueType,
                            LinearAlgebra::ValidMathOperation::Multiply,
                            typename X::num1,
                            typename Derivative<ValueType, VariableIndex, typename X::num2, ParametersLength>::derivative,
                            ParametersLength
                        >,
                        ParametersLength
                    >,
                    Power<ValueType, 2, typename X::num2, ParametersLength>,
                    ParametersLength
                >{};
            }
        } else if constexpr (requires { typename X::power; }) {
            return Operation<
                ValueType,
                LinearAlgebra::ValidMathOperation::Multiply,
                Constant<ValueType, X::p, ParametersLength>,
                Operation<
                    ValueType,
                    LinearAlgebra::ValidMathOperation::Multiply,
                    typename Derivative<ValueType, VariableIndex, typename X::num, ParametersLength>::derivative,
                    Power<ValueType, X::p - 1, typename X::num, ParametersLength>,
                    ParametersLength
                >,
                ParametersLength
            >{};
        } else {
            Kokkos::abort("Invalid operation");
        }
    }

    template<typename ValueType, std::size_t VariableIndex, typename X, std::size_t ParametersLength>
    KOKKOS_FUNCTION
    ValueType Derivative<ValueType, VariableIndex, X, ParametersLength>::operator()(const LinearAlgebra::KokkosView<ValueType> x) const {
        return derivative{}(x);
    }
}

#endif
