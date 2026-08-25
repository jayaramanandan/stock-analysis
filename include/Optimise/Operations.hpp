#ifndef STOCK_ANALYSIS_OPERATIONS_HPP
#define STOCK_ANALYSIS_OPERATIONS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    template <typename ValueType, auto V>
    struct Constant {
        using constant = void;
        static constexpr ValueType value = V;

        constexpr ValueType operator()(ValueType) const {
            return value;
        }
    };

    template <typename ValueType, std::size_t Index>
    struct Variable {
        using variable = void;
        static constexpr auto index = Index;

        constexpr ValueType operator()(const ValueType x) const {
            return x;
        }
    };

    template <typename ValueType, std::size_t P, typename Num>
    struct Power {
        using power = void;
        static constexpr auto p = P;
        using num = Num;

        ValueType operator()(const ValueType x) const {
            const Num number{};
            return Kokkos::pow(number(x), p);
        }
    };

    template <typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2>
    struct Operation {
        using operation = Op;
        using num1 = Num1;
        using num2 = Num2;

        ValueType operator()(const ValueType x) const {
            const auto operation = LinearAlgebra::MathOperation<ValueType, Op>{};
            const Num1 x1{};
            const Num2 x2{};

            return operation(x1(x), x2(x));
        }
    };

    template <typename ValueType, std::size_t VariableIndex, typename X>
    struct Derivative {
    private:
        static auto getDerivative() {
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

    public:
        using derivative = decltype(getDerivative());

        ValueType operator()(const ValueType x) const {
            return derivative{}(x);
        }
    };

}

#endif
