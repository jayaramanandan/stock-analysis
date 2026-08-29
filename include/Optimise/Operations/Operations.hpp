#ifndef STOCK_ANALYSIS_OPERATIONS_HPP
#define STOCK_ANALYSIS_OPERATIONS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    template <typename ValueType, auto V>
    struct Constant {
        using constant = void;
        static constexpr ValueType value = V;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(ValueType) const;
    };

    template <typename ValueType, std::size_t Index>
    struct Variable {
        using variable = void;
        static constexpr auto index = Index;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(ValueType x) const;
    };

    template <typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2>
    struct Operation {
        using operation = Op;
        using num1 = Num1;
        using num2 = Num2;

        KOKKOS_FUNCTION
        ValueType operator()(ValueType x) const;
    };

    // Functions

    template <typename ValueType, std::size_t P, typename Num>
    struct Power {
        using power = void;
        static constexpr auto p = P;
        using num = Num;

        KOKKOS_FUNCTION
        ValueType operator()(ValueType x) const;
    };

    template <typename ValueType, std::size_t VariableIndex, typename X>
    struct Derivative {
    private:
        KOKKOS_FUNCTION
        static auto getDerivative();

    public:
        using derivative = decltype(getDerivative());

        KOKKOS_FUNCTION
        ValueType operator()(ValueType x) const;
    };

    // Elementary symbolic operations

    template <LinearAlgebra::ValidMathOperation Op, typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto generalisedElementaryOperation(Num1Type x, Num2Type y);

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator+(Num1Type x, Num2Type y);

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator-(Num1Type x, Num2Type y);

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator*(Num1Type x, Num2Type y);

    template <typename Num1Type, typename Num2Type>
    KOKKOS_FUNCTION
    constexpr auto operator/(Num1Type x, Num2Type y);

    template <std::size_t P, typename Num1Type>
    KOKKOS_FUNCTION
    constexpr auto power(Num1Type x);
}

#include "Operations.tpp"

#endif
