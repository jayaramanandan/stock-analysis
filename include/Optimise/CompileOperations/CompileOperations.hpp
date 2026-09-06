#ifndef STOCK_ANALYSIS_COMPILEOPERATIONS_HPP
#define STOCK_ANALYSIS_COMPILEOPERATIONS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    struct Expression {};

    template <typename ValueType, auto V, std::size_t ParametersLength>
    struct Constant: Expression {
        using constant = void;
        static constexpr ValueType value = V;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(LinearAlgebra::KokkosView<ValueType>) const;
    };

    template <typename ValueType, std::size_t Index, std::size_t ParametersLength>
    struct Variable: Expression {
        using variable = void;
        static constexpr auto index = Index;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(LinearAlgebra::KokkosView<ValueType> x) const;
    };

    template <typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2, std::size_t ParametersLength>
    struct Operation: Expression {
        using operation = void;

        static constexpr auto op = Op;
        using num1 = Num1;
        using num2 = Num2;

        KOKKOS_FUNCTION
        ValueType operator()(LinearAlgebra::KokkosView<ValueType> x) const;
    };

    // Functions

    template <typename ValueType, std::size_t P, typename Num, std::size_t ParametersLength>
    struct Power: Expression {
        using power = void;
        static constexpr auto p = P;
        using num = Num;

        KOKKOS_FUNCTION
        ValueType operator()(LinearAlgebra::KokkosView<ValueType> x) const;
    };

    template <typename ValueType, std::size_t VariableIndex, typename X, std::size_t ParametersLength>
    struct Derivative: Expression {
    private:
        KOKKOS_FUNCTION
        static constexpr auto getDerivative();

    public:
        using derivative = std::remove_cvref_t<decltype(getDerivative())>;

        KOKKOS_FUNCTION
        ValueType operator()(LinearAlgebra::KokkosView<ValueType> x) const;
    };
}

#include "CompileOperations.tpp"

#endif
