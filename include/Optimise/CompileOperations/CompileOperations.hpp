#ifndef STOCK_ANALYSIS_COMPILEOPERATIONS_HPP
#define STOCK_ANALYSIS_COMPILEOPERATIONS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    template <typename ValueType, auto V, std::size_t ParametersLength>
    struct Constant {
        using constant = void;
        static constexpr ValueType value = V;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(const LinearAlgebra::Matrix<ValueType, ParametersLength>&) const;
    };

    template <typename ValueType, std::size_t Index, std::size_t ParametersLength>
    struct Variable {
        using variable = void;
        static constexpr auto index = Index;

        KOKKOS_FUNCTION
        constexpr ValueType operator()(const LinearAlgebra::Matrix<ValueType, ParametersLength>& x) const;
    };

    template <typename ValueType, LinearAlgebra::ValidMathOperation Op, typename Num1, typename Num2, std::size_t ParametersLength>
    struct Operation {
        static constexpr auto operation = Op;
        using num1 = Num1;
        using num2 = Num2;

        KOKKOS_FUNCTION
        ValueType operator()(const LinearAlgebra::Matrix<ValueType, ParametersLength>& x) const;
    };

    // Functions

    template <typename ValueType, std::size_t P, typename Num, std::size_t ParametersLength>
    struct Power {
        using power = void;
        static constexpr auto p = P;
        using num = Num;

        KOKKOS_FUNCTION
        ValueType operator()(const LinearAlgebra::Matrix<ValueType, ParametersLength>& x) const;
    };

    template <typename ValueType, std::size_t VariableIndex, typename X, std::size_t ParametersLength>
    struct Derivative {
    private:
        KOKKOS_FUNCTION
        static auto getDerivative();

    public:
        using derivative = decltype(getDerivative());

        KOKKOS_FUNCTION
        ValueType operator()(const LinearAlgebra::Matrix<ValueType, ParametersLength>& x) const;
    };
}

#include "CompileOperations.tpp"

#endif
