#ifndef STOCK_ANALYSIS_LBFGS_TPP
#define STOCK_ANALYSIS_LBFGS_TPP

#include "LBFGS.hpp"

#include <Optimise.hpp>

namespace Optimise {
    template <typename NumericalType, std::size_t ParametersLength, typename ObjectiveFunction>
    LinearAlgebra::Matrix<NumericalType, ParametersLength> getGradientCompileTime(LinearAlgebra::Matrix<NumericalType, ParametersLength>& x, ObjectiveFunction&& objectiveFunction) {
        using objectiveFunctionType = decltype(objectiveFunction(x));

        const auto gradientArray = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<NumericalType, ParametersLength>{
                Derivative<NumericalType, I, objectiveFunctionType, ParametersLength>{}(x)...
            };
        }(std::make_index_sequence<ParametersLength>{});

        return Matrix<NumericalType, ParametersLength>(gradientArray);
    }

    template<typename NumericalType, std::size_t ParametersLength>
    LBFGS<NumericalType, ParametersLength>::LBFGS(const LinearAlgebra::Matrix<NumericalType, ParametersLength>& initialParameters): initialParameters(initialParameters) {}

    template<typename NumericalType, std::size_t ParametersLength>
    template <typename ObjectiveFunctionType>
    void LBFGS<NumericalType, ParametersLength>::optimiseCompileTimeExpression(const ObjectiveFunctionType objectiveFunction) const {
        
    }
}

#endif
