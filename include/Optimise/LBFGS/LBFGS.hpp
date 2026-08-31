#ifndef STOCK_ANALYSIS_LBFGS_HPP
#define STOCK_ANALYSIS_LBFGS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    template <typename NumericalType, std::size_t ParametersLength, typename ObjectiveFunction>
    LinearAlgebra::Matrix<NumericalType, ParametersLength> getGradientCompileTime(LinearAlgebra::Matrix<NumericalType, ParametersLength>& x, ObjectiveFunction&& objectiveFunction);

    template <typename NumericalType, std::size_t ParametersLength>
    class LBFGS {
        LinearAlgebra::Matrix<NumericalType, ParametersLength> initialParameters;

        public:
        LBFGS(const LinearAlgebra::Matrix<NumericalType, ParametersLength>& initialParameters);

        template <typename ObjectiveFunctionType>
        void optimiseCompileTimeExpression(ObjectiveFunctionType objectiveFunction) const;
    };
}

#include "LBFGS.tpp"

#endif
