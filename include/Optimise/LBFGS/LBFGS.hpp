#ifndef STOCK_ANALYSIS_LBFGS_HPP
#define STOCK_ANALYSIS_LBFGS_HPP

#include <LinearAlgebra.hpp>

namespace Optimise {
    template <typename NumericalType, std::size_t ParametersLength>
    class LBFGS {
        LinearAlgebra::Matrix<NumericalType, ParametersLength> initialParameters;

        public:
        LBFGS(const LinearAlgebra::Matrix<NumericalType, ParametersLength>& initialParameters);

        template <typename ObjectiveFunctionType>
        void optimise(ObjectiveFunctionType objectiveFunction) const;
    };
}

#include "LBFGS.tpp"

#endif
