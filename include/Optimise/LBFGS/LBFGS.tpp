#ifndef STOCK_ANALYSIS_LBFGS_TPP
#define STOCK_ANALYSIS_LBFGS_TPP

#include "LBFGS.hpp"

namespace Optimise {
    template<typename NumericalType, std::size_t ParametersLength>
    LBFGS<NumericalType, ParametersLength>::LBFGS(const LinearAlgebra::Matrix<NumericalType, ParametersLength>& initialParameters): initialParameters(initialParameters) {}

    template<typename NumericalType, std::size_t ParametersLength>
    template <typename ObjectiveFunctionType>
    void LBFGS<NumericalType, ParametersLength>::optimise(const ObjectiveFunctionType objectiveFunction) const {
        
    }
}

#endif
