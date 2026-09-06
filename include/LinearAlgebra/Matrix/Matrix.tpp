#ifndef STOCK_ANALYSIS_MATRIXOPERATIONS_TPP
#define STOCK_ANALYSIS_MATRIXOPERATIONS_TPP

#include "Matrix.hpp"

namespace LinearAlgebra {
    template<typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    Matrix<MatrixType, Dimension1, Dimension2>::Matrix(): view("LinearAlgebra::Matrix", Dimension1, Dimension2) {}

    template <typename MatrixType, std::size_t Dimension1, std::size_t Dimension2>
    std::string matrixToString(const Matrix<MatrixType, Dimension1, Dimension2>& matrix) {
        auto hostM = Kokkos::create_mirror_view(matrix.view);
        Kokkos::deep_copy(hostM, matrix.view);

        std::string outputString = "[\n";

        for (std::size_t i = 0; i < Dimension1; i++) {
            outputString += '\t';

            for (std::size_t j = 0; j < Dimension2; j++) {
                outputString += std::to_string(hostM(i, j));

                if (i != Dimension1 - 1 || j != Dimension2 - 1) {
                    outputString += ", ";
                }
            }

            outputString += '\n';
        }

        outputString += "\n]";

        return outputString;
    }
}

#endif
