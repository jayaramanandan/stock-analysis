#ifndef STOCK_ANALYSIS_VECTOR_HPP
#define STOCK_ANALYSIS_VECTOR_HPP

#include <string>

#include <Kokkos_Core.hpp>

#include "../MathOperation.hpp"

namespace LinearAlgebra {
    template <typename VectorType>
    class Vector {
        int length;
        Kokkos::View<VectorType*> v;

        public:
        explicit Vector(int length);
        explicit Vector(Kokkos::View<VectorType*> vectorView);

        Kokkos::View<VectorType*> getV() const;

        [[nodiscard]] std::string to_string() const;

        template <typename Operation>
        Kokkos::View<VectorType*> elementWiseOperation(Kokkos::View<VectorType*> v1, Kokkos::View<VectorType*> v2, MathOperation<VectorType, Operation> mathOperation) const;

        template <typename Operation>
        Kokkos::View<VectorType*> elementWiseOperation(Kokkos::View<VectorType*> v1, VectorType scalar, MathOperation<VectorType, Operation> mathOperation) const;

        template <typename Operation>
        Kokkos::View<VectorType*> elementWiseOperation(VectorType scalar, Kokkos::View<VectorType*> v2, MathOperation<VectorType, Operation> mathOperation) const;

        void fill(VectorType value);

        template<typename FillFunctor>
        void fill();

        VectorType operator[](std::size_t index) const;

        Vector operator+(const Vector& otherVector) const;

        Vector operator+(const VectorType& scalar) const;

        Vector operator-(const Vector& otherVector) const;

        Vector operator-(const VectorType& scalar) const;

        Vector operator*(const Vector& otherVector) const;

        Vector operator*(const VectorType& scalar) const;

        Vector operator/(const Vector& otherVector) const;

        Vector operator/(const VectorType& scalar) const;
    };

    template <typename VectorType>
    Vector<VectorType> operator+(const VectorType& scalar, const Vector<VectorType>& vector);

    template <typename VectorType>
    Vector<VectorType> operator-(const VectorType& scalar, const Vector<VectorType>& vector);

    template <typename VectorType>
    Vector<VectorType> operator*(const VectorType& scalar, const Vector<VectorType>& vector);

    template <typename VectorType>
    Vector<VectorType> operator/(const VectorType& scalar, const Vector<VectorType>& vector);
}

#include "Vector.tpp"

#endif
