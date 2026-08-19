#ifndef STOCK_ANALYSIS_VECTOR_TPP
#define STOCK_ANALYSIS_VECTOR_TPP

#include "setup/Kokkos_Setup_Cuda.hpp"

namespace LinearAlgebra {
    template <typename VectorType>
    Vector<VectorType>::Vector(const int length):
    length(length),
    v("LinearAlgebra::Vector", length)
    {}

    template<typename VectorType>
    Vector<VectorType>::Vector(Kokkos::View<VectorType *> vectorView) : length(vectorView.extent(0)) {
        this->v = vectorView;
    }

    template<typename VectorType>
    Kokkos::View<VectorType *> Vector<VectorType>::getV() const {
        return this->v;
    }

    template<typename VectorType>
    std::string Vector<VectorType>::to_string() const {
        auto vHost = Kokkos::create_mirror_view(this->v);
        Kokkos::deep_copy(vHost, this->v);

        std::string outputString = "[";

        for (int i = 0; i < this->length; i++) {
            outputString += std::to_string(vHost(i));

            if (i < this->length - 1) {
                outputString += ", ";
            }
        }

        outputString += ']';

        return outputString;
    }

    template<typename VectorType>
    template<typename Operation>
    Kokkos::View<VectorType*> Vector<VectorType>::elementWiseOperation(
        const Kokkos::View<VectorType*> v1,
        const Kokkos::View<VectorType*> v2,
        const MathOperation<VectorType, Operation> mathOperation
    ) const {
        const int lengthCopy = v1.extent(0);

        if (lengthCopy != v2.extent(0)) {
            throw std::invalid_argument(
                "Cannot perform operation on vectors of different lengths."
            );
        }

        Kokkos::View<VectorType*> vResult("v_result Vector::elementWiseOperation", lengthCopy);

        Kokkos::parallel_for(
            "LinearAlgebra::Vector::elementWiseOperation",
            lengthCopy,
            KOKKOS_LAMBDA(const int i) {
                vResult(i) = mathOperation(v1(i), v2(i));
            }
        );

        Kokkos::fence("LinearAlgebra::Vector::elementWiseOperation");

        return vResult;
    }

    template<typename VectorType>
    template<typename Operation>
    Kokkos::View<VectorType*> Vector<VectorType>::elementWiseOperation(
        const Kokkos::View<VectorType*> v1,
        const VectorType scalar,
        const MathOperation<VectorType, Operation> mathOperation
    ) const {
        const int lengthCopy = v1.extent(0);

        Kokkos::View<VectorType*> v_result("v_result Vector::elementWiseOperation", lengthCopy);

        Kokkos::parallel_for(
            "LinearAlgebra::Vector::elementWiseOperation",
            lengthCopy,
            KOKKOS_LAMBDA(const int i) {
                v_result(i) = mathOperation(v1(i), scalar);
            }
        );

        return v_result;
    }

    template<typename VectorType>
    template<typename Operation>
    Kokkos::View<VectorType*> Vector<VectorType>::elementWiseOperation(
        const VectorType scalar,
        const Kokkos::View<VectorType*> v2,
        const MathOperation<VectorType, Operation> mathOperation
    ) const {
        const int lengthCopy = v2.extent(0);

        Kokkos::View<VectorType*> v_result("v_result Vector::elementWiseOperation", lengthCopy);

        Kokkos::parallel_for(
            "LinearAlgebra::Vector::elementWiseOperation",
            lengthCopy,
            KOKKOS_LAMBDA(const int i) {
                v_result(i) = mathOperation(scalar, v2(i));
            }
        );

        return v_result;
    }

    template<typename VectorType>
    void Vector<VectorType>::fill(const VectorType value) {
        auto vReference = this->v;
        const auto lengthCopy = this->length;

        Kokkos::parallel_for(
            "LinearAlgebra::Vector::fill",
            lengthCopy,
            KOKKOS_LAMBDA(const int i) {
                vReference(i) = value;
            }
        );
    }

    template<typename VectorType>
    template<typename FillFunctor>
    void Vector<VectorType>::fill() {
        auto vReference = this->v;
        const auto lengthCopy = this->length;

        const FillFunctor fillFunctor{};

        Kokkos::parallel_for(
            "LinearAlgebra::Vector::fill",
            lengthCopy,
            KOKKOS_LAMBDA(const int i) {
                vReference(i) = fillFunctor(i);
            }
        );
    }

    template<typename VectorType>
    VectorType Vector<VectorType>::operator()(const std::size_t index) const {
        return this->v(index);
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator+(const Vector& otherVector) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                otherVector.getV(),
                MathOperation<VectorType, Add>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator+(const VectorType& scalar) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                scalar,
                MathOperation<VectorType, Add>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator-(const Vector& otherVector) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                otherVector.getV(),
                MathOperation<VectorType, Subtract>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator-(const VectorType& scalar) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                scalar,
                MathOperation<VectorType, Subtract>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator*(const Vector& otherVector) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                otherVector.getV(),
                MathOperation<VectorType, Multiply>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator*(const VectorType& scalar) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                scalar,
                MathOperation<VectorType, Multiply>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator/(const Vector& otherVector) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                otherVector.getV(),
                MathOperation<VectorType, Divide>{}
            )
        );
    }

    template<typename VectorType>
    Vector<VectorType> Vector<VectorType>::operator/(const VectorType& scalar) const {
        return Vector(
            this->elementWiseOperation(
                this->v,
                scalar,
                MathOperation<VectorType, Divide>{}
            )
        );
    }

    template <typename VectorType>
    Vector<VectorType> operator+(const VectorType& scalar, const Vector<VectorType>& vector) {
        return vector + scalar;
    }

    template <typename VectorType>
    Vector<VectorType> operator-(const VectorType& scalar, const Vector<VectorType>& vector) {
        return Vector(
            vector.elementWiseOperation(
                scalar,
                vector.getV(),
                MathOperation<VectorType, Subtract>{}
            )
        );
    }

    template <typename VectorType>
    Vector<VectorType> operator*(const VectorType& scalar, const Vector<VectorType>& vector) {
        return vector * scalar;
    }

    template <typename VectorType>
    Vector<VectorType> operator/(const VectorType& scalar, const Vector<VectorType>& vector) {
        return Vector(
            vector.elementWiseOperation(
                scalar,
                vector.getV(),
                MathOperation<VectorType, Divide>{}
            )
        );
    }
}

#endif
