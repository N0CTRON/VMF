#include <cmath>

#include "aos.hpp"

namespace VMF
{
    // ReLU
    template <typename vmfDevType>
    inline void ReLU(vmfDevType* x) { *x = std::max(*x, vmfDevType(0)); }

    template <typename vmfDevType>
    inline vmfDevType ReLU(vmfDevType x) { return std::max(x, vmfDevType(0)); }

    template <typename vmfDevType>
    inline void ReLUDerivative(vmfDevType* ) { *x = *x > vmfDevType(0) ? x : vmfDevType(0.0); }

    template <typename vmfDevType>
    inline vmfDevType ReLUDerivative(vmfDevType x) { return x > vmfDevType(0) ? x : vmfDevType(0.0); }

    // Leaky ReLU
    template <typename vmfDevType>
    inline void leakyReLU(vmfDevType* x) { *x = *x < vmfDevType(0) ? *x : *x * 0.01; }

    template <typename vmfDevType>
    inline vmfDevType leakyReLU(vmfDevType x) { return x < vmfDevType(0) ? x : x * 0.01; }

    template <typename vmfDevType>
    inline void leakyReLUDerivative(vmfDevType* x) { *x = *x < vmfDevType(0) ? *x * 100 : *x; }

    template <typename vmfDevType>
    inline vmfDevType leakyReLUDerivative(vmfDevType x) { return x < vmfDevType(0) ? x * 100 : x; }

    // Sigmoid
    template <typename vmfDevType>
    inline void sigmoid(vmfDevType* x) { *x = 1 / (1 + std::exp(-*x)); }

    template <typename vmfDevType>
    inline vmfDevType sigmoid(vmfDevType x) { return 1 / (1 + std::exp(-x)); }

    template <typename vmfDevType>
    inline void sigmoidDerivative(vmfDevType* x) { *x = *x * (vmdDevType(1.0) - *x) }

    template <typename vmfDevType>
    inline vmfDevType sigmoidDerivative(vmfDevType x) { return x * (vmfDevType(1.0) - x); }

    // Heaviside step function
    template <typename vmfDevType>
    inline void heaviside(vmfDevType* x) { *x = (*x >= vmfDevType(0)) * vmfDevType(1); }

    template <typename vmfDevType>
    inline vmfDevType heaviside(vmfDevType x) { return (x >= vmfDevType(0)) * vmfDevType(1); }

    // Factorial
    template <typename vmfDevType>
    inline void factorial(vmfDevType* x) { *x = factorial(*x); }

    template <typename vmfDevType>
    inline vmfDevType factorial(vmfDevType x)
    {
        if (x == 0) return 1;
        return x * factorial(x - 1);
    }

    // Tanh is defined in cmath.

    // Convolute 1D
    template <typename vmfDevType>
    inline void convolute1D(AOS<vmfDevType> input, AOS<vmfDevType> kernel, AOS<vmfDevType> output)
    {
        std::uint64_t outputSize = input.size() - kernel.size() + 1;
        for (std::uint64_t i = 0; i < outputSize; ++i)
        {
            output[i] = vmfDevType(0);
            for (std::uint64_t j = 0; j < kernel.size(); ++j) output[i] += input[i + j] * kernel[j];
        }
    }

    template <typename vmfDevType>
    inline AOS<vmfDevType> convolute1D(AOS<vmfDevType> input, AOS<vmfDevType> kernel)
    {
        constexpr std::uint64_t vmfSize = (std::uint64_t)sizeof(vmfDevType);
        std::uint64_t outputSize = input.size() - kernel.size() + 1;
        AOS<vmfDevType> output(outputSize, vmfDevType(0));
        for (std::uint64_t i = 0; i < outputSize; ++i)
            for (std::uint64_t j = 0; j < kernel.size(); ++j) output[i] += input[i + j] * kernel[j];
        return output;
    }

    // Dotproduct
    template <typename vmfDevType>
    inline void dotProduct(vmfDevType* vec1, vmfDevType* vec2, vmfDevType* result, std::uint64_t arraySize)
    {
        vmfDevType sum = 0;
        for (std::uint64_t i = 0; i < arraySize; ++i) sum += vec1[i] * vec2[i];
        *result = sum;
    }


    template <typename vmfDevType>
    inline vmfDevType dotProductHost(const vmfDevType* vec1, const vmfDevType* vec2, std::uint64_t arraySize)
    {
        vmfDevType sum(0);
        for (std::uint64_t i = 0; i < arraySize; ++i) sum += vec1[i] * vec2[i];
        return sum;
    }
}
