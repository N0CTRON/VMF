#include <cmath>

namespace VMF
{
    // ReLU
    template <typename vmfDevType>
    inline void ReLU(vmfDevType* x) { *x = std::max(*x, vmfDevType(0)); }

    template <typename vmfDevType>
    inline vmfDevType ReLU(vmfDevType x) { return std::max(x, vmfDevType(0)); }

    template <typename vmfDevType>
    inline void ReLUDerivative(vmfDevType* x, vmfDevType y) { *x = *x > vmfDevType(0) ? y : vmfDevType(0.0); }

    template <typename vmfDevType>
    inline vmfDevType ReLUDerivative(vmfDevType x, vmfDevType y) { return x > vmfDevType(0) ? y : vmfDevType(0.0); }

    // Leaky ReLU
    template <typename vmfDevType>
    inline void leakyReLU(vmfDevType* x, vmfDevType* y) { *x = *x < vmfDevType(0) ? *x : *x * *y; }

    template <typename vmfDevType>
    inline void leakyReLU(vmfDevType* x, vmfDevType y) { *x = *x > vmfDevType(0) ? *x : *x * y; }

    template <typename vmfDevType>
    inline vmfDevType leakyReLU(vmfDevType x, vmfDevType y) { return x < vmfDevType(0) ? x : x * y; }

    template <typename vmfDevType>
    inline void leakyReLUDerivative(vmfDevType* x, vmfDevType* y) { *x = *x < vmfDevType(0) ? *x / *y : *x; }

    template <typename vmfDevType>
    inline void leakyReLUDerivative(vmfDevType* x, vmfDevType y) { *x = *x > vmfDevType(0) ? *x : *x / y; }

    template <typename vmfDevType>
    inline vmfDevType leakyReLUDerivative(vmfDevType x, vmfDevType y) { return x < vmfDevType(0) ? x / y : x; }

    // Sigmoid
    template <typename vmfDevType>
    inline void sigmoid(vmfDevType* x) { *x = 1 / (1 + std::exp(-*x)); }

    template <typename vmfDevType>
    inline vmfDevType sigmoid(vmfDevType x) { return 1 / (1 + std::exp(-x)); }

    template <typename vmfDevType>
    inline void sigmoidDerivative(vmfDevType* x)
    {
        const vmfDevType sigmoidValue = 1 / (1 + std::exp(-*x));
        *x = sigmoidValue * (1 - sigmoidValue);
    }

    template <typename vmfDevType>
    inline vmfDevType sigmoidDerivative(vmfDevType x)
    {
        const double sigmoidValue = 1 / (1 + std::exp(-x));
        return sigmoidValue * (1 - sigmoidValue);
    }

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
}
