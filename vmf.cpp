#include <algorithm>
#include <cmath>

namespace VMF
{
    // ReLU
    template <typename vmfDevType>
    inline void ReLU(vmfDevType* x) { *x = std::max(*x, vmfDevType(0)); }

    template <typename vmfDevType>
    inline vmfDevType ReLU(vmfDevType x) { return std::max(x, vmfDevType(0)); }

    // Leaky ReLU
    template <typename vmfDevType>
    inline void leakyReLU(vmfDevType* x, vmfDevType* y) { *x *= (*x <= vmfDevType(0)) * (*x * *y) + (*x > vmfDevType(0)); }

    template <typename vmfDevType>
    inline vmfDevType leakyReLU(vmfDevType x, vmfDevType y) { return x * ((x <= vmfDevType(0)) * (x * y) + (x > vmfDevType(0))); }

    // Sigmoid
    template <typename vmfDevType>
    inline void sigmoid(vmfDevType* x) { *x = vmfDevType(0.5) * (vmfDevType(1.0) + std::tanh(vmfDevType(0.5) * *x)); }

    template <typename vmfDevType>
    inline vmfDevType sigmoid(vmfDevType x) { return vmfDevType(0.5) * (vmfDevType(1.0) + std::tanh(vmfDevType(0.5) * x)); }

    // Heaviside step function
    template <typename vmfDevType>
    inline void heaviside(vmfDevType* x) { *x = (*x >= vmfDevType(0)) * vmfDevType(1); }

    template <typename vmfDevType>
    inline vmfDevType heaviside(vmfDevType x) { return (x >= vmfDevType(0)) * vmfDevType(1); }

    // Tanh is defined in cmath.
}
