#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/cmath>

typedef unsigned long int cuSize; //<-- The GTX1060 3GB doesn't support size_t

namespace VMF_CUDA
{
    // ReLU
    template <typename vmfDevType> //<-- vmfDevType = VMF[Various Math Functions] Devoloper Type
    __global__ void ReLU(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] > vmfDevType(0) ? x[threadIndexX] : vmfDevType(0);
        __syncthreads();
    }

    // Leaky ReLU
    template <typename vmfDevType>
    __global__ void leakyReLU(vmfDevType* x, vmfDevType y, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] < vmfDevType(0) ? x[threadIndexX] : x[threadIndexX] * y;
        __syncthreads();
    }

    // Sigmoid
    template <typename vmfDevType>
    __global__ void sigmoid(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = vmfDevType(0.5) * (vmfDevType(1.0) + tanh(vmfDevType(0.5) * x[threadIndexX]));
        __syncthreads();
    }

    // Heaviside step function
    template <typename vmfDevType>
    __global__ void heaviside(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = (x[threadIndexX] >= vmfDevType(0)) * vmfDevType(1);
        __syncthreads();
    }

    // Tanh is defined in cuda/std/cmath.
}
