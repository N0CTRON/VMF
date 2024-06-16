#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/cmath>

namespace VMF_CUDA
{
    typedef unsigned long int cuSize; //<-- The GTX1060 3GB doesn't support size_t
    // ReLU
    template <typename vmfDevType> //<-- vmfDevType = VMF[Various Math Functions] Devoloper Type
    __global__ void ReLU(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] > vmfDevType(0) ? x[threadIndexX] : vmfDevType(0);
        __syncthreads();
    }

    template <typename vmfDevType>
    __global__ void ReLUDerivative(vmfDevType* x, vmfDevType y, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] > vmfDevType(0) ? y : vmfDevType(0.0);
        __syncthreads();
    }

    // Leaky ReLU
    template <typename vmfDevType>
    __global__ void leakyReLU(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] < vmfDevType(0) ? x[threadIndexX] : x[threadIndexX] * vmfDevType(0.01);
        __syncthreads();
    }

    template <typename vmfDevType>
    __global__ void leakyReLUDerivative(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = x[threadIndexX] < vmfDevType(0) ? x[threadIndexX] * 100 : x[threadIndexX];
        __syncthreads();
    }

    // Sigmoid
    template <typename vmfDevType>
    __global__ void sigmoid(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize) x[threadIndexX] = vmfDevType(1.0) / (vmfDevType(1.0) + exp(-x[threadIndexX]));
        __syncthreads();
    }

    template <typename vmfDevType>
    __global__ void sigmoidDerivative(vmfDevType* x, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadIndexX < arraySize)
        {
            const vmfDevType sigmoidValue = vmfDevType(1.0) / (vmfDevType(1.0) + exp(-x[threadIndexX]));
            x[threadIndexX] = sigmoidValue * (vmfDevType(1.0) - sigmoidValue);
        }
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

    template <typename vmfDevType>
    __global__ void convolute1D(vmfDevType* input, vmfDevType* kernel, vmfDevType* output, cuSize inputSize, cuSize kernelSize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        cuSize outputSize = inputSize - kernelSize + 1;
        if (threadIndexX < outputSize)
        {
            vmfDevType sum(0);
            for (cuSize j = 0; j < kernelSize; ++j) sum += input[threadIndexX + j] * kernel[j];
            output[threadIndexX] = sum;
        }
        __syncthreads();
    }

    template <typename vmfDevType>
    __global__ void dotProduct(vmfDevType* vars0, vmfDevType* vars1, vmfDevType* result, cuSize arraySize)
    {
        cuSize threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
        *result = 0;
        if (i < arraySize) *result += vars0[i] * vars1[i];
        __syncthreads();
    }
}
