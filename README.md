**Overview**
The Various Math Functions (VMF) library is a collection of mathematical functions for various applications. It provides implementations of ReLU, Leaky ReLU, Sigmoid, and Heaviside step functions, currently on CPU only, but were working on a CUDA version.

**Features**
The VMF library offers the following features:

* ReLU (Rectified Linear Unit) activation function
* Leaky ReLU activation function with adjustable slope
* Sigmoid function
* Heaviside step function

**Example Usage**
Here is an example of how to use the VMF library:
```cpp
#include <iostream>

#include "vmf.cpp"
#include "aos.hpp"

int main() {
    // Create a vector of floats
    AOS<float> x(5);

    // Initialize the vector with random values
    std::srand(std::time(NULL));
    for (int i = 0; i < 5; ++i) x[i] = (std::rand() % 2 == 1) ? (std::rand() % 5) : (-(std::rand() % 5));

    // Apply various activation functions
    VMF::heaviside(&x[1]);
    VMF::leakyReLU(&x[2], (float)0.01); // Apply leaky ReLU to x2 with a slope of 0.01
    VMF::ReLU(&x[3]);
    VMF::sigmoid(&x[4]);

    // Print the results
    std::cout << "x: " << x[0] << "\n";
    std::cout << "Heaviside step: " << x[1] << "\n";
    std::cout << "Leaky ReLU: " << x[2] << "\n";
    std::cout << "ReLU: " << x[3] << "\n";
    std::cout << "Sigmoid: " << x[4] << "\n";

    return 0;
}
```
**Building and Running**
To build and run the VMF library, follow these steps:

1. Clone the repository: `git clone https://github.com/N0CTRON/VMF`
2. Navigate to the repository directory: `cd VMF`
3. Compile the code: `nvcc -o main main.cpp vmf.cpp aos.hpp cuMM.cu`
4. Run the program: `chmod +x ./main && ./main`

**Note**
The VMF library is developed by XeTute, a Pakistani startup. It uses the AOS and cuMM libraries, which are also developed by XeTute.

**Importing**
To use the VMF library in your own projects, simply include the `vmf.cpp` / `vmf.cu` file in your code and use the provided functions.

**License**
The VMF library is licensed under the MIT License.
