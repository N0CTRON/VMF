#include <iostream>

#include "vmf.cpp"
#include "vmf.cu"
#include "cuMM.cu"
#include "aos.hpp"

// This is a demo of the "VMF" library, devoloped by XeTute. 
// It uses the "AOS" and "cuMM" header-only libraries, which are also dev. by XeTute.
// XeTutes website: "https://xetute.neocities.org/"
// AOS GitHub: "https://www.github.com/N0CTRON/array-on-steriods/"
// cuMM GitHub: "https://www.github.com/N0CTRON/cuMM/"

int main()
{
    std::srand(std::time(NULL));

    float x0 = std::rand() % 10;
    float x1 = x0;
    float x2 = x0;
    float x3 = x0;
    float x4 = x0;

    VMF::heaviside(&x1);
    VMF::leakyReLU(&x2, &x0); // Assuming you want to apply leaky ReLU to x2 with a slope of x0
    VMF::ReLU(&x3);
    VMF::sigmoid(&x4);

    std::cout << "x: " << x0
        << "\nHeaviside step: " << x1
        << "\nLeaky ReLU: " << x2
        << "\nReLU: " << x3
        << "\nSigmoid: " << x4 << '\n';

    return 0;
}
