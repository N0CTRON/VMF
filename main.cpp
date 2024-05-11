#include <iostream>
#include <vector>

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

    float seed((std::rand() % 2 == 1) ? (std::rand() % 5) : (-(std::rand() % 5)));
    AOS<float> x(5, seed);

    VMF::heaviside(&x[1]);
    VMF::leakyReLU(&x[2], (float)0.01); // Assuming you want to apply leaky ReLU to x2 with a slope of 0.01
    VMF::ReLU(&x[3]);
    VMF::sigmoid(&x[4]);

    std::cout << "x: " << x[0]
        << "\nHeaviside step: " << x[1]
        << "\nLeaky ReLU: " << x[2]
        << "\nReLU: " << x[3]
        << "\nSigmoid: " << x[4] << '\n';

    return 0;
}
