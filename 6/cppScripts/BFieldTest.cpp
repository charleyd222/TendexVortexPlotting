#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 

// Code to test my implementation of the B field since its a lot more complicated than the E field
using namespace std;
using namespace Eigen;

extern "C" {

struct mat {
    double m[3][3];
};

// Define hyperbolic functions
double sech(double x) {
    return 1.0 / cosh(x);
}
double coth(double x) {
    return 1.0 / tanh(x);
}

// Function to compute the matrix
mat f(double x, double y, double z, double R, double S, double vX) {
        Eigen::Matrix3f B;
        mat BM;
    
        // Precompute common
        double r = std::powf(x*x + y*y + z*z, .5);
        double r2 = r*r;
        double r3 = r2*r;
        
        double secPlus = sech(S*(r+R));
        double secPlus2 = secPlus * secPlus;
        double secMin = sech(S*(r-R));
        double secMin2 = secMin * secMin;
        double tanhPlus = tanh(S*(r+R));
        double tanhMin = tanh(S*(r-R));
        double CRS = coth(R*S);

        double secMinPlus2 = secMin2 - secPlus2;
        double secMinPlus2Tanh = (secMin2*tanhMin) - (secPlus2*tanhPlus);

        // Compute matrix (row, col)
        B(0, 0) = 0.0;
        B(0, 1) = .25 * CRS * S * x * z * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2);
        B(0, 2) = -.25 * CRS * S * x * y * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2);

        B(1, 0) = 0.0;
        B(1, 1) = .25 * CRS * S * y * z * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2);
        B(1, 2) = -.25 * CRS * S * (y * y * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2) - secMinPlus2/r);

        B(2, 0) = 0.0;
        B(2, 1) = .25 * CRS * S * (z * z * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2) - secMinPlus2/r);
        B(2, 2) = .25 * CRS * S * y * z * (S * secMinPlus2 / r3 + 2 * S*secMinPlus2Tanh / r2);


        // Process for output
        BM.m[0][0] = B(0,0);
        BM.m[0][1] = B(0,1);
        BM.m[0][2] = B(0,2);
        BM.m[1][0] = B(1,0);
        BM.m[1][1] = B(1,1);
        BM.m[1][2] = B(1,2);
        BM.m[2][0] = B(2,0);
        BM.m[2][1] = B(2,1);
        BM.m[2][2] = B(2,2);
        return BM;
}

}


int main() {
    return 0;
}