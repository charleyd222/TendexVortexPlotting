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
mat f(double x, double y, double z, double w) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f E;
    Eigen::Matrix3f e11;
    Eigen::Matrix3f e12;
    Eigen::Matrix3f e13;
    Eigen::Matrix3f e14;
    Eigen::Matrix3f eAd;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;

    mat EM;    
    double r = std::powf(x * x + y * y + z * z,.5);
    double r2 = r*r;
    double r3 = r2*r;
    double r4 = r2*r2;
    double r5 = r3*r2;
    double r6 = r3*r3;
    double w2 = w*w;
    
    double aI = std::cos(w); 
    double IpqXpXq = (aI * (x * x - y * y));

    // Term 1
    e1 << 105 * aI * (x*x*x*x - x*x * y*y) / (r5*r4) + aI * (15 * y*y - 75 * x*x) / (r5*r2) + 6 * aI / r5, 105 * aI * (x*x*x*y - x*y*y*y) / (r5*r4), 105 * aI * (x*x*x*z - x*z * y*y) / (r5*r4) - aI * 30 * x*z / (r5*r2),
          105 * aI * (x*x*x*y - x*y*y*y) / (r5*r4), 105 * aI * (x*x*y*y - y*y*y*y) / (r5*r4) + aI * (75 * y*y - 15 * x*x) / (r5*r2) - 6 * aI / r5, 105 * aI * (x*x*y*z - z*y*y*y) / (r5*r4) + aI * 30 * y*z / (r5*r2),
          105 * aI * (x*x*x*z - x*y*y*z) / (r5*r4) - aI * 30 * x*z / (r5*r2), 105 * aI * (x*x*y*z - z*y*y*y) / (r5*r4) + aI * 30 * y*z / (r5*r2), 105 * aI * (x*x*z*z - z*z*y*y) / (r5*r4) + aI * 15 * (y*y - x*x) / (r5*r2);

    // Term 2
    e2 << w2*aI / r3 * (3.0*z*z / r2 - 1), 0.0, -3.0 * aI * w2 * x * z / r5,
          0.0, w2*aI / r3 * (-3.0*z*z / r2 + 1), 3.0 * aI * w2 * y * z / r5, 
          -3.0 * aI * w2 * x * z / r5, 3.0 * aI * w2 * y * z / r5, 3.0 * aI * w2 * (x * x - y * y) / r5;

    // Term 3
    // With D/Dxp
    e3 << -w2 * aI * aI * 2 * (4 * x*x / r6 - 1/r4), 8 * -w2 * aI * aI * x*y / r6, 4 * -w2 * aI * aI * x*z / r6,
          8 * -w2 * aI * aI * x*y / r6, -w2 * aI * aI * 2 * (4 * y*y / r6 - 1/r4), 4 * -w2 * aI * aI * y*z / r6,
          4 * -w2 * aI * aI * x*z / r6, 4 * -w2 * aI * aI * y*z / r6, 0.0;

    //e3 << w2 * aI * aI * 2 * x / r4, w2 * aI * aI * (x+y) / r4, w2 * aI * aI * z / r4,
    //      w2 * aI * aI * (x+y) / r4, w2 * aI * aI * 2 * y / r4, w2 * aI * aI * z / r4,
    //      w2 * aI * aI * z / r4, w2 * aI * aI * z / r4, 0.0;

    // Term 4
    e4 << -1.0 * aI * w2*w2 / r, 0.0, 0.0,
          0.0, 1.0 * aI * w2*w2 / r, 0.0,
          0.0, 0.0, 0.0;

    

    E = e1+e2+e3+e4;


    // Process for output
    EM.m[0][0] = E(0,0);
    EM.m[0][1] = E(0,1);
    EM.m[0][2] = E(0,2);
    EM.m[1][0] = E(1,0);
    EM.m[1][1] = E(1,1);
    EM.m[1][2] = E(1,2);
    EM.m[2][0] = E(2,0);
    EM.m[2][1] = E(2,1);
    EM.m[2][2] = E(2,2);
    return EM;
}

}


int main() {
    return 0;
}