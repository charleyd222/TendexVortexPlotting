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
    mat EM;  
    double x3 = pow(x, 3);
    double x2 = pow(x, 2);
    double y2 = pow(y, 2);
    double z2 = pow(z, 2);
    double w2 = pow(w, 2);
    double x4 = pow(x, 4);
    double w4 = pow(w, 4);
    double y3 = pow(y, 3);
    double y4 = pow(y, 4);

    double r = powf(x2 + y2 + z2, .5);
    double r2 = x2 + y2 + z2;
    double r3 = pow(r,3);
    double r5 = pow(r,5);
    double r7 = pow(r,7);
    double r9 = pow(r,9);

    double temp1 = ((3*x2*cos(w))/(r5) - cos(w)/(r3) - (3*x*y*sin(w))/(r5));
    double temp2 = ((3*y2*cos(w))/(r5) - cos(w)/(r3) + (3*x*y*sin(w))/(r5));
    double temp3 = ((3*z2*cos(w))/(r5) - cos(w)/(r3));
    
    double temp8 = (-210*x2*y2*cos(w));
    double temp9 = (210*x2*y*z*cos(w));
    double temp10 = (105*x*y2*z*sin(w));
    double temp11 = (210*x*y2*z*cos(w));
    double temp12 = (105*x2*y*z*sin(w));
    double temp13 = (105*x2*y2*sin(w));
    double temp14 = (6*w2*x*y*cos(w));
    double temp15 = (105*x3*y*sin(w));
    double temp16 = (105*x*y3*sin(w));
    double temp17 = (6*w2*y*z*cos(w));
    double temp18 = (105*x3*z*sin(w));
    double temp19 = (6*w2*x*z*sin(w));
    double temp20 = (6*w2*x*z*cos(w));
    double temp21 = (105*y3*z*sin(w));
    double temp22 = (6*w2*y*z*sin(w));
    double temp23 = (90*x*y*cos(w));
    double temp24 = (30*y*z*cos(w));
    double temp25 = (30*x*z*sin(w));
    double temp26 = (30*x*z*cos(w));
    double temp27 = (30*y*z*sin(w));
    double temp28 = (x2 + y2 + z2);
    double temp29 = (3*x*y*sin(w));
    double temp30 = (15*y2*sin(w));
    double temp31 = (30*x2*cos(w));
    double temp32 = (30*y2*cos(w));
    double temp33 = (15*x2*sin(w));
    double temp34 = (3*z2*cos(w));
    double temp35 = (3*x2*cos(w));
    double temp36 = (3*y2*cos(w));
    double temp37 = (w4*sin(w));
    double temp38 = (w4*cos(w));
    double temp39 = (6*sin(w));
    double r50 = (2*sin(w));
    double r51 = (6*cos(w));
    double E11 = (-210*x3*y*cos(w))/r9 + temp23/r7 - temp14/r5 + (105*x4*sin(w))/r9 - temp13/r9 - (75*x2*sin(w))/r7 + temp30/r7 + temp39/r5 + temp37/r - w2*((3*z2*sin(w))/r5 - sin(w)/r3) - w2*((-6*x2*sin(w))/r5 + r50/r3);
    double E12 = temp8/r9 + temp31/r7 + temp32/r7 - r51/r5 - temp38/r2 + w2*temp3 + temp15/r9 - temp16/r9 - w2*temp1 - w2*temp2;
    double E13 = -temp9/r9 + temp24/r7 - temp17/r5 + temp18/r9 - temp10/r9 - temp25/r7 + temp19/r5;
    double E21 = temp8/r9 + temp31/r7 + temp32/r7 - r51/r5 - temp38/r + w2*temp3 + temp15/r9 - temp16/r9 - w2*temp1 - w2*temp2;
    double E22 = (-210*x*y3*cos(w))/r9 + temp23/r7 - temp14/r5 + temp13/r9 - (105*y4*sin(w))/r9 - temp33/r7 + (75*y2*sin(w))/r7 - temp39/r5 - temp37/r - w2*((6*y2*sin(w))/r5 - r50/r3) - w2*((-3*z2*sin(w))/r5 + sin(w)/r3);
    double E23 = -temp11/r9 + temp26/r7 - temp20/r5 + temp12/r9 - temp21/r9 + temp27/r7 - temp22/r5;
    double E31 = -temp9/r9 + temp24/r7 - temp17/r5 + temp18/r9 - temp10/r9 - temp25/r7 + temp19/r5;
    double E32 = -temp11/r9 + temp26/r7 - temp20/r5 + temp12/r9 - temp21/r9 + temp27/r7 - temp22/r5;
    double E33 = -(210*x*y*z2*cos(w))/r9 + (30*x*y*cos(w))/r7 + temp14/r5 + (105*x2*z2*sin(w))/r9 - (105*y2*z2*sin(w))/r9 - temp33/r7 + temp30/r7 - w2*((3*x2*sin(w))/r5 - sin(w)/r3) - w2*((-3*y2*sin(w))/r5 + sin(w)/r3);

    double trace = (210 / r9 * ((x3*y*cos(w)) + (x*y3*cos(w)) + (x*y*z2*cos(w)) - r2*x*y*cos(w)))
                  +(6 * w2 * x * y * cos(w) / r5)
                  +(105 * sin(w) / r9 * (y4 - x4 - (x2 * z2) + (y2 * z2) + (r2 * x2) - (r2 * y2)))
                  +(w2 * sin(w) * (3 * y2  - 3 * x2) / r5);

    EM.m[0][0] = -1.0 * E11 - (trace/3);
    EM.m[0][1] = -1.0 * E12;
    EM.m[0][2] = -1.0 * E13;
    EM.m[1][0] = -1.0 * E21;
    EM.m[1][1] = -1.0 * E22 - (trace/3);
    EM.m[1][2] = -1.0 * E23;
    EM.m[2][0] = -1.0 * E31;
    EM.m[2][1] = -1.0 * E32;
    EM.m[2][2] = -1.0 * E33 - (trace/3);
    return EM;
}

mat f234(double x, double y, double z, double w) {
    mat EM;  
    Eigen::Matrix3f E;
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e11;
    Eigen::Matrix3f e12;
    Eigen::Matrix3f e13;
    Eigen::Matrix3f e14;
    Eigen::Matrix3f eAd;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;    
    double r = std::powf(x * x + y * y + z * z,.5);
    double r2 = r*r;
    double r3 = r2*r;
    double r4 = r2*r2;
    double r5 = r3*r2;
    double r6 = r3*r3;
    double w2 = w*w;
    
    double aI = std::cos(w); 

    // Term 1
    e1 << 105 * aI * (x*x*x*x - x*x * y*y) / (r5*r4) + aI * (15 * y*y - 75 * x*x) / (r5*r2) + 6 * aI / r5, 105 * aI * (x*x*x*y - x*y*y*y) / (r5*r4), 105 * aI * (x*x*x*z - x*z * y*y) / (r5*r4) - aI * 30 * x*z / (r5*r2),
          105 * aI * (x*x*x*y - x*y*y*y) / (r5*r4), 105 * aI * (x*x*y*y - y*y*y*y) / (r5*r4) + aI * (75 * y*y - 15 * x*x) / (r5*r2) - 6 * aI / r5, 105 * aI * (x*x*y*z - z*y*y*y) / (r5*r4) + aI * 30 * y*z / (r5*r2),
          105 * aI * (x*x*x*z - x*y*y*z) / (r5*r4) - aI * 30 * x*z / (r5*r2), 105 * aI * (x*x*y*z - z*y*y*y) / (r5*r4) + aI * 30 * y*z / (r5*r2), 105 * aI * (x*x*z*z - z*z*y*y) / (r5*r4) + aI * 15 * (y*y - x*x) / (r5*r2);
    
    // Term 2
    e2 << w2*aI / r3 * (3.0*z*z / r2 - 1), 0.0, -3.0 * aI * w2 * x * z / r5,
          0.0, w2*aI / r3 * (-3.0*z*z / r2 + 1), 3.0 * aI * w2 * y * z / r5, 
          -3.0 * aI * w2 * x * z / r5, 3.0 * aI * w2 * y * z / r5, 3.0 * aI * w2 * (x * x - y * y) / r5;

    // Term 3
    e3 << -w2 * aI * (6.0*x*x / r5 - 2/r3), 0.0, -w2 * aI * 3.0*x*z / r5,
          0.0, -w2 * aI * (-6.0*y*y / r5 + 2/r3), w2 * aI * 3.0*y*z / r5,
          -w2 * aI * 3.0*x*z / r5, w2 * aI * 3.0*y*z / r5, 0;
    // Term 4

    e4 << -1.0 * aI * w2*w2 / r, 0.0, 0.0,
          0.0, 1.0 * aI * w2*w2 / r, 0.0,
          0.0, 0.0, 0.0;

    

    E = e1 - e2 - e3 - e4;


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