#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
#include <fstream>
 
using namespace std;
using namespace Eigen;

std::ofstream outputFile("output.txt");

Matrix3f f(const Vector3f& r_V, double w) {
    Matrix3f result;
    double x = r_V(0);
    double y = r_V(1);
    double z = r_V(2);
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

    result << -1.0 * E11 - (1/3) * trace, -1.0 * E12, -1.0 * E13,
              -1.0 * E21, -1.0 * E22 - (1/3) * trace, -1.0 * E23,
              -1.0 * E31, -1.0 * E32, -1.0 * E33 - (1/3) * trace;
    return result;

}

Matrix3f f23(const Vector3f& r_V, double w) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e11;
    Eigen::Matrix3f e12;
    Eigen::Matrix3f e13;
    Eigen::Matrix3f e14;
    Eigen::Matrix3f eAd;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;
    
    double x = r_V(0);
    double y = r_V(1);
    double z = r_V(2);
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

    return e1 - e2 - e3 - e4;

    
}

double eigen_solve_val(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cf eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors

    double maxEigenvalue = eigenvalues[2].real() * icity; // Large negative value for initialization

    // Loop through eigenvalues to find the largest
    for (int i = 0; i < 2; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
        }
    }
    //outputFile << "EVal:\n" << eigenvalues << "\n EVec\n" << eigenvectors << "\n";
    return maxEigenvalue * icity;
    }

// Function to get the eigenvector corresponding to the largest eigenvalue with the given sign
Vector3f eigen_solve(Matrix3f E_temp, int icity, Vector3f r_past) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cf eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors
    Vector3f result;

    int maxIndex = 0;
    double maxEigenvalue = eigenvalues[maxIndex].real() * icity; // Large negative value for initialization
    

    // Loop through eigenvalues to find the largest
    outputFile << "----Vec----\n"<<eigenvalues << "\n \n" << eigenvectors << '\n';
    
    for (int i = 0; i < 3; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue

        outputFile << std::abs(r_past.dot(eigenvectors.col(i).real())) << '\n';
        
        if (realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
    }

    //std::cout << eigenvalues << '\n' << maxIndex << ' ' << icity << '\n' << maxEigenvalue << '\n' << '\n';
    outputFile << maxIndex << '\n';
    result = eigenvectors.col(maxIndex).real();
    double dot = r_past.dot(result);
    
    if (dot < -.9) {
        //result *= -1;
    }
    outputFile << result << '\n';

    return result ;
}

// Function to get the eigenvector corresponding to the largest eigenvalue with the given sign
Vector3f eigen_solve(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cf eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors
    Vector3f result;

    int maxIndex = 0;
    double maxEigenvalue = eigenvalues[maxIndex].real() * icity; // Large negative value for initialization
    

    // Loop through eigenvalues to find the largest
    outputFile << "----Vec----\n"<<eigenvalues << "\n \n" << eigenvectors << '\n';
    
    for (int i = 0; i < 3; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
    }

    //std::cout << eigenvalues << '\n' << maxIndex << ' ' << icity << '\n' << maxEigenvalue << '\n' << '\n';
    outputFile << maxIndex << '\n';
    result = eigenvectors.col(maxIndex).real();

    outputFile << result << '\n';

    return result ;
}

extern "C" {

struct vect {
    double x[10000];
    double y[10000];
    double z[10000];
    double m[10000];
    int its;
    double hAvg;
};

vect rka_iter(double seed_x, double seed_y, double seed_z, int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0, double w, double factor, double dist) {    
    static const double a2 = 1.0 / 5.0, a3 = 3.0 / 10.0, a4 = 3.0 / 5.0, a5 = 1.0, a6 = 7.0 / 8.0;
    static const double b21 = 1.0 / 5.0;
    static const double b31 = 1.0 / 40.0, b32 = 9.0 / 40.0;
    static const double b41 = 3.0 / 10.0, b42 = -9.0 / 10.0, b43 = 6.0 / 5.0;
    static const double b51 = -11.0 / 54.0, b52 = 5.0/2.0, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0;
    static const double b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0;
    static const double c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0;
    static const double cd1 = 37.0 / 378.0 - 2825.0 / 27648.0, cd3 = 250.0 / 621.0 - 18575.0 / 48384.0, cd4 = 125.0 / 594.0 - 13525.0 / 55296.0, cd5 = -277.0 / 14336.0, cd6 = 512.0 / 1771.0 - 1.0 / 4.0;
    bool sizing = true;
    vect r_change_vect;

    Vector3f r = {seed_x, seed_y, seed_z};
    Vector3f r_past = eigen_solve(f(r, w), icity); //eigen_solve(f(r,w), icity);
    Vector3f r_change;
    int i;
    double h;
    double hAvgTotal = 0;

    for (i = 0; i < num_its; i++) {
        h = h0;
        sizing = true;
        

        while (sizing) {
            // Compute Runge-Kutta stages
            Matrix3f E1 = f(r, w);
            Vector3f k1 = h * factor * eigen_solve(E1, icity, r_past);

            Matrix3f E2 = f(r + b21 * k1, w);
            Vector3f k2 = h * factor * eigen_solve(E2, icity, r_past);

            Matrix3f E3 = f(r + b31 * k1 + b32 * k2, w);
            Vector3f k3 = h * factor * eigen_solve(E3, icity, r_past);

            Matrix3f E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3, w);
            Vector3f k4 = h * factor * eigen_solve(E4, icity, r_past);

            Matrix3f E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, w);
            Vector3f k5 = h * factor * eigen_solve(E5, icity, r_past);

            Matrix3f E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, w);
            Vector3f k6 = h * factor * eigen_solve(E6, icity, r_past);

            Vector3f delta = cd1 * k1 + cd3 * k3 + cd4 * k4 + cd5 * k5 + cd6 * k6;
            
            if ((delta.array().abs() > delta_0).any()) {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.25f);
            } else {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.2f);

                r_change = c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;
                
                // Set step_sizing to false
                sizing = false;
            }
        }

        //std::cout << r_change(0) << ' ' << r_change(1) << " " << '\n';
        
        // Account for sign missmatch
        double r_mag = r_change.norm();
        double dot = r_past.dot(r_change/r_mag);
        outputFile << "R:\n" << r << "\nDot: " << dot << "\n R Past:\n" << r_past << "\nCurrent:\n" << r_change/r_mag << ' ';
        if (dot < -.9) {
            r_change *= -1;
            outputFile << "True";
        }
        //outputFile << "\n_________\n";

        r += r_change;
        r_past = r_change / r_mag;
        double val = eigen_solve_val(f(r, w), icity);

        r_change_vect.x[i] = r(0);
        r_change_vect.y[i] = r(1);
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;

        double rNorm = abs(r.norm());
        hAvgTotal += h;

        if (rNorm < ending_tolerance || rNorm > dist*2) {
            break;
        }
    
    }

    r_change_vect.its = i;
    r_change_vect.hAvg = (hAvgTotal / i);

    outputFile.close();
    return r_change_vect;
}

};

int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
