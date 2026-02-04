#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

Matrix3f f(const Vector3f& r_V, double w) {
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

    

    return e2 + e3 + e4 - e1;

    
}

double eigen_solve_val(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues

    double maxEigenvalue = -1e6; // Large negative value for initialization
    int maxIndex = -1;

    // Loop through eigenvalues to find the largest with the same sign as `icity`
    for (int i = 0; i < 3; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (icity * realVal > 0 && realVal * icity > maxEigenvalue) {
            
            maxEigenvalue = realVal * icity;
            maxIndex = i;
            
        }
    }

    double result;
    if (maxIndex < 0) {
        result = 0.0;
    } else {
        result = eigenvalues(maxIndex).real();
    }

    return result;
    }

// Function to get the eigenvector corresponding to the largest eigenvalue with the given sign
Vector3f eigen_solve(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cf eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors
    Vector3f result;

    double maxEigenvalue = -1e6; // Large negative value for initialization
    int maxIndex = -1;

    // Loop through eigenvalues to find the largest with the same sign as `icity`
    for (int i = 0; i < 3; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (icity * realVal > 0 && realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
        if (realVal > 0 && realVal > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
    }


    // Convert complex eigenvector to real vector (assuming it's real-valued)
    if (maxIndex < 0) {
        result << 0.0,0.0,0.0;
        return result;
    }
    
    result = eigenvectors.col(maxIndex).real();
    
    return result;
}

// Curl Calc
// Compute partial derivative using central difference
Vector3f curlCalc(Vector3f r_V, double icity, double w, double h) {
    Vector3f v;
    Vector3f x_change;
    Vector3f y_change;
    Vector3f z_change;
    x_change << h, 0.0, 0.0;
    y_change << 0.0, h, 0.0;
    z_change << 0.0, 0.0, h;

    Vector3f dF_dx = (eigen_solve(f(r_V + x_change, w), icity) - eigen_solve(f(r_V - x_change, w), icity)) / (2*h);
    Vector3f dF_dy = (eigen_solve(f(r_V + y_change, w), icity) - eigen_solve(f(r_V - y_change, w), icity)) / (2*h);
    Vector3f dF_dz = (eigen_solve(f(r_V + z_change, w), icity) - eigen_solve(f(r_V - z_change, w), icity)) / (2*h);
    double dFx_dy = dF_dy(0);
    double dFx_dz = dF_dz(0);
    double dFy_dx = dF_dx(1);
    double dFy_dz = dF_dz(1);
    double dFz_dx = dF_dx(2);
    double dFz_dy = dF_dy(2);

    v << dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFx_dy - dFy_dx;

    return v;
}


extern "C" {

struct vect {
    double x[10000];
    double y[10000];
    double z[10000];
    double m[10000];
    double twist[10000];
    int its;
    double hAvg;
};

vect rka_iter(double seed_x, double seed_y, double seed_z, 
              int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0, double w) {    
    static const double a2 = 1.0 / 5.0, a3 = 3.0 / 10.0, a4 = 3.0 / 5.0, a5 = 1.0, a6 = 7.0 / 8.0;
    static const double b21 = 1.0 / 5.0;
    static const double b31 = 1.0 / 40.0, b32 = 9.0 / 40.0;
    static const double b41 = 3.0 / 10.0, b42 = -9.0 / 10.0, b43 = 6.0 / 5.0;
    static const double b51 = -11.0 / 54.0, b52 = 5.0/2.0, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0;
    static const double b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0;
    static const double c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0;
    static const double cd1 = 37.0 / 378.0 - 2825.0 / 27648.0, cd3 = 250.0 / 621.0 - 18575.0 / 48384.0, cd4 = 125.0 / 594.0 - 13525.0 / 55296.0, cd5 = -277.0 / 14336.0, cd6 = 512.0 / 1771.0 - 1.0 / 4.0;
    bool sizing = true;
    double hAvg = 0;
    vect r_change_vect;

    Vector3f r = {seed_x, seed_y, seed_z};
    Vector3f r_past = {0.0, 0.0, 0.0};
    Vector3f r_change;
    int i;
    double h;

    for (i = 0; i < num_its; i++) {
        h = h0;
        sizing = true;

        while (sizing) {
            // Compute Runge-Kutta stages
            Matrix3f E1 = f(r, w);
            Vector3f k1 = h * eigen_solve(E1, icity);

            Matrix3f E2 = f(r + b21 * k1, w);
            Vector3f k2 = h * eigen_solve(E2, icity);

            Matrix3f E3 = f(r + b31 * k1 + b32 * k2, w);
            Vector3f k3 = h * eigen_solve(E3, icity);

            Matrix3f E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3, w);
            Vector3f k4 = h * eigen_solve(E4, icity);

            Matrix3f E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, w);
            Vector3f k5 = h * eigen_solve(E5, icity);

            Matrix3f E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, w);
            Vector3f k6 = h * eigen_solve(E6, icity);

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
        hAvg += h;
        
        // Account for sign missmatch
        double r_mag = r_change.norm();
        double dot = r_past.dot(r_change/r_mag);
        double val = eigen_solve_val(f(r, w), icity);
        Vector3f e_vecs = eigen_solve(f(r, w), icity);
        double e_vecX = e_vecs(0);
        double e_vecY = e_vecs(1);
        double e_vecZ = e_vecs(2);
        Vector3f e_curl = curlCalc(r, icity, w, h);
        
        if (dot < -.95) {
            r_change *= -1;
        }

        r += r_change;
        r_past = r_change / r_mag;

        r_change_vect.x[i] = r(0);
        r_change_vect.y[i] = r(1);
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;
        r_change_vect.twist[i] = (e_vecX * e_curl(0)) + (e_vecY * e_curl(1)) + (e_vecZ * e_curl(2));

        if (abs(r.norm()) < ending_tolerance) {
            break;
        }
    
    }

    r_change_vect.its = i;
    r_change_vect.hAvg = (hAvg / i);


    return r_change_vect;
}
}

int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
