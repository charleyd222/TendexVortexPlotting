#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

Matrix3f f2(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e01;
    Eigen::Matrix3f e02;
    Eigen::Matrix3f e03;
    Eigen::Matrix3f e04;
    Eigen::Matrix3f e11;
    Eigen::Matrix3f e12;
    Eigen::Matrix3f e13;
    Eigen::Matrix3f e14;    
    Eigen::Matrix3f e21;
    Eigen::Matrix3f e22;
    Eigen::Matrix3f e23;
    Eigen::Matrix3f e24;   
    Eigen::Matrix3f e31;
    Eigen::Matrix3f e32;
    Eigen::Matrix3f e33;
    Eigen::Matrix3f e34;  

    //      0
    //  3       1
    //      2

    double x00 = r_V(0);
    double x01 = r_V(1) + sepY;
    double x02 = r_V(2);
    double r02 = (x00 * x00 + x01 * x01);
    double r05 = std::pow(x00 * x00 + x01 * x01, 2.5);
    
    double x10 = r_V(0) + sepX;
    double x11 = r_V(1);
    double x12 = r_V(2);
    double r12 = (x10 * x10 + x11 * x11);
    double r15 = std::pow(x10 * x10 + x11 * x11, 2.5);

    double x20 = r_V(0);
    double x21 = r_V(1) - sepY;
    double x22 = r_V(2);
    double r22 = (x20 * x20 + x21 * x21);
    double r25 = std::pow(x20 * x20 + x21 * x21, 2.5);

    double x30 = r_V(0) - sepX;
    double x31 = r_V(1);
    double x32 = r_V(2);    
    double r32 = (x30 * x30 + x31 * x31);
    double r35 = std::pow(x30 * x30 + x31 * x31, 2.5);
    
    double IpqXpXq0 = ((x00 * x00 - x01 * x01));


    e01 << 1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 0.0;
    
    e02 << (x00 * x00), 0.0, 0.0,
        0.0, -1.0 * (x01 * x01), 0.0,
        0.0, 0.0, 0.0;

    e03 << IpqXpXq0, 0.0, 0.0,
        0.0, IpqXpXq0, 0.0,
        0.0, 0.0, IpqXpXq0;

    e04 << x00 * x00 * IpqXpXq0, x00 * x01 * IpqXpXq0, 0.0,
        x00 * x01 * IpqXpXq0, x01 * x01 * IpqXpXq0, 0.0,
        0.0, 0.0, 0.0;
    
    double I11 = (cosT * cosT) - (sinT * sinT);
    double I12 = 2 * cosT * sinT;
    double I22 = (sinT * sinT) - (cosT * cosT);
    double IpqXpXq1 = (I11 * x10 * x10) + 2 * (I12 * x10 * x11) + (I22 * x11 * x11);
           
    e11 << I11, I12, 0,
        I12, I22, 0,
        0, 0, 0;

    e12 << (x10 * x10 * I11) + (x10 * x11 * I12), .5 * x10 * (x10 * I12  + x11 * I11) + .5 * x11 * (x10 * I22  + x11 * I12), 0,
            .5 * x10 * (x10 * I12  + x11 * I11) + .5 * x11 * (x10 * I22  + x11 * I12), (x10 * x11 * I12) + (x11 * x11 * I22), 0,
            0, 0, 0;
   
    // Const
    e13 << IpqXpXq1, 0.0, 0.0,
            0.0, IpqXpXq1, 0.0,
            0.0, 0.0, IpqXpXq1;

    e14 << x10 * x10 * IpqXpXq1, x10 * x11 * IpqXpXq1, 0.0,
            x10 * x11 * IpqXpXq1, x11 * x11 * IpqXpXq1, 0.0,
            0.0, 0.0, 0.0;

    double IpqXpXq2 = (I11 * x20 * x20) + 2 * (I12 * x20 * x21) + (I22 * x21 * x21);
                   
    e21 << I11, I12, 0,
        I12, I22, 0,
        0, 0, 0;

    e22 << (x20 * x20 * I11) + (x20 * x21 * I12), .5 * x20 * (x20 * I12  + x21 * I11) + .5 * x21 * (x20 * I22  + x21 * I12), 0,
            .5 * x20 * (x20 * I12  + x21 * I11) + .5 * x21 * (x20 * I22  + x21 * I12), (x20 * x21 * I12) + (x21 * x21 * I22), 0,
            0, 0, 0;
    
    e23 << IpqXpXq2, 0.0, 0.0,
            0.0, IpqXpXq2, 0.0,
            0.0, 0.0, IpqXpXq2;

    e24 << x20 * x20 * IpqXpXq2, x20 * x21 * IpqXpXq2, 0.0,
           x20 * x21 * IpqXpXq2, x21 * x21 * IpqXpXq2, 0.0,
           0.0, 0.0, 0.0;

    double IpqXpXq3 = (I11 * x30 * x30) + 2 * (I12 * x30 * x31) + (I22 * x31 * x31);
            
    e21 << I11, I12, 0,
           I12, I22, 0,
           0, 0, 0;

    e22 << (x30 * x30 * I11) + (x30 * x31 * I12), .5 * x30 * (x30 * I12  + x31 * I11) + .5 * x31 * (x30 * I22  + x31 * I12), 0,
            .5 * x30 * (x30 * I12  + x31 * I11) + .5 * x31 * (x30 * I22  + x31 * I12), (x30 * x31 * I12) + (x31 * x31 * I22), 0,
            0, 0, 0;
    
    e23 << IpqXpXq3, 0.0, 0.0,
            0.0, IpqXpXq3, 0.0,
            0.0, 0.0, IpqXpXq3;

    e24 << x30 * x30 * IpqXpXq3, x30 * x31 * IpqXpXq3, 0.0,
            x30 * x31 * IpqXpXq3, x31 * x31 * IpqXpXq3, 0.0,
            0.0, 0.0, 0.0;
    

    return ((-6 * e01) / (r05) + (60 * e02 + 15 * e03) / (r05 * r02) + (-105 * e04) / (r05 * r02 * r02)) +
           ((-6 * e11) / (r15) + (60 * e12 + 15 * e13) / (r15 * r12) + (-105 * e14) / (r15 * r12 * r12)) +
           ((-6 * e21) / (r25) + (60 * e22 + 15 * e23) / (r25 * r22) + (-105 * e24) / (r25 * r22 * r22)) + 
           ((-6 * e31) / (r35) + (60 * e32 + 15 * e33) / (r35 * r32) + (-105 * e34) / (r35 * r32 * r32));

    
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

    if (maxIndex < 0) {
        
        return 0.0;
    } 

    return eigenvalues(maxIndex).real();
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
    }
    // Convert complex eigenvector to real vector (assuming it's real-valued)
    if (maxIndex < 0) {
        result << 0.0,0.0,0.0;
        return result;
    } 
    
    result = eigenvectors.col(maxIndex).real();
    
    return result / result.norm();
}

extern "C" {

struct vect {
    double x[10000];
    double y[10000];
    double z[10000];
    double m[10000];
    int its;
};

vect rka_iter_double(double t, double sepX, double sepY, double seed_x, double seed_y, double seed_z, int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0) {    
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
    Vector3f r_past = {0.0, 0.0, 0.0};
    Vector3f r_change;
    int i;
    double h = h0;
    double deltNorm;
    double sinT = std::sin(t);
    double cosT = std::cos(t);
    for (i = 0; i < num_its; i++) {
        sizing = true;

        while (sizing) {
            // Compute Runge-Kutta stages
            Matrix3f E1 = f2(r, sepX, sepY, sinT, cosT);
            Vector3f k1 = h * eigen_solve(E1, icity);

            Matrix3f E2 = f2(r + b21 * k1, sepX, sepY, sinT, cosT);
            Vector3f k2 = h * eigen_solve(E2, icity);

            Matrix3f E3 = f2(r + b31 * k1 + b32 * k2, sepX, sepY, sinT, cosT);
            Vector3f k3 = h * eigen_solve(E3, icity);
            
            Matrix3f E4 = f2(r + b41 * k1 + b42 * k2 + b43 * k3, sepX, sepY, sinT, cosT);
            Vector3f k4 = h * eigen_solve(E4, icity);

            Matrix3f E5 = f2(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, sepX, sepY, sinT, cosT);
            Vector3f k5 = h * eigen_solve(E5, icity);
            
            Matrix3f E6 = f2(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, sepX, sepY, sinT, cosT);
            Vector3f k6 = h * eigen_solve(E6, icity);

            Vector3f delta = cd1 * k1 + cd3 * k3 + cd4 * k4 + cd5 * k5 + cd6 * k6;

            deltNorm = delta.norm();

            if (deltNorm == 0.0) {
                r_change = delta;
                break;
                
            } else if ((delta.array().abs() > delta_0).any()) {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.25f);
            } else {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.2f);

                r_change = c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;
                
                // Set step_sizing to false
                sizing = false;
            }
        }
        if (deltNorm == 0.0) {
            break;
        }

        // Account for sign missmatch
        double r_mag = r_change.norm();
        double dot = r_past.dot(r_change/r_mag);

        if (dot < -.8) {
            r_change *= -1;
        }

        r += r_change;
        r_past = r_change / r_mag;
        double val = eigen_solve_val(f2(r, sepX, sepY, sinT, cosT), icity);

        double x = r(0);
        double y = r(1);

        //std::cout << x << " " << y << " " << dot << "\n";

        r_change_vect.x[i] = x;
        r_change_vect.y[i] = y;
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;

        if (abs((x + sepX)*(x+sepX) + (y + sepY)*(y+sepY)) < ending_tolerance || (x*x + y*y) < ending_tolerance) {
            break;
        }
    
    }

    r_change_vect.its = i;


    return r_change_vect;
}

};

int main() {
    return 0;
}
