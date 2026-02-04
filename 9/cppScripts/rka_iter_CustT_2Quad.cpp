#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

Matrix3f f2D(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4; 
    Eigen::Matrix3f E;
    Eigen::Matrix3f eTrace;

    double x0 = r_V(0) - sepX;
    double x1 = r_V(1) - sepY;
    double x2 = r_V(2);
    double r2 = (x0 * x0 + x1 * x1);
    double r5 = std::pow(x0 * x0 + x1 * x1, 2.5);

    double I_diag = (cosT * cosT) - (sinT * sinT);
    double I_off = 2 * cosT * sinT;
    double IpqXpXq = (I_diag * x0 * x0) + 2 * (I_off * x0 * x1) + (-I_diag * x1 * x1);
           
    e1 << I_diag, I_off, 0,
        I_off, -I_diag, 0,
        0, 0, 0;

    e2 << (x0 * x0 * I_diag) + (x0 * x1 * I_off), .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), 0,
            .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), (x0 * x1 * I_off) + (x1 * x1 * -I_diag), 0,
            0, 0, 0;
   
    // Const
    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x0 * x1 * IpqXpXq, 0.0,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, 0.0,
          0.0, 0.0, 0.0;

    E = ((6 * e1) / (r5) - (60 * e2 + 15 * e3) / (r5 * r2) + (105 * e4) / (r5 * r2 * r2));

    double trace = E(0,0) + E(1,1) + E(2,2);
    eTrace << (1/3)*trace, 0.0, 0.0,
              0.0, (1/3)*trace, 0.0,
              0.0, 0.0, (1/3)*trace;
    //double norm = std::pow(r_V.norm(), 5);
    //eNorm << norm, norm, norm, 
    //         norm, norm, norm,
    //         norm, norm, norm;
    E = (E - eTrace);
    //E = E.cwiseProduct(eNorm);
    
    return E;
}

Matrix3f f3D(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;
    Eigen::Matrix3f e5;
    Eigen::Matrix3f E;
    Eigen::Matrix3f eTrace;

    double x0 = r_V(0) + sepX;
    double x1 = r_V(1) + sepY;
    double x2 = r_V(2);
    double r2 = (x0 * x0 + x1 * x1 + x2 * x2);
    double r5 = std::pow(x0 * x0 + x1 * x1 + x2 * x2, 2.5);
    
    double I_diag = (cosT * cosT) - (sinT * sinT);
    double I_off = 2 * cosT * sinT;
    double IpqXpXq = (I_diag * x0 * x0) + 2 * (I_off * x0 * x1) + (-I_diag * x1 * x1);
           
    e1 << I_diag, I_off, 0,
          I_off, -I_diag, 0,
          0, 0, 0;

    e2 << (x0 * x0 * I_diag) + (x0 * x1 * I_off), .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), 0,
          .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), (x0 * x1 * I_off) + (x1 * x1 * -I_diag), 0,
          0, 0, 0;
   
    // Const
    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x1 * x0 * IpqXpXq, x2 * x0 * IpqXpXq,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, x2 * x1 * IpqXpXq,
          x0 * x2 * IpqXpXq, x1 * x2 * IpqXpXq, x2 * x2 * IpqXpXq;

    e5 << 0.0, 0.0, x2 * (I_diag * x0 + I_off * x1),
          0.0, 0.0, x2 * (I_off * x0 - I_diag * x1),
          x2 * (I_diag * x0 + I_off * x1), x2 * (I_off * x0 - I_diag * x1), 0.0;

    E = ((6 * e1) / (r5) - (60 * e2 + 30 * e5 + 15 * e3) / (r5 * r2) + (105 * e4) / (r5 * r2 * r2));


    double trace = E(0,0) + E(1,1) + E(2,2);
    eTrace << (1/3)*trace, 0.0, 0.0,
              0.0, (1/3)*trace, 0.0,
              0.0, 0.0, (1/3)*trace;

    E = (E - eTrace);
    
    return E;
}

Matrix3f f(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;

    e1 = f3D(r_V, sepX, 0, 0, 1);
    e2 = f3D(r_V, -sepX, 0, sinT, cosT);
    //e3 = f3D(r_V, 0, sepY, sinT, cosT);
    //e4 = f3D(r_V, 0, -sepY, sinT, cosT);
    
    return e1+e2;//+e3+e4;
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

    if (maxIndex == 2) {
        
    }

    if (maxIndex < 0) {
        return 0.0;
    } 

    //double determiniant = E_temp.determinant();

    //std::cout << eigenvalues(maxIndex).real() << '\n';

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
    double hArray[10000];
    double det[10000];
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
            Matrix3f E1 = f(r, sepX, sepY, sinT, cosT);
            Vector3f k1 = h * eigen_solve(E1, icity);

            Matrix3f E2 = f(r + b21 * k1, sepX, sepY, sinT, cosT);
            Vector3f k2 = h * eigen_solve(E2, icity);

            Matrix3f E3 = f(r + b31 * k1 + b32 * k2, sepX, sepY, sinT, cosT);
            Vector3f k3 = h * eigen_solve(E3, icity);
            
            Matrix3f E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3, sepX, sepY, sinT, cosT);
            Vector3f k4 = h * eigen_solve(E4, icity);

            Matrix3f E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, sepX, sepY, sinT, cosT);
            Vector3f k5 = h * eigen_solve(E5, icity);
            
            Matrix3f E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, sepX, sepY, sinT, cosT);
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
        double determiniant = 0.0;
        double val;
        val = eigen_solve_val(f(r, sepX, sepY, sinT, cosT), icity);

        double x = r(0);
        double y = r(1);

        //std::cout << r(2) << "\n";

        //std::cout << x << " " << y << " " << dot << "\n";

        r_change_vect.x[i] = x;
        r_change_vect.y[i] = y;
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;
        r_change_vect.hArray[i] = h;
        r_change_vect.det[i] = determiniant;

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
