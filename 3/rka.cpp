#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

// Define the ODE function
Matrix3f f(Vector3f r) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;
    float x0 = r(0);
    float x1 = r(1);
    float x2 = r(2);
    float rM = sqrt((x0 * x0) + (x1 * x1) + (x2 * x2));
    
    float aI = 1.0;
    float IpqXpXq = (aI * (x0 * x0 - x1 * x1));

    e1 << aI, 0.0, 0.0,
        0.0, -1 * aI, 0.0,
        0.0, 0.0, 0.0;

    e2 << 2 * aI * x0 * x0, 0.0, 0.0,
          0.0, -2 * aI * x1 * x1, 0.0,
          0.0, 0.0, 0.0;

    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x0 * x1 * IpqXpXq, 0.0,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, 0.0,
          0.0, 0.0, 0.0;
    return (-6 * e1) + (30 * e2 + 15 * e3) * (1.0 / (rM * rM)) + (-105 * e4) * (1.0 / (rM * rM * rM * rM));

    
}

// Function to get the eigenvector corresponding to the largest eigenvalue with the given sign
Vector3f eigen_solve(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cf eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors

    float maxEigenvalue = -1e6; // Large negative value for initialization
    int maxIndex = -1;

    // Loop through eigenvalues to find the largest with the same sign as `icity`
    for (int i = 0; i < 3; ++i) {
        float realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (icity * realVal > 0 && realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
    }

    // Convert complex eigenvector to real vector (assuming it's real-valued)
    Vector3f result = eigenvectors.col(maxIndex).real();

    return result;
}

extern "C" {

struct vect {
    float x;
    float y;
    float z;
};

// Adaptive Runge-Kutta-Fehlberg (RK45) implementation
vect rka(float v1, float v2, float v3, int icity) {
    Vector3f r = {v1, v2, v3};
    float delta_0 = .0004;
    float safety = .95;
    float h = .0001;
    Vector3f r_change;
    
    // Butcher tableau for RK45
    static const float a2 = 1.0 / 5.0, a3 = 3.0 / 10.0, a4 = 3.0 / 5.0, a5 = 1.0, a6 = 7.0 / 8.0;
    static const float b21 = 1.0 / 5.0;
    static const float b31 = 1.0 / 40.0, b32 = 9.0 / 40.0;
    static const float b41 = 3.0 / 10.0, b42 = -9.0 / 10.0, b43 = 6.0 / 5.0;
    static const float b51 = -11.0 / 54.0, b52 = -5.0/2.0, b53 = -70.0 / 27.0, b54 = -35.0 / 27.0;
    static const float b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0;
    static const float c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 621.0, c6 = 512.0 / 1771.0;
    static const float cd1 = 37.0 / 378.0 - 2825.0 / 27648.0, cd3 = 250.0 / 621.0 - 18575.0 / 48384.0, cd4 = 125.0 / 621.0 - 13525.0 / 55296.0, cd5 = -277.0 / 14336.0, cd6 = 512.0 / 1771.0 - 1.0 / 4.0;
    bool sizing = true;

    while (sizing) {// Butcher tableau for RK45
        // Compute Runge-Kutta stages
        Matrix3f E1 = f(r);
        Vector3f k1 = h * eigen_solve(E1, icity);

        Matrix3f E2 = f(r + b21 * k1);
        Vector3f k2 = h * eigen_solve(E2, icity);

        Matrix3f E3 = f(r + b31 * k1 + b32 * k2);
        Vector3f k3 = h * eigen_solve(E3, icity);

        Matrix3f E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3);
        Vector3f k4 = h * eigen_solve(E4, icity);

        Matrix3f E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4);
        Vector3f k5 = h * eigen_solve(E5, icity);

        Matrix3f E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5);
        Vector3f k6 = h * eigen_solve(E6, icity);

        Vector3f delta = cd1 * k1 + cd3 * k3 + cd4 * k4 + cd5 * k5 + cd6 * k6;
        
        if ((delta.array().abs() > delta_0).any()) {
            h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.25f);
        } else {
            h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.2f);

            // Assuming the calculations for r_change involve variables like k1, k3, k4, k6
            r_change = c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;
            
            // Set step_sizing to false
            sizing = false;
        }
    }

    vect r_change_vect;

    r_change_vect.x = r_change(0);
    r_change_vect.y = r_change(1);
    r_change_vect.z = r_change(2);

    return r_change_vect;
}

};


int main() {
    Vector3f r = {1,1,1};  // Initial condition y(0) = 1
    int icity = 1;
    float v1;

    rka(1.0, 1.0, 1.0, icity);


    
    //runge_kutta_adaptive(r0, icity);
    return 0;
}
