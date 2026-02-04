#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

// Define hyperbolic functions
double sech(double x) {
    return 1.0 / cosh(x);
}
double coth(double x) {
    return 1.0 / tanh(x);
}

// Function to compute the matrix
Vector3f f(const Vector3f& r_V, double R, double S, double icity, double factor) {
    Eigen::Vector3f v;

    // Precompute common
    double x = r_V(0);
    double y = r_V(1);
    double z = r_V(2);
    double r = r_V.norm();
    double r2 = r*r;
    double r3 = r2*r;
    
    double secPlus = sech(S*(r+R));
    double secPlus2 = secPlus * secPlus;
    double secMin = sech(S*(r-R));
    double secMin2 = secMin * secMin;
    double tanhPlus = tanh(S*(r+R));
    double tanhMin = tanh(S*(r-R));
    double CRS = coth(R*S);

    double sechMinPlus2 = secMin2 - secPlus2;
    double sechMinPlus2Tanh = (secMin2*tanhMin) - (secPlus2*tanhPlus);  

    double DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r)));
    double DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r))) + (S * -1 * sechMinPlus2 / r));
    double DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r))) + (S * -1 * sechMinPlus2 / r));
    double DyDxB = -.5 * coth(R*S) * S * y * x * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r)));
    double DzDxB = -.5 * coth(R*S) * S * x * z * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r)));

    double vec1;
    double vec2;
    double vec3 = 1.0;

    if (icity == 1.0) {
        vec1 = (-1 * DyDxB * DzDzB - DzDxB * DzDyB - DzDxB * std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB))) / (DzDzB * std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB)));
        vec2 = -DzDyB + std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB)) / DzDzB;
    }
    else if (icity == -1.0) {
        vec1 = (DyDxB * DzDzB + DzDxB * DzDyB - DzDxB * std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB))) / (DzDzB * std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB)));
        vec2 = -DzDyB - std::sqrt(std::abs(DzDyB * DzDyB - DyDyB*DzDzB)) / DzDzB;
    }

    v << -1.0 * factor * vec1, -1.0 * factor * vec2, factor * vec3;

    return v;
}
double eigen_solve_val(const Vector3f& r_V, double R, double S, int icity) {
    // Precompute common
    double x = r_V(0);
    double y = r_V(1);
    double z = r_V(2);
    double r = r_V.norm();
    double r2 = r*r;
    double r3 = r2*r;
    
    double secPlus = sech(S*(r+R));
    double secPlus2 = secPlus * secPlus;
    double secMin = sech(S*(r-R));
    double secMin2 = secMin * secMin;
    double tanhPlus = tanh(S*(r+R));
    double tanhMin = tanh(S*(r-R));
    double CRS = coth(R*S);

    double sechMinPlus2 = secMin2 - secPlus2;
    double sechMinPlus2Tanh = (secMin2*tanhMin) - (secPlus2*tanhPlus);  

    double DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r)));
    double DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r))) + (S * -1 * sechMinPlus2 / r));
    double DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus2 / (r*r*r) + (2 * S * sechMinPlus2Tanh / (r*r))) + (S * -1 * sechMinPlus2 / r));
    
    return icity * .5 * std::sqrt(std::abs((DzDyB*DzDyB) - DyDyB*DzDzB));
}

// Curl Calc
// Compute partial derivative using central difference
Vector3f curlCalc(Vector3f r_V, double icity, double R, double S, double h) {
    Vector3f v;
    Vector3f x_change;
    Vector3f y_change;
    Vector3f z_change;
    x_change << h, 0.0, 0.0;
    y_change << 0.0, h, 0.0;
    z_change << 0.0, 0.0, h;

    Vector3f dF_dx = (f(r_V + x_change, R, S, icity, 1.0) - f(r_V - x_change, R, S, icity, 1.0)) / (2*h);
    Vector3f dF_dy = (f(r_V + y_change, R, S, icity, 1.0) - f(r_V - y_change, R, S, icity, 1.0)) / (2*h);
    Vector3f dF_dz = (f(r_V + z_change, R, S, icity, 1.0) - f(r_V - z_change, R, S, icity, 1.0)) / (2*h);
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

vect rka_iter(double R, double sigma, double vX, double seed_x, double seed_y, double seed_z, int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0, double factor) {    
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
    double h;
    double hAvg = 0;

    for (i = 0; i < num_its; i++) {
        h = h0;
        sizing = true;

        while (sizing) {
            // Compute Runge-Kutta stages
            Vector3f k1 = h * f(r, R, sigma, icity, factor);

            Vector3f k2 = h * f(r + b21 * k1, R, sigma, icity, factor);

            Vector3f k3 = h * f(r + b31 * k1 + b32 * k2, R, sigma, icity, factor);

            Vector3f k4 = h * f(r + b41 * k1 + b42 * k2 + b43 * k3, R, sigma, icity, factor);

            Vector3f k5 = h * f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, R, sigma, icity, factor);

            Vector3f k6 = h * f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, R, sigma, icity, factor);

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

        if (dot < -.95) {
            r_change *= -1;
        }

        r += r_change;
        r_past = r_change / r_mag;
        double val = eigen_solve_val(r, R, sigma, icity);
        Vector3f e_vecs = f(r, R, sigma, icity, factor);
        double e_vecX = e_vecs(0);
        double e_vecY = e_vecs(1);
        double e_vecZ = e_vecs(2);
        Vector3f e_curl = curlCalc(r, icity, R, sigma, h);

        r_change_vect.x[i] = r(0);
        r_change_vect.y[i] = r(1);
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;
        r_change_vect.twist[i] = (e_vecX * e_curl(0)) + (e_vecY * e_curl(1)) + (e_vecZ * e_curl(2));

        hAvg += h;

        if (abs(r.norm()) < ending_tolerance) {
            break;
        }
    
    }
    r_change_vect.hAvg = (hAvg / i);
    r_change_vect.its = i;
    


    return r_change_vect;
}
}

int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
