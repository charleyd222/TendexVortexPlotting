#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

Vector3f f3D(const Eigen::Vector3f& r) {
    double r_mag = r.norm();
    // Dipole moment along z axis
    Eigen::Vector3f p(0., 1e-4, 0.);

    // Avoid division by zero near the dipole location
    if (r_mag < 1e-12)
        return Eigen::Vector3f::Zero();

    Eigen::Vector3f r_hat = r / r_mag;

    double k = 1.;

    Eigen::Vector3f E = k * (3 * r * r.dot(p) - p * r_mag * r_mag) / (r_mag*r_mag*r_mag*r_mag*r_mag);

    return E;
}


Vector3f f(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Vector3f e1;
    Eigen::Vector3f e2;

    e1 = f3D(r_V);//, sepX, 0, 0, 1);
//    e2 = f3D(r_V);//, -sepX, 0, 0, 1);
    
    return e1;//+e2;//+e3+e4;
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
            Vector3f k1 = h * f(r, sepX, sepY, sinT, cosT);
            Vector3f k2 = h * f(r + b21 * k1, sepX, sepY, sinT, cosT);
            Vector3f k3 = h * f(r + b31 * k1 + b32 * k2, sepX, sepY, sinT, cosT);
            Vector3f k4 = h * f(r + b41 * k1 + b42 * k2 + b43 * k3, sepX, sepY, sinT, cosT);
            Vector3f k5 = h * f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, sepX, sepY, sinT, cosT);
            Vector3f k6 = h * f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, sepX, sepY, sinT, cosT);

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
        val = f(r, sepX, sepY, sinT, cosT).norm();

        double x = r(0);
        double y = r(1);

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
