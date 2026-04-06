#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

struct vis_params {
    double M;
    double omega1;
    double omega2;
    double t;
    double C2;
    double w2;
};

struct vect {
    double x[10000];
    double y[10000];
    double z[10000];
    double m[10000];
    int its;
};

struct mat {
    double m11;
    double m12;
    double m21;
    double m22;
};

Matrix3f EleMQuad(const Vector3f& r_V, double M, double C, double omega, double t, double w) {
    Eigen::Matrix3f E;

    double r_val = r_V(0);
    double theta = r_V(1);
    double phi = r_V(2);

    double ret = t-r_val;
    
    double a = (C / (2*r_val)) * M * (omega * omega * omega * omega) * (1+pow(cos(theta), 2)) * cos(2 * (phi - w - (ret * omega))) * (1 + pow(sin(theta),4));
    double b = (C / (2*r_val)) * 2 * M * (omega * omega * omega * omega) * cos(theta) * sin(2 * (phi - w - (ret * omega))) * (1 + pow(sin(theta),4));
    

    E << 0, 0, 0,
         0, -a, b,
         0, b, a;

    return E;
}

Matrix3f EleCQuad(const Vector3f& r_V, double M, double C, double omega, double t, double w) {
    Eigen::Matrix3f E;

    double r_val = r_V(0);
    double theta = r_V(1);
    double phi = r_V(2);

    double ret = t-r_val;
    
    double a = (8 / (3*r_val)) * C * M * (omega * omega * omega * omega) * cos(theta) * sin(theta) * sin(theta) * sin(2 * (phi - w - (ret * omega)));
    double b = (-4 / (3*r_val)) * C * M * (omega * omega * omega * omega) * (1 + pow(cos(theta), 2)) * sin(theta) * sin(theta) * cos(2 * (phi - w - (ret * omega)));
    

    E << 0, 0, 0,
         0, -a, b,
         0, b, a;

    return E;
}

Matrix3f EleCOcto(const Vector3f& r_V, double M, double C, double omega, double t, double w) {
    Eigen::Matrix3f E;

    double r_val = r_V(0);
    double theta = r_V(1);
    double phi = r_V(2);
    double v = omega;

    double ret = t-r_val;

    double a = M * v * pow(omega,5) * cos(2 * theta) * cos(2 * (phi - w - (omega * ret))) * r_val * r_val * abs(sin(theta));
    double b = -M * v * pow(omega,5) * (5*cos(theta) + 3*cos(3*theta)) * r_val * r_val * abs(sin(theta)) * sin(2 * (phi - w - (omega * ret)));

    E << 0, 0, 0,
         0, a, b,
         0, b, -a;

    return E;
}

Matrix3f MagMQuad(const Vector3f& r_V, double M, double C, double omega, double t, double w) {
    Eigen::Matrix3f E;

    double r_val = r_V(0);
    double theta = r_V(1);
    double phi = r_V(2);

    double ret = t-r_val;
    
    double a = (C / (2*r_val)) * M * (omega * omega * omega * omega) * (1+pow(cos(theta), 2)) * cos(2 * (phi - w - (ret * omega))) * (1 + pow(sin(theta),4));
    double b = (C / (2*r_val)) * 2 * M * (omega * omega * omega * omega) * cos(theta) * sin(2 * (phi - w - (ret * omega))) * (1 + pow(sin(theta),4));
    

    E << 0, 0, 0,
         0, -a, b,
         0, b, a;

    return E;
}

Matrix3f MagCQuad(const Vector3f& r_V, double M, double C, double omega, double t, double w) {
    Eigen::Matrix3f E;

    double r_val = r_V(0);
    double theta = r_V(1);
    double phi = r_V(2);

    double ret = t-r_val;
    
    double a = (8 / (3*r_val)) * C * M * (omega * omega * omega * omega) * cos(theta) * sin(theta) * sin(theta) * sin(2 * (phi - w - (ret * omega)));
    double b = (-4 / (3*r_val)) * C * M * (omega * omega * omega * omega) * (1 + pow(cos(theta), 2)) * sin(theta) * sin(theta) * cos(2 * (phi - w - (ret * omega)));
    

    E << 0, 0, 0,
         0, -a, b,
         0, b, a;

    return E;
}

Matrix3f f(const Vector3f& r_V, vis_params vis_params) {
    double M = vis_params.M;
    double C = vis_params.C2;
    double omega1 = vis_params.omega1;
    double omega2 = vis_params.omega2;
    double t = vis_params.t;
    double t = r_V(0);
    double w = vis_params.w2;

    return EleMQuad(r_V, M, 1, omega1, t, 0) + EleCQuad(r_V, M, C, omega2, t, w);
    //return EleCOcto(r_V, M, C, omega2, t, w);
}

Matrix3f fB(const Vector3f& r_V, vis_params vis_params) {
    double M = vis_params.M;
    double C = vis_params.C2;
    double omega1 = vis_params.omega1;
    double omega2 = vis_params.omega2;
    double t = vis_params.t;
    //double t = r_V(0);
    double w = vis_params.w2;

    return MagMQuad(r_V, M, 1, omega1, t, 0) + MagCQuad(r_V, M, C, omega2, t, w);
    //return EleCOcto(r_V, M, C, omega2, t, w);
}

double eigen_solve_val(Matrix3f E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3f> solver(E_temp);
    Vector3cf eigenvalues = solver.eigenvalues();   // Complex eigenvalues

    double result = eigenvalues.real().maxCoeff();
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
    }
    // Convert complex eigenvector to real vector (assuming it's real-valued)
    if (maxIndex < 0) {
        result << 0.0,0.0,0.0;
        return result;
    } 
    
    result = eigenvectors.col(maxIndex).real();
    
    return result / result.norm();
}

Vector3f super_poynting(const Vector3f& r_V, vis_params vis_params) {
    Matrix3f E;
    Matrix3f B;
    E = f(r_V, vis_params);
    B = fB(r_V, vis_params);

    double E11 = E[0,0];
    double E12 = E[0,1];
    double E13 = E[0,2];
    double E21 = E[1,0];
    double E22 = E[1,1];
    double E23 = E[1,2];
    double E31 = E[2,0];
    double E32 = E[2,1];
    double E33 = E[2,2];

    double B11 = B[0,0];
    double B12 = B[0,1];
    double B13 = B[0,2];
    double B21 = B[1,0];
    double B22 = B[1,1];
    double B23 = B[1,2];
    double B31 = B[2,0];
    double B32 = B[2,1];
    double B33 = B[2,2];

    
}

extern "C" {

double val_return(double x, double y, double z, int icity, vis_params vals, int spherical= 0) {
    Vector3f r;
    if (spherical == 1) {
        r = {x, y, z};
    } else {
        double r_val = sqrt(x*x + y*y + z*z);
        if (abs(r_val) < 1e-6) {
            return 0.0;
        }
        r = {r_val, acos(z / r_val), atan2(y,x)};
    }

    return eigen_solve_val(f(r, vals), icity);
}

vect rka_iter(double seed_r, double seed_theta, double seed_phi, int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0, vis_params vals) {    
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

    Vector3f r = {seed_r, seed_theta, seed_phi};
    Vector3f r_past = {0.0, 0.0, 0.0};
    Vector3f r_change;
    int i;
    double h;

    for (i = 0; i < num_its; i++) {
        h = h0;
        sizing = true;

        while (sizing) {
            // Compute Runge-Kutta stages
            Matrix3f E1 = f(r, vals);
            Vector3f k1 = h * eigen_solve(E1, icity);

            Matrix3f E2 = f(r + b21 * k1, vals);
            Vector3f k2 = h * eigen_solve(E2, icity);

            Matrix3f E3 = f(r + b31 * k1 + b32 * k2, vals);
            Vector3f k3 = h * eigen_solve(E3, icity);

            Matrix3f E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3, vals);
            Vector3f k4 = h * eigen_solve(E4, icity);

            Matrix3f E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, vals);
            Vector3f k5 = h * eigen_solve(E5, icity);

            Matrix3f E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, vals);
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

        //std::cout << r_change(0) << ' ' << r_change(1) << " " << '\n';
        
        // Account for sign missmatch
        double r_mag = r_change.norm();
        double dot = r_past.dot(r_change/r_mag);

        if (dot < -.95) {
            r_change *= -1;
        }

        r += r_change;
        r_past = r_change / r_mag;
        double val = eigen_solve_val(f(r, vals), icity);

        r_change_vect.x[i] = r(0);
        r_change_vect.y[i] = r(1);
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;

        if (abs(val) < ending_tolerance) {
            break;
        }
    
    }

    r_change_vect.its = i;


    return r_change_vect;
}

mat test_val(double x, double y, double z, vis_params vals, int spherical = 0) {
    Vector3f r;
    mat m;
    m.m11 = 0.0;
    m.m12 = 0.0;
    m.m21 = 0.0;
    m.m22 = 0.0;

    if (spherical == 0) {
        double r_val = sqrt(x*x + y*y + z*z);
        if (abs(r_val) < 1e-6) {
            r = {0,0,0};
        }
        else {
            r = {r_val, acos(z / r_val), atan2(y,x)};
        }
        
    }
    else {
        r = {x, y, z};
    }

    
    Matrix3f m_temp = f(r, vals);

    m.m11 = m_temp(1,1);
    m.m12 = m_temp(1,2);
    m.m21 = m_temp(2,1);
    m.m22 = m_temp(2,2);

    return m;
}

vect singular_find(double r_val, int icity, double ending_tolerance, double delta_0, double safety,
    double h0, vis_params model_param, double dPhi, double dTheta, int section_it, double phi_min,
    double phi_max, double theta_min, double theta_max
) {
    int count = 0;
    vect r;

    for (double t = theta_min; t <= theta_max; t += dTheta) {
        for (double p = phi_min; p <= phi_max; p += dPhi) {

            double rad[2] = {0.0, M_PI};
            double pointsPhi[2] = {
                p + 0.5 * dPhi * std::cos(rad[0]),
                p + 0.5 * dPhi * std::cos(rad[1])
            };
            double pointsTheta[2] = {
                t + 0.5 * dTheta * std::sin(rad[0]),
                t + 0.5 * dTheta * std::sin(rad[1])
            };

            std::vector<std::pair<double,double>> dots;

            for (int i = 0; i < 2; ++i) {
                vect v = rka_iter(
                    r_val,
                    pointsTheta[i],
                    pointsPhi[i],
                    section_it,
                    icity,
                    ending_tolerance,
                    delta_0,
                    safety,
                    h0,
                    model_param
                );

                if (v.its > section_it - 1) {
                    double y0 = v.y[0];
                    double x0 = v.z[0];
                    dots.emplace_back(y0, x0);
                }
            }

            if (dots.size() > 0) {
                double v0x = dots.front().first;
                double v0y = dots.front().second;
                double v1x = dots.back().first;
                double v1y = dots.back().second;

                double dot =
                    (v0x*v1x + v0y*v1y) /
                    (std::hypot(v0x, v0y) * std::hypot(v1x, v1y));

                if (dot > 1.0) dot = 1.0;
                if (dot < -1.0) dot = -1.0;

                double angle = std::acos(dot);

                if (std::abs(angle) == 0.0) {
                    r.x[count] = p;
                    r.y[count] = t;
                    count++;
                }
            }
        }
    }

    
    r.its = count;
    return r;
}

};

int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
